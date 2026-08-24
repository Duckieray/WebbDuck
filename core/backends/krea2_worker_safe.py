"""Safety/compatibility wrapper for the Krea 2 isolated worker.

The main Krea worker owns checkpoint loading and the model-specific execution
primitives. This wrapper keeps the hardware policy easy to audit while we
validate constrained-GPU behavior on real machines:

* conditioning is explicitly cast to the selected inference dtype before it is
  reused by Diffusers, preventing accidental FP32 latents/attention;
* constrained GPUs reserve additional transient memory before choosing resident
  execution;
* CUDA OOM recovery uses fresh denoiser instances and progressively safer
  profiles so stale offload hooks cannot contaminate a retry.

Once this policy has enough hardware coverage it can be folded back into the
base worker without changing the public generation API.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backends import krea2_worker as base
from runtime_hardware import detect_torch_hardware


def _selected_dtype(hardware: dict[str, Any]) -> torch.dtype:
    device = base._torch_device(hardware)
    if device == "cuda":
        return torch.bfloat16 if bool(hardware.get("bf16")) else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _coerce_conditioning_dtype(
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep masks boolean, but force floating conditioning to inference dtype."""
    prompt_embeds = prompt_embeds.to(device="cpu", dtype=dtype)
    if negative_embeds is not None:
        negative_embeds = negative_embeds.to(device="cpu", dtype=dtype)
    return prompt_embeds, negative_embeds


def _memory_reserve_gb(
    hardware: dict[str, Any],
    width: int,
    height: int,
) -> float:
    """Reserve room for attention, FP8 dequant temporaries and desktop jitter."""
    total_vram_gb = float(hardware.get("total_vram_gb") or 0.0)
    reserve = base._activation_reserve_gb(total_vram_gb, width, height)

    # 16/18 GB-class desktop GPUs are especially sensitive to compositor and
    # browser VRAM plus Krea's transient SDPA/dequant allocations. Real hardware
    # showed a 1.22 GiB attention allocation after the worker had already
    # consumed ~11 GiB, so a 2 GiB theoretical reserve is not sufficient.
    if base._torch_device(hardware) == "cuda" and 0 < total_vram_gb <= 18.5:
        reserve = max(reserve, 4.0)

    explicit = str(os.getenv("WEBBDUCK_KREA2_VRAM_RESERVE_GB") or "").strip()
    if explicit:
        try:
            reserve = max(0.5, float(explicit))
        except ValueError:
            pass
    return reserve


def _retry_profiles(current_mode: str, hardware: dict[str, Any]) -> list[tuple[str, bool]]:
    """Return strictly more conservative (mode_override, stream_prefetch) retries."""
    retries: list[tuple[str, bool]] = []
    current = str(current_mode or "")

    # Resident and streamed block modes both fall first to synchronous block
    # offload. Streams prefetch the next layer and intentionally cost extra VRAM.
    if current.startswith("resident") or "block-stream" in current:
        retries.append(("transformer-block", False))

    # Any non-leaf, non-sequential profile can then fall to synchronous leaf
    # offload. Never select leaf again when it is already the current profile.
    if "leaf" not in current and current != "sequential":
        retries.append(("group", False))

    if current != "sequential":
        retries.append(("sequential", False))

    deduped: list[tuple[str, bool]] = []
    for item in retries:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _dispose_pipe(pipe: Any, device: str) -> None:
    if pipe is None:
        return
    transformer = getattr(pipe, "transformer", None)
    try:
        if transformer is not None:
            transformer.to("cpu")
    except Exception:
        pass
    try:
        pipe.transformer = None
    except Exception:
        pass
    try:
        pipe.text_encoder = None
    except Exception:
        pass
    try:
        pipe.vae = None
    except Exception:
        pass
    del transformer
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _fresh_denoiser(
    request: dict[str, Any],
    *,
    dtype: torch.dtype,
    hardware: dict[str, Any],
    reserve_gb: float,
    mode_override: str,
    stream_prefetch: bool,
    report: Any,
) -> tuple[Any, dict[str, Any], str, float]:
    """Reload Krea CPU-side and configure a clean, safer denoiser profile."""
    report("Reloading Krea denoiser for safe retry", 0.53)
    fresh_hardware = dict(hardware)
    fresh_hardware["stream_prefetch"] = bool(stream_prefetch)
    fresh_pipe, fresh_load_info = base._load_pipeline(
        request,
        dtype,
        fresh_hardware,
        report=None,
    )

    # Conditioning was already computed. Do not keep duplicate support models
    # around during the retry denoiser phase.
    retry_text_encoder = getattr(fresh_pipe, "text_encoder", None)
    retry_vae = getattr(fresh_pipe, "vae", None)
    fresh_pipe.text_encoder = None
    fresh_pipe.vae = None
    del retry_text_encoder, retry_vae
    gc.collect()
    if base._torch_device(fresh_hardware) == "cuda":
        torch.cuda.empty_cache()
        fresh_hardware = detect_torch_hardware(torch)
        fresh_hardware["stream_prefetch"] = bool(stream_prefetch)

    storage_gb = base._module_storage_gb(fresh_pipe.transformer)
    mode = base._configure_execution(
        fresh_pipe,
        hardware=fresh_hardware,
        load_info=fresh_load_info,
        transformer_storage_gb=storage_gb,
        reserve_gb=reserve_gb,
        mode_override=mode_override,
    )
    return fresh_pipe, fresh_load_info, mode, storage_gb


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    worker_started = time.perf_counter()

    prompt = str(request["prompt"])
    width = base._snap(int(request.get("width") or 1024))
    height = base._snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 28))
    guidance = float(request.get("guidance") if request.get("guidance") is not None else 4.5)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") or 0)
    timing: dict[str, float] = {}

    def report(stage: str, progress: float, step: int = 0, total_steps: int = 0) -> None:
        base._write_progress(progress_path, stage, progress, step, total_steps)

    report("Profiling Krea hardware", 0.04)
    hardware = detect_torch_hardware(torch)
    device = base._torch_device(hardware)
    dtype = _selected_dtype(hardware)

    report("Initializing Krea runtime", 0.06)
    load_started = time.perf_counter()
    pipe, load_info = base._load_pipeline(request, dtype, hardware, report=report)
    timing["pipeline_load_seconds"] = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    prompt_embeds, prompt_mask, negative_embeds, negative_mask, encode_mode = base._encode_prompt_phase(
        pipe,
        prompt=prompt,
        guidance=guidance,
        hardware=hardware,
        report=report,
    )
    prompt_embeds, negative_embeds = _coerce_conditioning_dtype(
        prompt_embeds,
        negative_embeds,
        dtype,
    )
    timing["prompt_encode_seconds"] = time.perf_counter() - encode_started

    # Preserve masks as bool on CPU; floating conditioning must match the
    # denoiser dtype so Diffusers creates latents in BF16/FP16 rather than FP32.
    prompt_mask = prompt_mask.to(device="cpu", dtype=torch.bool)
    if negative_mask is not None:
        negative_mask = negative_mask.to(device="cpu", dtype=torch.bool)

    vae = pipe.vae
    pipe.vae = None

    if device == "cuda":
        hardware = detect_torch_hardware(torch)
    transformer_storage_gb = base._module_storage_gb(pipe.transformer)
    reserve_gb = _memory_reserve_gb(hardware, width, height)

    report("Selecting Krea GPU profile", 0.50)
    setup_started = time.perf_counter()
    execution_mode = base._configure_execution(
        pipe,
        hardware=hardware,
        load_info=load_info,
        transformer_storage_gb=transformer_storage_gb,
        reserve_gb=reserve_gb,
    )
    initial_execution_mode = execution_mode
    initial_load_info = dict(load_info)
    fallback_attempts: list[dict[str, Any]] = []
    fallback_reason: str | None = None

    # Estimates can still be optimistic. Validate *actual* post-placement free
    # memory before the first forward and proactively downgrade if resident
    # placement consumed the reserve we meant to protect.
    if device == "cuda" and execution_mode.startswith("resident"):
        post_setup = detect_torch_hardware(torch)
        post_free = float(post_setup.get("free_vram_gb") or 0.0)
        if post_free < reserve_gb:
            fallback_reason = "resident_headroom"
            fallback_attempts.append(
                {
                    "from": execution_mode,
                    "to": "transformer-block-sync",
                    "reason": "post_setup_free_vram_below_reserve",
                    "free_vram_gb": round(post_free, 3),
                    "reserve_gb": round(reserve_gb, 3),
                }
            )
            _dispose_pipe(pipe, device)
            hardware = detect_torch_hardware(torch)
            pipe, load_info, execution_mode, transformer_storage_gb = _fresh_denoiser(
                request,
                dtype=dtype,
                hardware=hardware,
                reserve_gb=reserve_gb,
                mode_override="transformer-block",
                stream_prefetch=False,
                report=report,
            )

    timing["execution_setup_seconds"] = time.perf_counter() - setup_started
    timing["offload_setup_seconds"] = timing["execution_setup_seconds"]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    latent_batches: list[torch.Tensor] = []
    total_denoise_steps = max(1, steps * num_images)
    denoise_started = time.perf_counter()

    for index in range(num_images):
        report(
            "Starting Krea denoising",
            0.54 + (0.35 * (index * steps) / total_denoise_steps),
            index * steps,
            total_denoise_steps,
        )

        retry_queue = _retry_profiles(execution_mode, hardware)
        while True:
            try:
                latent = base._denoise_one(
                    pipe,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    negative_embeds=negative_embeds,
                    negative_mask=negative_mask,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    seed=seed + index,
                    device=device,
                    index=index,
                    total_denoise_steps=total_denoise_steps,
                    report=report,
                )
                break
            except Exception as exc:
                if device != "cuda" or not base._is_cuda_oom(exc) or not retry_queue:
                    raise

                mode_override, use_stream = retry_queue.pop(0)
                previous_mode = execution_mode
                fallback_reason = "cuda_oom"
                report("Krea OOM; retrying with safer GPU profile", 0.53)
                _dispose_pipe(pipe, device)
                hardware = detect_torch_hardware(torch)
                reload_started = time.perf_counter()
                pipe, load_info, execution_mode, transformer_storage_gb = _fresh_denoiser(
                    request,
                    dtype=dtype,
                    hardware=hardware,
                    reserve_gb=reserve_gb,
                    mode_override=mode_override,
                    stream_prefetch=use_stream,
                    report=report,
                )
                timing["fallback_reload_seconds"] = timing.get("fallback_reload_seconds", 0.0) + (
                    time.perf_counter() - reload_started
                )
                fallback_attempts.append(
                    {
                        "from": previous_mode,
                        "to": execution_mode,
                        "reason": "cuda_oom",
                        "seed": seed + index,
                    }
                )
                # Recalculate the ladder from the new profile. It is safe to
                # retry because each attempt recreates the same latent noise from
                # the same seed and returns only after a complete denoise.
                retry_queue = _retry_profiles(execution_mode, hardware)

        latent_batches.append(latent)

    timing["denoise_seconds"] = time.perf_counter() - denoise_started

    del prompt_embeds, prompt_mask, negative_embeds, negative_mask
    report("Releasing Krea denoiser", 0.90, total_denoise_steps, total_denoise_steps)
    base._release_transformer(pipe, device)

    report("Decoding Krea image", 0.92, total_denoise_steps, total_denoise_steps)
    decode_started = time.perf_counter()
    pipe.vae = vae
    if max(width, height) > 1536:
        try:
            vae.enable_tiling()
        except Exception:
            pass

    for index, latents in enumerate(latent_batches):
        images = base._decode_latents(
            pipe,
            vae,
            latents,
            width=width,
            height=height,
            device=device,
        )
        image = images[0]
        report("Writing Krea image", 0.97, (index + 1) * steps, total_denoise_steps)
        write_started = time.perf_counter()
        path = output_dir / f"krea2_{index:03d}.png"
        image.save(path)
        timing["image_write_seconds"] = timing.get("image_write_seconds", 0.0) + (
            time.perf_counter() - write_started
        )
        saved.append(str(path))

    timing["decode_seconds"] = time.perf_counter() - decode_started
    timing["denoise_decode_seconds"] = timing["denoise_seconds"] + timing["decode_seconds"]
    timing["inference_seconds"] = timing["prompt_encode_seconds"] + timing["denoise_decode_seconds"]
    timing["worker_total_seconds"] = time.perf_counter() - worker_started
    report("Returning Krea image", 0.975, total_denoise_steps, total_denoise_steps)

    transformer_passes_per_step = 2 if guidance > 0 else 1
    runtime_hardware = dict(hardware)
    runtime_hardware["free_vram_gb"] = round(float(runtime_hardware.get("free_vram_gb") or 0.0), 2)
    runtime_hardware["total_vram_gb"] = round(float(runtime_hardware.get("total_vram_gb") or 0.0), 2)

    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "timing": {key: round(value, 6) for key, value in timing.items()},
        "runtime": {
            **initial_load_info,
            "offload": execution_mode,
            "initial_offload": initial_execution_mode,
            "fallback_reason": fallback_reason,
            "fallback_attempts": fallback_attempts,
            "text_encoder_mode": encode_mode,
            "conditioning_dtype": str(dtype).replace("torch.", ""),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "hardware": runtime_hardware,
            "transformer_storage_gb": round(transformer_storage_gb, 2),
            "activation_reserve_gb": round(reserve_gb, 2),
            "system_ram_gb": round(base._system_ram_gb(), 2),
            "transformer_forward_passes": steps * num_images * transformer_passes_per_step,
        },
    }


base._run = _run

if __name__ == "__main__":
    raise SystemExit(base.main())
