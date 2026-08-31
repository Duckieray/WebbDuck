"""Standalone FLUX Diffusers worker.

This file intentionally imports torch/diffusers only inside the isolated runtime.
It can therefore be launched with WEBBDUCK_FLUX_PYTHON pointing at a backend-
specific environment without constraining WebbDuck's core dependency set.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
import traceback
from pathlib import Path

try:
    from flux_lora import apply_flux_loras
except ModuleNotFoundError:
    # Allow importing this module from a test host where the worker's backends
    # directory is not on sys.path. The launcher runs this file as a script, so
    # its parent directory is already searchable in the real runtime.
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flux_lora import apply_flux_loras

# FLUX.2 image/identity conditioning allocates transient attention buffers that
# can fragment VRAM and OOM at tight headroom. Use the expandable-segments
# allocator by default while preserving an explicit operator/user override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_DEFAULT_FLUX2_9B_COMPONENT_MODEL = "black-forest-labs/FLUX.2-klein-9B"
_MAX_FLUX_IDENTITY_REFERENCES = 5


def _snap(value: int, multiple: int = 8) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _flux_demotion_tiers(
    width: int,
    height: int,
    condition_count: int,
    lora_count: int,
) -> list[dict]:
    """Ordered fallback configs for OOM-aware retries.

    Each tier records how many conditioning images and LoRA adapters to keep
    plus a pixel budget. Tiers are non-increasing in every resource dimension
    (conditioning images, LoRAs, and pixel area), so a worker can retry a failed
    image at the next tier without ever requesting more memory than the tier
    that just failed.

    Priority intends to protect identity conditioning: fewer pixels and fewer
    adapters are preferred over dropping reference images outright.
    """
    tiers: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    def _add(cond_keep: int, w: int, h: int, lora_keep: int) -> None:
        entry = {
            "condition_count": int(max(0, cond_keep)),
            "width": int(w),
            "height": int(h),
            "lora_count": int(max(0, lora_keep)),
        }
        key = (entry["condition_count"], entry["width"], entry["height"], entry["lora_count"])
        if key not in seen:
            seen.add(key)
            tiers.append(entry)

    full_refs = int(condition_count)
    full_width = int(width)
    full_height = int(height)
    full_loras = int(lora_count)

    # Carried budget. Only ever decreases for the whole ladder.
    cond_kept = full_refs
    w_used = full_width
    h_used = full_height
    lora_kept = full_loras

    # Tier 0: the requested configuration, complete pipeline.
    _add(cond_kept, w_used, h_used, lora_kept)

    # Reduce pixel area first: one step cuts attention activations for every
    # conditioning image at once, unlike dropping a single reference.
    for scale in (0.9, 0.8):
        w = _snap(int(full_width * scale))
        h = _snap(int(full_height * scale))
        if min(w, h) >= 512:
            w_used = w
            h_used = h
            _add(cond_kept, w_used, h_used, lora_kept)

    # Then reduce the conditioning image count at the carried resolution, which
    # keeps remaining identity references intact for the rest of the ladder.
    for cond_target in (3, 2, 1):
        if cond_target >= cond_kept:
            continue
        cond_kept = cond_target
        _add(cond_kept, w_used, h_used, lora_kept)

    # Deactivate adapters one at a time as a controlled style trade-off, still
    # at the carried conditioning/resolution budget.
    for lora_target in range(max(0, full_loras - 1), -1, -1):
        if lora_target >= lora_kept:
            continue
        lora_kept = lora_target
        _add(cond_kept, w_used, h_used, lora_kept)

    # Last resort: combine the smallest conditioning set with a shrunken image.
    w = _snap(int(full_width * 0.75))
    h = _snap(int(full_height * 0.75))
    if min(w, h) >= 512:
        for cond_target in (max(0, cond_kept), 0):
            for lora_target in range(lora_kept, -1, -1):
                _add(cond_target, w, h, lora_target)
    return tiers


def _write_progress(
    progress_path: Path | None,
    stage: str,
    progress: float,
    step: int = 0,
    total_steps: int = 0,
) -> None:
    if progress_path is None:
        return
    try:
        payload = {
            "stage": str(stage),
            "progress": max(0.0, min(1.0, float(progress))),
            "step": int(step),
            "total_steps": int(total_steps),
            "updated_at": time.time(),
        }
        temp = progress_path.with_name(f".{progress_path.name}.tmp")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temp, progress_path)
    except Exception:
        pass


def _load_flux_pipeline(request: dict, dtype, progress_path: Path | None = None):
    from diffusers import Flux2KleinPipeline

    model_path = str(request["model_path"])
    model_format = str(request.get("model_format") or "diffusers").strip().lower()
    component_model = str(
        request.get("component_model")
        or os.getenv("WEBBDUCK_FLUX2_COMPONENT_MODEL")
        or _DEFAULT_FLUX2_9B_COMPONENT_MODEL
    )
    hf_token = str(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()

    if model_format != "gguf":
        _write_progress(progress_path, "Loading FLUX.2 pipeline", 0.10)
        print(f"[flux] loading Diffusers pipeline from {model_path}", flush=True)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=hf_token or None,
        )
        _write_progress(progress_path, "FLUX.2 pipeline loaded", 0.48)
        return pipe, {
            "model_format": model_format,
            "component_model": model_path,
            "quantization": "none",
        }

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"FLUX GGUF checkpoint does not exist: {model_path}")

    try:
        from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
        from gguf import GGUFReader  # noqa: F401 - explicit runtime dependency check
    except Exception as exc:
        raise RuntimeError(
            "FLUX GGUF runtime dependencies are missing. "
            "Run `python tools/prepare_model_runtimes.py flux` and restart WebbDuck. "
            f"Import error: {exc}"
        ) from exc

    detection = request.get("detection") or {}
    quantization = str(detection.get("quantization") or "unknown")
    _write_progress(progress_path, f"Loading FLUX.2 GGUF transformer ({quantization})", 0.08)
    print(
        f"[flux] loading GGUF transformer {model_path} "
        f"({quantization}) with config {component_model}/transformer",
        flush=True,
    )

    # Supplying the exact Klein transformer config is intentional. Letting
    # Diffusers infer the FLUX.2 transformer shape from GGUF can select the
    # larger FLUX.2 dev shape for modulation layers and produce a shape mismatch.
    transformer = Flux2Transformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=component_model,
        subfolder="transformer",
        torch_dtype=dtype,
        token=hf_token or None,
    )

    _write_progress(progress_path, "Loading FLUX.2 support components", 0.22)
    print(
        f"[flux] loading support components from {component_model} "
        "(transformer supplied by local GGUF)",
        flush=True,
    )
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            component_model,
            transformer=transformer,
            torch_dtype=dtype,
            token=hf_token or None,
        )
    except Exception as exc:
        message = str(exc or exc.__class__.__name__)
        lower = message.lower()
        if any(
            token in lower
            for token in (
                "gated",
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "not a valid model identifier",
                "repository not found",
            )
        ):
            raise RuntimeError(
                f"FLUX.2 GGUF support components come from gated repository {component_model}. "
                "Accept the model terms on Hugging Face and configure an authorized Hugging Face "
                "token in WebbDuck Settings, then retry."
            ) from exc
        raise

    _write_progress(progress_path, "FLUX.2 model components ready", 0.48)
    return pipe, {
        "model_format": "gguf",
        "component_model": component_model,
        "quantization": quantization,
        "parameter_size": detection.get("parameter_size"),
        "variant": detection.get("variant"),
        "gguf_path": model_path,
    }


def _prompt_sequence_length(pipe, prompt: str) -> tuple[int, int]:
    """Choose the smallest safe Qwen context bucket for the actual prompt.

    Flux2Klein pads every prompt to max_sequence_length. Running all prompts at
    the default 512 tokens wastes substantial Qwen compute, especially when the
    text encoder must remain on CPU. The attention mask makes shorter padding
    equivalent for the real prompt tokens, so use a compact bucket with margin.
    """
    explicit = str(os.getenv("WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH") or "").strip()
    if explicit:
        try:
            value = max(32, min(512, int(explicit)))
            return value, value
        except ValueError:
            pass

    token_count = 128
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is not None:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if ids:
                token_count = int(len(ids))
        except Exception:
            token_count = 128

    needed = min(512, max(32, token_count + 16))
    for bucket in (64, 96, 128, 192, 256, 384, 512):
        if needed <= bucket:
            return bucket, token_count
    return 512, token_count


def _encode_prompt_on_cpu(
    pipe,
    prompt: str,
    *,
    progress_path: Path | None,
):
    """Encode Qwen prompt embeddings without ever materializing Qwen on GPU."""
    import torch

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("FLUX.2 pipeline has no text encoder for prompt encoding")

    sequence_length, token_count = _prompt_sequence_length(pipe, prompt)
    _write_progress(
        progress_path,
        f"Encoding FLUX.2 prompt on CPU ({token_count} tokens)",
        0.53,
    )
    print(
        f"[flux] constrained-memory prompt phase: CPU Qwen3 encode "
        f"(tokens={token_count}, padded={sequence_length})",
        flush=True,
    )

    started = time.monotonic()
    try:
        text_encoder.to("cpu")
    except Exception:
        pass

    with torch.inference_mode():
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            max_sequence_length=sequence_length,
        )

    prompt_embeds = prompt_embeds.detach().to("cpu")
    elapsed = time.monotonic() - started
    _write_progress(progress_path, "FLUX.2 prompt encoded", 0.57)
    print(f"[flux] CPU prompt encoding completed in {elapsed:.2f}s", flush=True)
    return prompt_embeds, sequence_length, elapsed


def _load_rgb_image(path_value) -> object:
    """Load an RGB PIL image without leaving a file handle open."""
    from PIL import Image

    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"FLUX conditioning image does not exist: {path}")
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    import torch

    model_format = str(request.get("model_format") or "diffusers").strip().lower()
    prompt = str(request["prompt"])
    width = _snap(int(request.get("width") or 1024))
    height = _snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 4))
    guidance = float(request.get("guidance") if request.get("guidance") is not None else 1.0)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") if request.get("seed") is not None else 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    total_vram_gb = 0.0
    free_vram_gb = 0.0
    gpu_name = None
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / 1024**3
            gpu_name = props.name
        except Exception:
            total_vram_gb = 0.0
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            free_vram_gb = free_bytes / 1024**3
        except Exception:
            free_vram_gb = 0.0

    _write_progress(progress_path, "Preparing FLUX.2 runtime", 0.06)
    load_started = time.monotonic()
    pipe, runtime = _load_flux_pipeline(request, dtype, progress_path)
    pipeline_load_seconds = time.monotonic() - load_started

    raw_loras = request.get("loras") or []

    def _lora_progress(stage: str, progress: float) -> None:
        _write_progress(progress_path, stage, progress)

    applied_loras = apply_flux_loras(
        pipe,
        raw_loras,
        progress=_lora_progress if raw_loras else None,
    )

    offload_requested = str(os.getenv("WEBBDUCK_FLUX_OFFLOAD", "auto")).strip().lower()
    detection = request.get("detection") or {}
    is_gguf_9b = model_format == "gguf" and str(detection.get("parameter_size") or "").upper() == "9B"

    # FLUX.2 Klein 9B ships an ~16.4 GB BF16 Qwen3 text encoder. It cannot fit
    # by itself on a nominal 16 GB card. Model CPU offload is insufficient here:
    # Accelerate still moves the entire text encoder to the execution device when
    # encode_prompt runs. Precompute prompt_embeds on CPU first, then skip Qwen
    # during the GPU pipeline call. Also use this path on larger cards when live
    # free VRAM is too low to materialize Qwen safely.
    phased_prompt_encoding = bool(
        device == "cuda"
        and is_gguf_9b
        and (total_vram_gb < 20.0 or (free_vram_gb and free_vram_gb < 19.0))
    )
    prompt_embeds_cpu = None
    prompt_sequence_length = None
    prompt_encode_seconds = 0.0
    if phased_prompt_encoding:
        prompt_embeds_cpu, prompt_sequence_length, prompt_encode_seconds = _encode_prompt_on_cpu(
            pipe,
            prompt,
            progress_path=progress_path,
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    use_cpu_offload = device == "cuda" and (
        offload_requested in {"1", "true", "yes", "cpu", "model_cpu"}
        or (
            offload_requested == "auto"
            and (
                total_vram_gb < 18.0
                or (is_gguf_9b and total_vram_gb < 24.0)
            )
        )
    )

    _write_progress(progress_path, "Configuring FLUX.2 GPU memory", 0.58 if phased_prompt_encoding else 0.52)
    if use_cpu_offload:
        print(
            f"[flux] enabling model CPU offload "
            f"(GPU={gpu_name or 'unknown'}, VRAM={total_vram_gb:.2f} GiB)",
            flush=True,
        )
        pipe.enable_model_cpu_offload()
        selected_offload = "phased_prompt+model_cpu" if phased_prompt_encoding else "model_cpu"
    else:
        print(
            f"[flux] placing pipeline on {device} "
            f"(GPU={gpu_name or device}, VRAM={total_vram_gb:.2f} GiB)",
            flush=True,
        )
        pipe.to(device)
        selected_offload = "phased_prompt+resident" if phased_prompt_encoding else "resident"

    # Decode memory is independent of transformer quantization. Tiling/slicing
    # are cheap safeguards for 1024-class images and constrained accelerators.
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass
        if hasattr(vae, "enable_slicing"):
            try:
                vae.enable_slicing()
            except Exception:
                pass

    input_path = request.get("input_image")
    input_image = _load_rgb_image(input_path) if input_path else None

    raw_reference_images = request.get("reference_images") or []
    if not isinstance(raw_reference_images, list):
        raise ValueError("FLUX reference_images must be a list.")
    if len(raw_reference_images) > _MAX_FLUX_IDENTITY_REFERENCES:
        raise ValueError(
            f"FLUX.2 identity supports at most {_MAX_FLUX_IDENTITY_REFERENCES} reference images."
        )
    reference_images = [_load_rgb_image(path) for path in raw_reference_images]

    conditioning_images = []
    if input_image is not None:
        conditioning_images.append(input_image)
    conditioning_images.extend(reference_images)

    try:
        call_sig = inspect.signature(pipe.__call__)
        call_params = call_sig.parameters
    except Exception:
        call_params = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    demotion_tiers = _flux_demotion_tiers(
        width,
        height,
        len(conditioning_images),
        len(applied_loras),
    )
    working_tier_index = 0
    oom_retry_count = 0

    def _apply_tier(pipe_for_tier, kwargs_target: dict, tier: dict) -> dict:
        keep = max(0, int(tier["condition_count"]))
        images_for_tier = conditioning_images[:keep] if keep > 0 else []
        kwargs_target["width"] = int(tier["width"])
        kwargs_target["height"] = int(tier["height"])
        if conditioning_images:
            if images_for_tier:
                kwargs_target["image"] = (
                    images_for_tier if len(images_for_tier) > 1 else images_for_tier[0]
                )
            else:
                kwargs_target.pop("image", None)
        lora_target = max(0, int(tier["lora_count"]))
        active_names = [entry["adapter_name"] for entry in applied_loras[:lora_target]]
        active_weights = [entry["weight"] for entry in applied_loras[:lora_target]]
        if active_names:
            pipe_for_tier.set_adapters(active_names, adapter_weights=active_weights)
        else:
            try:
                pipe_for_tier.unload_lora_weights()
            except Exception:
                pass
        return kwargs_target

    inference_started = time.monotonic()
    for index in range(num_images):
        kwargs = {
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
        }
        if prompt_embeds_cpu is not None:
            # Diffusers accepts pre-generated embeddings, but encode_prompt does
            # not move supplied embeddings to the execution device. They are only
            # a few MB, so transfer them explicitly after Qwen has finished.
            kwargs["prompt"] = None
            kwargs["prompt_embeds"] = prompt_embeds_cpu.to(device=device, dtype=dtype)
            if prompt_sequence_length is not None:
                kwargs["max_sequence_length"] = int(prompt_sequence_length)
        else:
            kwargs["prompt"] = prompt

        if conditioning_images:
            if call_params and "image" not in call_params:
                raise RuntimeError("Installed Flux2KleinPipeline does not expose image editing support")
            kwargs["image"] = (
                conditioning_images
                if len(conditioning_images) > 1
                else conditioning_images[0]
            )

        if not call_params or "callback_on_step_end" in call_params:
            def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                completed = int(step_index) + 1
                fraction = completed / max(1, steps)
                _write_progress(
                    progress_path,
                    "Denoising FLUX.2",
                    0.64 + 0.28 * fraction,
                    completed,
                    steps,
                )
                return callback_kwargs

            kwargs["callback_on_step_end"] = _on_step_end

        start_stage = (
            "Starting FLUX.2 denoising with cached prompt"
            if prompt_embeds_cpu is not None
            else "Encoding prompt / starting FLUX.2 inference"
        )
        _write_progress(progress_path, start_stage, 0.62, 0, steps)
        print(
            f"[flux] inference {index + 1}/{num_images}: "
            f"{width}x{height}, steps={steps}, guidance={guidance}, "
            f"identity_refs={len(reference_images)}, "
            f"loras={len(applied_loras)}",
            flush=True,
        )
        result = None
        tier_used = demotion_tiers[working_tier_index]
        for tier_index in range(working_tier_index, len(demotion_tiers)):
            tier = demotion_tiers[tier_index]
            attempt_kwargs = dict(kwargs)
            # Re-seed for every attempt so a demoted retry is still exactly
            # reproducible from the requested seed despite earlier OOMs.
            attempt_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed + index)
            _apply_tier(pipe, attempt_kwargs, tier)
            try:
                result = pipe(**attempt_kwargs)
                tier_used = tier
                working_tier_index = tier_index
                break
            except torch.cuda.OutOfMemoryError:
                oom_retry_count += 1
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if tier_index + 1 >= len(demotion_tiers):
                    raise
                demote_summary = {
                    "condition_count": tier["condition_count"],
                    "lora_count": tier["lora_count"],
                    "width": tier["width"],
                    "height": tier["height"],
                }
                print(
                    f"[flux] CUDA out of memory at "
                    f"{attempt_kwargs.get('width')}x{attempt_kwargs.get('height')} "
                    f"refs={tier['condition_count']} loras={tier['lora_count']}; "
                    f"retrying with reduced budget {demote_summary}",
                    flush=True,
                )
                _write_progress(
                    progress_path,
                    "FLUX.2 memory pressure: reducing request budget",
                    0.62,
                    0,
                    steps,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda error" in message:
                    oom_retry_count += 1
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if tier_index + 1 >= len(demotion_tiers):
                        raise
                    continue
                raise
        if result is None:
            raise RuntimeError("FLUX.2 inference produced no result")
        _write_progress(progress_path, "Decoding and saving FLUX.2 image", 0.94, steps, steps)
        image = result.images[0]
        path = output_dir / f"flux_{index:03d}.png"
        image.save(path)
        saved.append(str(path))
    inference_seconds = time.monotonic() - inference_started

    effective_width = int(tier_used.get("width", width))
    effective_height = int(tier_used.get("height", height))
    effective_condition_count = int(tier_used.get("condition_count", len(conditioning_images)))
    effective_lora_count = int(tier_used.get("lora_count", len(applied_loras)))
    runtime.update(
        {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "free_vram_gb_before_load": round(free_vram_gb, 2),
            "offload_mode": selected_offload,
            "phased_prompt_encoding": phased_prompt_encoding,
            "prompt_encoding_device": "cpu" if phased_prompt_encoding else device,
            "prompt_sequence_length": prompt_sequence_length,
            "prompt_encode_seconds": round(prompt_encode_seconds, 3),
            "pipeline_load_seconds": round(pipeline_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "requested_width": width,
            "requested_height": height,
            "effective_width": effective_width,
            "effective_height": effective_height,
            "steps": steps,
            "guidance": guidance,
            "lora_count": len(applied_loras),
            "effective_lora_count": effective_lora_count,
            "loras": applied_loras,
            "identity_provider": request.get("identity_provider"),
            "identity_name": request.get("identity_name"),
            "identity_reference_count": len(reference_images),
            "conditioning_image_count": len(conditioning_images),
            "effective_conditioning_image_count": effective_condition_count,
            "oom_retry_count": oom_retry_count,
            "demotion_tier": tier_used,
            "demotion_enabled": True,
        }
    )
    _write_progress(progress_path, "FLUX.2 complete", 1.0, steps, steps)
    return {"ok": True, "images": saved, "seed": seed, "runtime": runtime}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress")
    args = parser.parse_args()

    result_path = Path(args.result)
    progress_path = Path(args.progress) if args.progress else None
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = _run(request, Path(args.output_dir), progress_path)
    except Exception as exc:
        _write_progress(progress_path, f"FLUX.2 failed: {exc}", 0.0)
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
