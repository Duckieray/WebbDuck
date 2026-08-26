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


_DEFAULT_FLUX2_9B_COMPONENT_MODEL = "black-forest-labs/FLUX.2-klein-9B"


def _snap(value: int, multiple: int = 8) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


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


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    import torch
    from PIL import Image

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

    offload_requested = str(os.getenv("WEBBDUCK_FLUX_OFFLOAD", "auto")).strip().lower()
    detection = request.get("detection") or {}
    is_gguf_9b = model_format == "gguf" and str(detection.get("parameter_size") or "").upper() == "9B"
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

    _write_progress(progress_path, "Configuring FLUX.2 GPU memory", 0.52)
    if use_cpu_offload:
        print(
            f"[flux] enabling model CPU offload "
            f"(GPU={gpu_name or 'unknown'}, VRAM={total_vram_gb:.2f} GiB)",
            flush=True,
        )
        pipe.enable_model_cpu_offload()
        selected_offload = "model_cpu"
    else:
        print(
            f"[flux] placing pipeline on {device} "
            f"(GPU={gpu_name or device}, VRAM={total_vram_gb:.2f} GiB)",
            flush=True,
        )
        pipe.to(device)
        selected_offload = "resident"

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

    input_image = None
    input_path = request.get("input_image")
    if input_path:
        input_image = Image.open(str(input_path)).convert("RGB")

    try:
        call_sig = inspect.signature(pipe.__call__)
        call_params = call_sig.parameters
    except Exception:
        call_params = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    inference_started = time.monotonic()
    for index in range(num_images):
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
        }
        if input_image is not None:
            if call_params and "image" not in call_params:
                raise RuntimeError("Installed Flux2KleinPipeline does not expose image editing support")
            kwargs["image"] = input_image
        generator_device = device if device == "cuda" else "cpu"
        kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed + index)

        if not call_params or "callback_on_step_end" in call_params:
            def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                completed = int(step_index) + 1
                fraction = completed / max(1, steps)
                _write_progress(
                    progress_path,
                    "Denoising FLUX.2",
                    0.60 + 0.32 * fraction,
                    completed,
                    steps,
                )
                return callback_kwargs

            kwargs["callback_on_step_end"] = _on_step_end

        _write_progress(progress_path, "Encoding prompt / starting FLUX.2 inference", 0.58, 0, steps)
        print(
            f"[flux] inference {index + 1}/{num_images}: "
            f"{width}x{height}, steps={steps}, guidance={guidance}",
            flush=True,
        )
        result = pipe(**kwargs)
        _write_progress(progress_path, "Decoding and saving FLUX.2 image", 0.94, steps, steps)
        image = result.images[0]
        path = output_dir / f"flux_{index:03d}.png"
        image.save(path)
        saved.append(str(path))
    inference_seconds = time.monotonic() - inference_started

    runtime.update(
        {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "free_vram_gb_before_load": round(free_vram_gb, 2),
            "offload_mode": selected_offload,
            "pipeline_load_seconds": round(pipeline_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "effective_width": width,
            "effective_height": height,
            "steps": steps,
            "guidance": guidance,
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
