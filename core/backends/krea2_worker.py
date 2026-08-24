"""Standalone Krea 2 Diffusers worker.

Full Diffusers checkpoints load normally. Community single-file transformer
checkpoints are streamed directly into a transformer scaffold created from the
official Krea config on the meta device. This avoids downloading/loading a
second official transformer merely to overwrite it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backends.krea2_weights import map_krea2_source_key, strip_krea2_prefix
from models.quantization import classify_comfy_quant, parse_comfy_quant_payload


def _snap(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _component_source(variant: str) -> str:
    explicit = str(os.getenv("WEBBDUCK_KREA2_COMPONENT_MODEL") or "").strip()
    if explicit:
        return explicit
    if variant == "turbo":
        return str(os.getenv("WEBBDUCK_KREA2_TURBO_COMPONENT_MODEL") or "krea/Krea-2-Turbo")
    return str(os.getenv("WEBBDUCK_KREA2_BASE_COMPONENT_MODEL") or "krea/Krea-2-Raw")


def _write_progress(
    progress_path: Path | None,
    stage: str,
    progress: float,
    step: int = 0,
    total_steps: int = 0,
) -> None:
    """Publish one atomic progress snapshot for the host WebbDuck process."""
    if progress_path is None:
        return
    payload = {
        "stage": str(stage),
        "progress": max(0.0, min(1.0, float(progress))),
        "step": max(0, int(step)),
        "total_steps": max(0, int(total_steps)),
        "updated_at": time.time(),
    }
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_name(progress_path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, progress_path)
    except Exception:
        # Progress is telemetry only and must never fail generation.
        pass


def _source_layer_prefix(weight_key: str) -> str | None:
    return weight_key[: -len(".weight")] if weight_key.endswith(".weight") else None


def _quant_config(handle: Any, keys: set[str], source_key: str) -> dict[str, Any]:
    prefix = _source_layer_prefix(source_key)
    if not prefix:
        return {}
    marker = f"{prefix}.comfy_quant"
    if marker not in keys:
        return {}
    return parse_comfy_quant_payload(handle.get_tensor(marker))


def _metadata_quant_configs(handle: Any) -> list[dict[str, Any]]:
    """Read optional top-level Comfy quantization metadata from safetensors."""
    try:
        metadata = handle.metadata() or {}
    except Exception:
        return []
    raw = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    layers = parsed.get("layers")
    if not isinstance(layers, dict):
        return []
    return [dict(value) for value in layers.values() if isinstance(value, dict)]


def _dequantize_source_tensor(
    handle: Any,
    keys: set[str],
    source_key: str,
    tensor: Any,
    target_dtype: Any,
) -> Any:
    import torch

    config = _quant_config(handle, keys, source_key)
    quant_kind = classify_comfy_quant(config)
    if quant_kind == "convrot":
        raise RuntimeError(
            "This Krea checkpoint uses INT8 ConvRot. ConvRot rotates both stored weights and runtime "
            "activations, so it cannot be safely flattened into a normal Diffusers transformer. "
            "Use an FP8/BF16 checkpoint until WebbDuck's native ConvRot backend is installed."
        )
    if quant_kind == "unsupported":
        raise RuntimeError(
            f"Unsupported Krea quantization format {config.get('format')!r} in tensor {source_key!r}."
        )

    prefix = _source_layer_prefix(source_key)
    scale_key = f"{prefix}.weight_scale" if prefix else ""
    is_float8 = str(tensor.dtype).startswith("torch.float8")

    if quant_kind == "fp8" or is_float8:
        # Comfy FP8 stores q = W / scale. Reconstruct the ordinary weight for
        # this correctness-first backend. Native FP8 matmul can replace this
        # later without changing the model-facing contract.
        value = tensor.float()
        if scale_key and scale_key in keys:
            value = value * handle.get_tensor(scale_key).float()
        return value.to(dtype=target_dtype)

    if tensor.dtype in {torch.int8, torch.uint8}:
        raise RuntimeError(
            f"Integer-quantized Krea tensor {source_key!r} has no supported dequantization contract."
        )
    return tensor.to(dtype=target_dtype)


def _reshape_for_target(value: Any, target: Any, mode: str | None, source_key: str) -> Any:
    if mode in {"six_rows", "two_rows"}:
        if int(value.numel()) != int(target.numel()):
            raise RuntimeError(
                f"Krea modulation tensor {source_key!r} has {value.numel()} values; "
                f"expected {target.numel()}."
            )
        return value.reshape(target.shape)
    if tuple(value.shape) != tuple(target.shape):
        raise RuntimeError(
            f"Krea tensor shape mismatch for {source_key!r}: source={tuple(value.shape)} "
            f"target={tuple(target.shape)}."
        )
    return value


def _assign_target_tensor(
    transformer: Any,
    target_key: str,
    target: Any,
    value: Any,
) -> None:
    """Materialize a meta parameter or copy into an already allocated target."""
    if bool(getattr(target, "is_meta", False)):
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(
            transformer,
            target_key,
            device="cpu",
            value=value,
            dtype=value.dtype,
        )
        return
    target.copy_(value)


def overlay_single_file_transformer(
    transformer: Any,
    checkpoint_path: Path,
    *,
    target_dtype: Any | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Stream one Krea transformer checkpoint into a Diffusers model/scaffold."""
    import torch
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    target_state = transformer.state_dict(keep_vars=True)
    expected = set(target_state)
    mapped: set[str] = set()
    ignored: list[str] = []

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())

        quant_configs = _metadata_quant_configs(handle)
        for marker in (key for key in keys if key.endswith(".comfy_quant")):
            parsed = parse_comfy_quant_payload(handle.get_tensor(marker))
            if parsed:
                quant_configs.append(parsed)
        if any(classify_comfy_quant(config) == "convrot" for config in quant_configs):
            raise RuntimeError(
                "INT8 ConvRot Krea checkpoint detected. This format requires runtime activation "
                "rotation and is intentionally not handled by the FP8/BF16 overlay loader."
            )

        ignored_suffixes = (
            ".comfy_quant",
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
            ".pre_quant_scale",
        )
        source_keys = [key for key in sorted(keys) if not key.endswith(ignored_suffixes)]
        total_source_keys = max(1, len(source_keys))
        report_every = max(1, total_source_keys // 50)

        with torch.no_grad():
            for source_index, source_key in enumerate(source_keys, start=1):
                target_key, reshape_mode = map_krea2_source_key(source_key)
                normalized = strip_krea2_prefix(source_key)
                if target_key is None and normalized in target_state:
                    target_key = normalized
                if target_key is None:
                    ignored.append(source_key)
                else:
                    target = target_state.get(target_key)
                    if target is None:
                        raise RuntimeError(
                            f"Krea checkpoint tensor {source_key!r} maps to unknown Diffusers key {target_key!r}."
                        )

                    source = handle.get_tensor(source_key)
                    effective_dtype = target_dtype if target_dtype is not None else target.dtype
                    value = _dequantize_source_tensor(
                        handle,
                        keys,
                        source_key,
                        source,
                        effective_dtype,
                    )
                    value = _reshape_for_target(value, target, reshape_mode, source_key)
                    _assign_target_tensor(transformer, target_key, target, value)
                    mapped.add(target_key)
                    del source, value

                if callable(progress_callback) and (
                    source_index == total_source_keys or source_index % report_every == 0
                ):
                    try:
                        progress_callback(source_index, total_source_keys)
                    except Exception:
                        pass

    missing = sorted(expected - mapped)
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(
            f"Krea single-file overlay was incomplete: {len(missing)} Diffusers transformer tensors "
            f"were not supplied by the checkpoint. First missing keys: {preview}"
        )

    return {
        "mapped_tensors": len(mapped),
        "ignored_tensors": len(ignored),
    }


def _build_empty_transformer(component_source: str) -> Any:
    """Instantiate the official Krea transformer shape without its weights."""
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel

    config = Krea2Transformer2DModel.load_config(
        component_source,
        subfolder="transformer",
    )
    with init_empty_weights():
        transformer = Krea2Transformer2DModel.from_config(config)
    return transformer


def _configure_offload(pipe: Any, device: str, total_vram_gb: float) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    import torch

    mode = str(os.getenv("WEBBDUCK_KREA2_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        # Krea's 13B transformer is ~26 GB in BF16, so a 16 GB card still needs
        # leaf-level offload. Diffusers group offload with CUDA streams preserves
        # that low-memory behavior while prefetching upcoming layers to reduce the
        # synchronization overhead of sequential_cpu_offload.
        mode = "group" if total_vram_gb < 20.0 else ("model" if total_vram_gb < 28.0 else "none")

    if mode in {"group", "stream", "group-stream", "leaf"}:
        low_cpu_mem_usage = str(
            os.getenv("WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
                record_stream=True,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            return "group-leaf-stream"
        except Exception:
            # Compatibility fallback for a component/hook combination that does
            # not support group offloading on the installed Diffusers build.
            pipe.enable_sequential_cpu_offload(device="cuda")
            return "sequential-fallback"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"
    if mode in {"model", "cpu", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    pipe.to("cuda")
    return "none"


def _load_pipeline(request: dict, dtype: Any, report: Any | None = None) -> tuple[Any, dict[str, Any]]:
    from diffusers import Krea2Pipeline

    model_path = Path(str(request["model_path"])).expanduser()
    model_format = str(request.get("model_format") or "").lower()
    variant = str(request.get("variant") or "base").lower()

    if model_format != "single" and model_path.is_dir():
        if callable(report):
            report("Loading Krea pipeline", 0.10)
        pipe = Krea2Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if callable(report):
            report("Krea pipeline loaded", 0.42)
        return pipe, {"source": "diffusers", "variant": variant}

    if not model_path.is_file():
        raise RuntimeError(f"Krea checkpoint path does not exist: {model_path}")

    quantization = str(request.get("quantization") or "none").lower()
    if quantization == "convrot":
        raise RuntimeError(
            "INT8 ConvRot Krea checkpoints require WebbDuck's future native ConvRot runtime; "
            "the generic single-file loader will not silently mis-handle them."
        )

    component_source = str(request.get("component_source") or _component_source(variant))
    if callable(report):
        report("Building Krea transformer", 0.08)
    transformer = _build_empty_transformer(component_source)

    if callable(report):
        report("Loading Krea checkpoint", 0.12)

    def overlay_progress(current: int, total: int) -> None:
        fraction = float(current) / float(max(1, total))
        if callable(report):
            report("Loading Krea checkpoint", 0.12 + (0.20 * fraction), current, total)

    overlay = overlay_single_file_transformer(
        transformer,
        model_path,
        target_dtype=dtype,
        progress_callback=overlay_progress,
    )
    if callable(report):
        report("Loading Krea support models", 0.34)
    pipe = Krea2Pipeline.from_pretrained(
        component_source,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    is_distilled = variant == "turbo"
    if hasattr(pipe, "is_distilled"):
        pipe.is_distilled = is_distilled
    register_to_config = getattr(pipe, "register_to_config", None)
    if callable(register_to_config):
        register_to_config(is_distilled=is_distilled)

    if callable(report):
        report("Krea pipeline loaded", 0.42)
    return pipe, {
        "source": "single",
        "variant": variant,
        "quantization": quantization,
        "component_source": component_source,
        "component_transformer_weights_loaded": False,
        **overlay,
    }


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    import torch

    prompt = str(request["prompt"])
    width = _snap(int(request.get("width") or 1024))
    height = _snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 28))
    guidance = float(request.get("guidance") if request.get("guidance") is not None else 4.5)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") or 0)
    timing: dict[str, float] = {}

    def report(stage: str, progress: float, step: int = 0, total_steps: int = 0) -> None:
        _write_progress(progress_path, stage, progress, step, total_steps)

    report("Initializing Krea runtime", 0.06)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    load_started = time.perf_counter()
    pipe, load_info = _load_pipeline(request, dtype, report=report)
    timing["pipeline_load_seconds"] = time.perf_counter() - load_started

    total_vram_gb = 0.0
    if device == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total_vram_gb = 0.0

    report("Optimizing Krea GPU path", 0.44)
    offload_started = time.perf_counter()
    offload = _configure_offload(pipe, device, total_vram_gb)
    timing["offload_setup_seconds"] = time.perf_counter() - offload_started

    # Diffusers warns that VAE tiling combined with streamed group offload may
    # need a dummy forward pass to establish device placement. Skip tiling for
    # the streamed group mode; the transformer has already been offloaded, so a
    # 1024px VAE decode is normally safe on the target 16 GB class GPU.
    if not offload.startswith("group"):
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    total_denoise_steps = max(1, steps * num_images)
    inference_started = time.perf_counter()

    for index in range(num_images):
        generator_device = device if device == "cuda" else "cpu"
        report(
            "Encoding Krea prompt",
            0.46 + (0.47 * (index * steps) / total_denoise_steps),
            index * steps,
            total_denoise_steps,
        )

        def on_step_end(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict) -> dict:
            completed = index * steps + step_index + 1
            progress = 0.48 + (0.47 * completed / total_denoise_steps)
            stage = "Denoising with Krea 2"
            if completed >= total_denoise_steps:
                stage = "Decoding Krea image"
            report(stage, progress, completed, total_denoise_steps)
            return callback_kwargs

        with torch.inference_mode():
            result = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator(device=generator_device).manual_seed(seed + index),
                callback_on_step_end=on_step_end,
            )
        image = result.images[0]
        report("Writing Krea image", 0.97, (index + 1) * steps, total_denoise_steps)
        path = output_dir / f"krea2_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    timing["inference_seconds"] = time.perf_counter() - inference_started
    timing["worker_total_seconds"] = sum(timing.values())
    report("Returning Krea image", 0.975, total_denoise_steps, total_denoise_steps)

    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "timing": {key: round(value, 6) for key, value in timing.items()},
        "runtime": {
            **load_info,
            "offload": offload,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "total_vram_gb": round(total_vram_gb, 2),
        },
    }


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
        result = _run(request, Path(args.output_dir), progress_path=progress_path)
    except Exception as exc:
        _write_progress(progress_path, "Krea generation error", 0.0)
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
