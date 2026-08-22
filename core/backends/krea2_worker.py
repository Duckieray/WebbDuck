"""Standalone Krea 2 Diffusers worker.

Full Diffusers checkpoints load normally. Community single-file transformer
checkpoints are overlaid onto the official Krea component pipeline one tensor
at a time, so discovery/loading does not depend on Diffusers implementing
Krea2Transformer2DModel.from_single_file().
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

try:
    from core.backends.krea2_weights import (
        classify_comfy_quant,
        map_krea2_source_key,
        parse_comfy_quant_payload,
        strip_krea2_prefix,
    )
except ImportError:  # launched directly as this file
    from krea2_weights import (  # type: ignore
        classify_comfy_quant,
        map_krea2_source_key,
        parse_comfy_quant_payload,
        strip_krea2_prefix,
    )


def _snap(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _component_source(variant: str) -> str:
    explicit = str(os.getenv("WEBBDUCK_KREA2_COMPONENT_MODEL") or "").strip()
    if explicit:
        return explicit
    if variant == "turbo":
        return str(os.getenv("WEBBDUCK_KREA2_TURBO_COMPONENT_MODEL") or "krea/Krea-2-Turbo")
    return str(os.getenv("WEBBDUCK_KREA2_BASE_COMPONENT_MODEL") or "krea/Krea-2-Raw")


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


def overlay_single_file_transformer(transformer: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Stream one Krea transformer checkpoint into an instantiated Diffusers model."""
    import torch
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    target_state = transformer.state_dict(keep_vars=True)
    expected = set(target_state)
    mapped: set[str] = set()
    ignored: list[str] = []

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        # Reject ConvRot before copying any data so a partially overlaid model
        # can never proceed after an unsupported-format error.
        for marker in (key for key in keys if key.endswith(".comfy_quant")):
            config = parse_comfy_quant_payload(handle.get_tensor(marker))
            if classify_comfy_quant(config) == "convrot":
                raise RuntimeError(
                    "INT8 ConvRot Krea checkpoint detected. This format requires runtime activation "
                    "rotation and is intentionally not handled by the FP8/BF16 overlay loader."
                )

        with torch.no_grad():
            for source_key in sorted(keys):
                if source_key.endswith(
                    (
                        ".comfy_quant",
                        ".weight_scale",
                        ".weight_scale_2",
                        ".input_scale",
                        ".pre_quant_scale",
                    )
                ):
                    continue

                target_key, reshape_mode = map_krea2_source_key(source_key)
                normalized = strip_krea2_prefix(source_key)
                if target_key is None and normalized in target_state:
                    target_key = normalized
                if target_key is None:
                    ignored.append(source_key)
                    continue

                target = target_state.get(target_key)
                if target is None:
                    raise RuntimeError(
                        f"Krea checkpoint tensor {source_key!r} maps to unknown Diffusers key {target_key!r}."
                    )

                source = handle.get_tensor(source_key)
                value = _dequantize_source_tensor(
                    handle,
                    keys,
                    source_key,
                    source,
                    target.dtype,
                )
                value = _reshape_for_target(value, target, reshape_mode, source_key)
                target.copy_(value)
                mapped.add(target_key)
                del source, value

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


def _configure_offload(pipe: Any, device: str, total_vram_gb: float) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("WEBBDUCK_KREA2_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        # A BF16 Krea transformer is larger than a 16 GB card by itself. Model
        # CPU offload moves whole components and is therefore insufficient on
        # that class of GPU; sequential offload keeps individual submodules on
        # the accelerator only while they execute.
        mode = "sequential" if total_vram_gb < 20.0 else ("model" if total_vram_gb < 28.0 else "none")

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"
    if mode in {"model", "cpu", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    pipe.to("cuda")
    return "none"


def _load_pipeline(request: dict, dtype: Any) -> tuple[Any, dict[str, Any]]:
    from diffusers import Krea2Pipeline

    model_path = Path(str(request["model_path"])).expanduser()
    model_format = str(request.get("model_format") or "").lower()
    variant = str(request.get("variant") or "base").lower()

    if model_format != "single" and model_path.is_dir():
        pipe = Krea2Pipeline.from_pretrained(str(model_path), torch_dtype=dtype)
        return pipe, {"source": "diffusers", "variant": variant}

    if not model_path.is_file():
        raise RuntimeError(f"Krea checkpoint path does not exist: {model_path}")

    quantization = str(request.get("quantization") or "none").lower()
    if quantization == "convrot":
        raise RuntimeError(
            "INT8 ConvRot Krea checkpoints require WebbDuck's future native ConvRot runtime; "
            "the generic single-file loader will not silently mis-handle them."
        )

    component_source = _component_source(variant)
    pipe = Krea2Pipeline.from_pretrained(component_source, torch_dtype=dtype)
    overlay = overlay_single_file_transformer(pipe.transformer, model_path)

    is_distilled = variant == "turbo"
    if hasattr(pipe, "is_distilled"):
        pipe.is_distilled = is_distilled
    register_to_config = getattr(pipe, "register_to_config", None)
    if callable(register_to_config):
        register_to_config(is_distilled=is_distilled)

    return pipe, {
        "source": "single",
        "variant": variant,
        "quantization": quantization,
        "component_source": component_source,
        **overlay,
    }


def _run(request: dict, output_dir: Path) -> dict:
    import torch

    prompt = str(request["prompt"])
    width = _snap(int(request.get("width") or 1024))
    height = _snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 28))
    guidance = float(request.get("guidance") if request.get("guidance") is not None else 4.5)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") or 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    pipe, load_info = _load_pipeline(request, dtype)

    total_vram_gb = 0.0
    if device == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total_vram_gb = 0.0
    offload = _configure_offload(pipe, device, total_vram_gb)

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index in range(num_images):
        generator_device = device if device == "cuda" else "cpu"
        result = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device=generator_device).manual_seed(seed + index),
        )
        image = result.images[0]
        path = output_dir / f"krea2_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    return {
        "ok": True,
        "images": saved,
        "seed": seed,
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
    args = parser.parse_args()

    result_path = Path(args.result)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = _run(request, Path(args.output_dir))
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
