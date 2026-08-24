"""Standalone Krea 2 Diffusers worker.

Full Diffusers checkpoints load normally. Community single-file transformer
checkpoints are streamed directly into a transformer scaffold created from the
official Krea config on the meta device. Scaled FP8 single-file weights stay
quantized in GPU memory instead of being expanded to a full BF16 transformer.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ScaledFP8Linear(nn.Module):
    """Linear layer that keeps scaled FP8 checkpoint weights resident.

    Krea community checkpoints commonly store large linear weights as FP8 plus
    a scale tensor. Keeping that representation avoids expanding the complete
    13B transformer to BF16. Each layer is dequantized locally on the GPU only
    for the duration of its matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        qweight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qweight is None:
            raise RuntimeError("Scaled FP8 Krea linear is missing qweight")
        weight = self.qweight.to(dtype=x.dtype)
        if self.scale is not None:
            scale = self.scale.to(dtype=x.dtype)
            if scale.ndim == 1 and int(scale.numel()) == self.out_features:
                scale = scale.reshape(self.out_features, 1)
            weight = weight * scale
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


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


def _module_and_leaf(root: nn.Module, tensor_key: str) -> tuple[nn.Module, str]:
    module_path, _, leaf = tensor_key.rpartition(".")
    module = root.get_submodule(module_path) if module_path else root
    return module, leaf


def _live_target_tensor(root: nn.Module, tensor_key: str) -> torch.Tensor | None:
    module, leaf = _module_and_leaf(root, tensor_key)
    if leaf in module._parameters:
        return module._parameters[leaf]
    if leaf in module._buffers:
        return module._buffers[leaf]
    value = getattr(module, leaf, None)
    return value if isinstance(value, torch.Tensor) else None


def _replace_module(root: nn.Module, module_path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = module_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, replacement)


def _install_scaled_fp8_linear(
    transformer: nn.Module,
    target_key: str,
    source: torch.Tensor,
    scale: torch.Tensor | None,
    reshape_mode: str | None,
) -> bool:
    """Replace one target Linear with a scaled-FP8 resident representation."""
    if reshape_mode is not None or not target_key.endswith(".weight"):
        return False
    if not str(source.dtype).startswith("torch.float8"):
        return False

    module_path = target_key[: -len(".weight")]
    try:
        current = transformer.get_submodule(module_path)
    except Exception:
        return False
    if not isinstance(current, (nn.Linear, ScaledFP8Linear)):
        return False

    expected_shape = (int(current.out_features), int(current.in_features))
    if tuple(source.shape) != expected_shape:
        return False

    preserved_bias: torch.Tensor | None = None
    if getattr(current, "bias", None) is not None:
        bias = current.bias
        if isinstance(bias, torch.Tensor) and not bool(getattr(bias, "is_meta", False)):
            preserved_bias = bias.detach().to(device="cpu")

    normalized_scale = scale
    if normalized_scale is None:
        normalized_scale = torch.ones((), dtype=torch.float32)
    normalized_scale = normalized_scale.detach().to(device="cpu").clone()
    if normalized_scale.ndim == 1 and int(normalized_scale.numel()) == expected_shape[0]:
        normalized_scale = normalized_scale.reshape(expected_shape[0], 1)

    replacement = ScaledFP8Linear(
        expected_shape[1],
        expected_shape[0],
        qweight=source.detach().to(device="cpu").clone(),
        scale=normalized_scale,
        bias=preserved_bias,
    )
    _replace_module(transformer, module_path, replacement)
    return True


def _assign_target_tensor(
    transformer: nn.Module,
    target_key: str,
    value: torch.Tensor,
) -> None:
    """Materialize a meta parameter/buffer or copy into an allocated target."""
    module, leaf = _module_and_leaf(transformer, target_key)
    if isinstance(module, ScaledFP8Linear) and leaf == "bias":
        module.bias = value.detach().to(device="cpu")
        return

    target = _live_target_tensor(transformer, target_key)
    if target is None:
        raise RuntimeError(f"Krea target tensor {target_key!r} is unavailable during overlay")

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
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_path)
    initial_state = transformer.state_dict(keep_vars=True)
    expected = set(initial_state)
    mapped: set[str] = set()
    ignored: list[str] = []
    preserved_fp8_linears = 0

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
                if target_key is None and normalized in expected:
                    target_key = normalized
                if target_key is None:
                    ignored.append(source_key)
                else:
                    if target_key not in expected:
                        raise RuntimeError(
                            f"Krea checkpoint tensor {source_key!r} maps to unknown Diffusers key {target_key!r}."
                        )

                    source = handle.get_tensor(source_key)
                    prefix = _source_layer_prefix(source_key)
                    scale_key = f"{prefix}.weight_scale" if prefix else ""
                    scale = handle.get_tensor(scale_key) if scale_key and scale_key in keys else None

                    if _install_scaled_fp8_linear(
                        transformer,
                        target_key,
                        source,
                        scale,
                        reshape_mode,
                    ):
                        preserved_fp8_linears += 1
                        mapped.add(target_key)
                    else:
                        target = _live_target_tensor(transformer, target_key)
                        if target is None:
                            if isinstance(
                                transformer.get_submodule(target_key.rpartition(".")[0]),
                                ScaledFP8Linear,
                            ) and target_key.endswith(".weight"):
                                mapped.add(target_key)
                                target = None
                            else:
                                raise RuntimeError(
                                    f"Krea target tensor {target_key!r} disappeared during overlay"
                                )
                        if target is not None:
                            effective_dtype = target_dtype if target_dtype is not None else target.dtype
                            value = _dequantize_source_tensor(
                                handle,
                                keys,
                                source_key,
                                source,
                                effective_dtype,
                            )
                            value = _reshape_for_target(value, target, reshape_mode, source_key)
                            _assign_target_tensor(transformer, target_key, value)
                            mapped.add(target_key)
                            del value
                    del source

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
        "preserved_fp8_linears": preserved_fp8_linears,
        "execution_quantization": "scaled-fp8" if preserved_fp8_linears else "dense",
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


def _system_ram_gb() -> float:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return float(pages * page_size) / 1024**3
    except Exception:
        return 0.0


def _auto_execution_mode(
    *,
    preserved_fp8_linears: int,
    total_vram_gb: float,
) -> str:
    if preserved_fp8_linears > 0 and total_vram_gb >= 15.0:
        return "resident-fp8"
    if total_vram_gb < 32.0:
        return "transformer-block"
    return "none"


def _configure_execution(
    pipe: Any,
    *,
    device: str,
    total_vram_gb: float,
    load_info: dict[str, Any],
) -> str:
    """Configure only the denoiser/VAE after prompt encoding has finished."""
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("WEBBDUCK_KREA2_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        mode = _auto_execution_mode(
            preserved_fp8_linears=int(load_info.get("preserved_fp8_linears") or 0),
            total_vram_gb=total_vram_gb,
        )

    pipe.vae.to("cuda")

    if mode in {"resident-fp8", "resident", "gpu", "none"}:
        pipe.transformer.to("cuda")
        return "resident-fp8" if int(load_info.get("preserved_fp8_linears") or 0) else "resident"

    if mode in {"transformer-block", "block", "group-block"}:
        blocks_raw = str(os.getenv("WEBBDUCK_KREA2_BLOCKS_PER_GROUP", "2")).strip()
        try:
            blocks_per_group = max(1, int(blocks_raw))
        except Exception:
            blocks_per_group = 2

        ram_gb = _system_ram_gb()
        low_cpu_mem_default = "1" if 0 < ram_gb < 48.0 else "0"
        low_cpu_mem_usage = str(
            os.getenv("WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM", low_cpu_mem_default)
        ).strip().lower() in {"1", "true", "yes", "on"}

        try:
            pipe.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=blocks_per_group,
                use_stream=True,
                record_stream=True,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            return f"transformer-block-stream-{blocks_per_group}"
        except Exception:
            pipe.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
                record_stream=True,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            return "transformer-leaf-stream-fallback"

    if mode in {"group", "stream", "group-stream", "leaf"}:
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            record_stream=True,
            low_cpu_mem_usage=False,
        )
        return "transformer-leaf-stream"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"

    if mode in {"model", "cpu", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    raise RuntimeError(f"Unknown WEBBDUCK_KREA2_OFFLOAD mode: {mode}")


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
            report("Krea pipeline loaded", 0.34)
        return pipe, {
            "source": "diffusers",
            "variant": variant,
            "preserved_fp8_linears": 0,
            "execution_quantization": "dense",
        }

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
            report("Loading Krea checkpoint", 0.12 + (0.14 * fraction), current, total)

    overlay = overlay_single_file_transformer(
        transformer,
        model_path,
        target_dtype=dtype,
        progress_callback=overlay_progress,
    )
    if callable(report):
        report("Loading Krea support models", 0.28)
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
        report("Krea pipeline loaded", 0.34)
    return pipe, {
        "source": "single",
        "variant": variant,
        "quantization": quantization,
        "component_source": component_source,
        "component_transformer_weights_loaded": False,
        **overlay,
    }


def _encode_prompt_phase(
    pipe: Any,
    *,
    prompt: str,
    guidance: float,
    device: str,
    report: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Encode conditioning while the large Krea transformer remains on CPU."""
    target_device = torch.device(device)
    if callable(report):
        report("Loading Krea text encoder to GPU", 0.37)

    pipe.text_encoder.eval().requires_grad_(False)
    pipe.text_encoder.to(target_device)

    if callable(report):
        report("Encoding Krea prompt", 0.40)
    with torch.inference_mode():
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            prompt=prompt,
            device=target_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        negative_embeds = None
        negative_mask = None
        if guidance > 0:
            if callable(report):
                report("Encoding Krea negative prompt", 0.44)
            negative_embeds, negative_mask = pipe.encode_prompt(
                prompt="",
                device=target_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

    if callable(report):
        report("Freeing Krea text encoder", 0.48)
    pipe.text_encoder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    return prompt_embeds, prompt_mask, negative_embeds, negative_mask


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    worker_started = time.perf_counter()

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

    encode_started = time.perf_counter()
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = _encode_prompt_phase(
        pipe,
        prompt=prompt,
        guidance=guidance,
        device=device,
        report=report,
    )
    timing["prompt_encode_seconds"] = time.perf_counter() - encode_started

    report("Optimizing Krea denoiser", 0.50)
    setup_started = time.perf_counter()
    execution_mode = _configure_execution(
        pipe,
        device=device,
        total_vram_gb=total_vram_gb,
        load_info=load_info,
    )
    timing["execution_setup_seconds"] = time.perf_counter() - setup_started
    timing["offload_setup_seconds"] = timing["execution_setup_seconds"]

    if max(width, height) > 1536:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    total_denoise_steps = max(1, steps * num_images)
    denoise_started = time.perf_counter()

    for index in range(num_images):
        generator_device = device if device == "cuda" else "cpu"
        report(
            "Starting Krea denoising",
            0.54 + (0.41 * (index * steps) / total_denoise_steps),
            index * steps,
            total_denoise_steps,
        )

        def on_step_end(
            _pipe: Any,
            step_index: int,
            _timestep: Any,
            callback_kwargs: dict,
        ) -> dict:
            completed = index * steps + step_index + 1
            progress = 0.55 + (0.40 * completed / total_denoise_steps)
            stage = "Denoising with Krea 2"
            if completed >= total_denoise_steps:
                stage = "Decoding Krea image"
            report(stage, progress, completed, total_denoise_steps)
            return callback_kwargs

        call_kwargs: dict[str, Any] = {
            "prompt": None,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_mask,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator(device=generator_device).manual_seed(seed + index),
            "callback_on_step_end": on_step_end,
        }
        if guidance > 0:
            call_kwargs["negative_prompt_embeds"] = negative_embeds
            call_kwargs["negative_prompt_embeds_mask"] = negative_mask

        with torch.inference_mode():
            result = pipe(**call_kwargs)
        image = result.images[0]
        report("Writing Krea image", 0.97, (index + 1) * steps, total_denoise_steps)
        write_started = time.perf_counter()
        path = output_dir / f"krea2_{index:03d}.png"
        image.save(path)
        timing["image_write_seconds"] = timing.get("image_write_seconds", 0.0) + (
            time.perf_counter() - write_started
        )
        saved.append(str(path))

    timing["denoise_decode_seconds"] = time.perf_counter() - denoise_started
    timing["inference_seconds"] = timing["prompt_encode_seconds"] + timing["denoise_decode_seconds"]
    timing["worker_total_seconds"] = time.perf_counter() - worker_started
    report("Returning Krea image", 0.975, total_denoise_steps, total_denoise_steps)

    transformer_passes_per_step = 2 if guidance > 0 else 1
    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "timing": {key: round(value, 6) for key, value in timing.items()},
        "runtime": {
            **load_info,
            "offload": execution_mode,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "total_vram_gb": round(total_vram_gb, 2),
            "system_ram_gb": round(_system_ram_gb(), 2),
            "transformer_forward_passes": steps * num_images * transformer_passes_per_step,
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
