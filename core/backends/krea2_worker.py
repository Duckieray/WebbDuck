"""Standalone Krea 2 Diffusers worker.

Full Diffusers checkpoints load normally. Community single-file transformer
checkpoints are streamed directly into a transformer scaffold created from the
official Krea config on the meta device. Scaled FP8 single-file weights stay
quantized when supported instead of being expanded to a full BF16 transformer.

Execution is hardware-driven: the worker profiles the actual accelerator and
free VRAM, runs text encoding / denoising / VAE decode as separate phases, and
chooses resident or offloaded execution from live memory headroom rather than a
hard-coded GPU model.
"""

from __future__ import annotations

import argparse
import gc
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
from runtime_hardware import detect_torch_hardware


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
    """Linear layer that retains a checkpoint's FP8 weight + scale storage.

    Each layer is dequantized only for its current matmul. This is deliberately
    storage-oriented rather than a claim of native FP8 tensor-core execution;
    it removes the full-model BF16 residency cost while preserving Diffusers'
    existing Krea transformer graph.
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


def _assign_target_tensor(transformer: nn.Module, target_key: str, value: torch.Tensor) -> None:
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
    preserve_fp8: bool = True,
) -> dict[str, Any]:
    """Stream one Krea transformer checkpoint into a Diffusers scaffold."""
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

                    if preserve_fp8 and _install_scaled_fp8_linear(
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
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel

    config = Krea2Transformer2DModel.load_config(component_source, subfolder="transformer")
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


def _module_storage_gb(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total_bytes = 0
    seen: set[int] = set()
    for tensor in [*module.parameters(), *module.buffers()]:
        if tensor is None or bool(getattr(tensor, "is_meta", False)):
            continue
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        total_bytes += int(tensor.numel()) * int(tensor.element_size())
    return float(total_bytes) / 1024**3


def _activation_reserve_gb(total_vram_gb: float, width: int, height: int) -> float:
    megapixels = max(0.25, float(width * height) / 1_000_000.0)
    return max(2.0, float(total_vram_gb) * 0.10, 1.35 + 0.85 * megapixels)


def _torch_device(hardware: dict[str, Any]) -> str:
    accelerator = str(hardware.get("accelerator") or "cpu")
    if accelerator in {"cuda", "rocm"}:
        return "cuda"
    if accelerator == "mps":
        return "mps"
    return "cpu"


def _auto_execution_mode(
    *,
    preserved_fp8_linears: int,
    hardware: dict[str, Any],
    transformer_storage_gb: float,
    reserve_gb: float,
) -> str:
    device = _torch_device(hardware)
    if device == "cpu":
        return "cpu"

    free_vram_gb = float(hardware.get("free_vram_gb") or 0.0)
    enough_headroom = free_vram_gb >= transformer_storage_gb + reserve_gb
    fp8_supported = bool(hardware.get("fp8_storage"))

    if preserved_fp8_linears > 0 and fp8_supported and enough_headroom:
        return "resident-fp8"
    if preserved_fp8_linears == 0 and enough_headroom:
        return "resident"
    if str(hardware.get("accelerator")) in {"cuda", "rocm"}:
        return "transformer-block"
    # MPS does not currently have a validated streamed group-offload path here.
    # Prefer correctness on CPU when a dense transformer cannot fit resident.
    return "cpu"


def _configure_execution(
    pipe: Any,
    *,
    hardware: dict[str, Any],
    load_info: dict[str, Any],
    transformer_storage_gb: float,
    reserve_gb: float,
    mode_override: str | None = None,
) -> str:
    """Configure the denoiser only; text encoder/VAE are separate phases."""
    device = _torch_device(hardware)
    mode = str(mode_override or os.getenv("WEBBDUCK_KREA2_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        mode = _auto_execution_mode(
            preserved_fp8_linears=int(load_info.get("preserved_fp8_linears") or 0),
            hardware=hardware,
            transformer_storage_gb=transformer_storage_gb,
            reserve_gb=reserve_gb,
        )

    if mode == "cpu":
        pipe.transformer.to("cpu")
        return "cpu"

    if mode in {"resident-fp8", "resident", "gpu", "none"}:
        pipe.transformer.to(device)
        return "resident-fp8" if int(load_info.get("preserved_fp8_linears") or 0) else "resident"

    onload_device = torch.device(device)
    use_stream = bool(hardware.get("stream_prefetch")) and device == "cuda"

    if mode in {"transformer-block", "block", "group-block"}:
        blocks_default = "1" if use_stream else "2"
        blocks_raw = str(os.getenv("WEBBDUCK_KREA2_BLOCKS_PER_GROUP", blocks_default)).strip()
        try:
            blocks_per_group = max(1, int(blocks_raw))
        except Exception:
            blocks_per_group = int(blocks_default)
        if use_stream:
            # Diffusers stream prefetch supports one block per group.
            blocks_per_group = 1

        ram_gb = _system_ram_gb()
        low_cpu_mem_default = "1" if 0 < ram_gb < 48.0 else "0"
        low_cpu_mem_usage = str(
            os.getenv("WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM", low_cpu_mem_default)
        ).strip().lower() in {"1", "true", "yes", "on"}

        try:
            pipe.transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=blocks_per_group,
                use_stream=use_stream,
                record_stream=use_stream,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            suffix = "stream" if use_stream else "sync"
            return f"transformer-block-{suffix}-{blocks_per_group}"
        except Exception:
            pipe.transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=use_stream,
                record_stream=use_stream,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            return "transformer-leaf-stream-fallback" if use_stream else "transformer-leaf-sync-fallback"

    if mode in {"group", "stream", "group-stream", "leaf"}:
        pipe.transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=use_stream,
            record_stream=use_stream,
            low_cpu_mem_usage=False,
        )
        return "transformer-leaf-stream" if use_stream else "transformer-leaf-sync"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device=device)
        return "sequential"

    if mode in {"model", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device=device)
        return "model"

    raise RuntimeError(f"Unknown WEBBDUCK_KREA2_OFFLOAD mode: {mode}")


def _load_pipeline(
    request: dict,
    dtype: Any,
    hardware: dict[str, Any],
    report: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
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
        preserve_fp8=bool(hardware.get("fp8_storage")),
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
    hardware: dict[str, Any],
    report: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, str]:
    """Encode once using the fastest text-encoder strategy that fits live VRAM."""
    device = _torch_device(hardware)
    encoder = pipe.text_encoder
    encoder_storage_gb = _module_storage_gb(encoder)
    free_vram_gb = float(hardware.get("free_vram_gb") or 0.0)
    encode_reserve_gb = max(1.25, float(hardware.get("total_vram_gb") or 0.0) * 0.08)
    encode_mode = "cpu"
    encode_device = torch.device("cpu")

    if device != "cpu" and free_vram_gb >= encoder_storage_gb + encode_reserve_gb:
        if callable(report):
            report("Loading Krea text encoder to GPU", 0.37)
        encoder.eval().requires_grad_(False)
        encoder.to(device)
        encode_mode = "resident"
        encode_device = torch.device(device)
    elif device == "cuda":
        try:
            from diffusers.hooks import apply_group_offloading

            use_stream = bool(hardware.get("stream_prefetch"))
            if callable(report):
                report("Streaming Krea text encoder", 0.37)
            apply_group_offloading(
                encoder,
                onload_device=torch.device(device),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=use_stream,
                record_stream=use_stream,
                low_cpu_mem_usage=_system_ram_gb() < 48.0,
            )
            encode_mode = "block-stream" if use_stream else "block-sync"
            encode_device = torch.device(device)
        except Exception:
            encode_mode = "cpu-fallback"
            encode_device = torch.device("cpu")

    if callable(report):
        report("Encoding Krea prompt", 0.40)
    with torch.inference_mode():
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            prompt=prompt,
            device=encode_device,
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
                device=encode_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

    # Conditioning is cheap to keep on CPU between phases and doing so makes
    # the live free-VRAM measurement for denoiser selection truthful.
    prompt_embeds = prompt_embeds.to("cpu")
    prompt_mask = prompt_mask.to("cpu")
    if negative_embeds is not None:
        negative_embeds = negative_embeds.to("cpu")
    if negative_mask is not None:
        negative_mask = negative_mask.to("cpu")

    if callable(report):
        report("Releasing Krea text encoder", 0.48)
    pipe.text_encoder = None
    del encoder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return prompt_embeds, prompt_mask, negative_embeds, negative_mask, encode_mode


def _release_transformer(pipe: Any, device: str) -> None:
    transformer = getattr(pipe, "transformer", None)
    pipe.transformer = None
    del transformer
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _decode_latents(
    pipe: Any,
    vae: Any,
    latents: torch.Tensor,
    *,
    width: int,
    height: int,
    device: str,
) -> list[Any]:
    vae.to(device)
    latents = latents.to(device=device, dtype=vae.dtype)
    latents = pipe._unpack_latents(latents, height, width)
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    with torch.inference_mode():
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def _denoise_one(
    pipe: Any,
    *,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    device: str,
    index: int,
    total_denoise_steps: int,
    report: Any,
) -> torch.Tensor:
    prompt_gpu = prompt_embeds.to(device)
    mask_gpu = prompt_mask.to(device)
    negative_gpu = negative_embeds.to(device) if negative_embeds is not None else None
    negative_mask_gpu = negative_mask.to(device) if negative_mask is not None else None

    def on_step_end(
        _pipe: Any,
        step_index: int,
        _timestep: Any,
        callback_kwargs: dict,
    ) -> dict:
        completed = index * steps + step_index + 1
        progress = 0.55 + (0.34 * completed / total_denoise_steps)
        report("Denoising with Krea 2", progress, completed, total_denoise_steps)
        return callback_kwargs

    call_kwargs: dict[str, Any] = {
        "prompt": None,
        "prompt_embeds": prompt_gpu,
        "prompt_embeds_mask": mask_gpu,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "callback_on_step_end": on_step_end,
        "output_type": "latent",
    }
    if guidance > 0:
        call_kwargs["negative_prompt_embeds"] = negative_gpu
        call_kwargs["negative_prompt_embeds_mask"] = negative_mask_gpu

    with torch.inference_mode():
        result = pipe(**call_kwargs)
    latents = result.images.detach().to("cpu")
    del prompt_gpu, mask_gpu, negative_gpu, negative_mask_gpu, result
    if device == "cuda":
        torch.cuda.empty_cache()
    return latents


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


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

    report("Profiling Krea hardware", 0.04)
    hardware = detect_torch_hardware(torch)
    device = _torch_device(hardware)
    if device == "cuda":
        dtype = torch.bfloat16 if bool(hardware.get("bf16")) else torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    report("Initializing Krea runtime", 0.06)
    load_started = time.perf_counter()
    pipe, load_info = _load_pipeline(request, dtype, hardware, report=report)
    timing["pipeline_load_seconds"] = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    prompt_embeds, prompt_mask, negative_embeds, negative_mask, encode_mode = _encode_prompt_phase(
        pipe,
        prompt=prompt,
        guidance=guidance,
        hardware=hardware,
        report=report,
    )
    timing["prompt_encode_seconds"] = time.perf_counter() - encode_started

    # The VAE is explicitly detached during denoising. It is loaded onto the GPU
    # only after the transformer has been released, preventing two large phases
    # from competing for constrained VRAM.
    vae = pipe.vae
    pipe.vae = None

    if device == "cuda":
        hardware = detect_torch_hardware(torch)
    transformer_storage_gb = _module_storage_gb(pipe.transformer)
    reserve_gb = _activation_reserve_gb(
        float(hardware.get("total_vram_gb") or 0.0),
        width,
        height,
    )

    report("Selecting Krea GPU profile", 0.50)
    setup_started = time.perf_counter()
    execution_mode = _configure_execution(
        pipe,
        hardware=hardware,
        load_info=load_info,
        transformer_storage_gb=transformer_storage_gb,
        reserve_gb=reserve_gb,
    )
    initial_execution_mode = execution_mode
    fallback_reason: str | None = None
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
        try:
            latent = _denoise_one(
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
        except Exception as exc:
            # A desktop compositor/browser/game can consume VRAM after profile
            # selection. Resident execution therefore gets one automatic retry
            # using the safer transformer-only block-offload profile.
            if device == "cuda" and execution_mode.startswith("resident") and _is_cuda_oom(exc):
                fallback_reason = "resident_oom"
                report("VRAM changed; retrying Krea with safe offload", 0.53)
                try:
                    pipe.transformer.to("cpu")
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                hardware = detect_torch_hardware(torch)
                execution_mode = _configure_execution(
                    pipe,
                    hardware=hardware,
                    load_info=load_info,
                    transformer_storage_gb=transformer_storage_gb,
                    reserve_gb=reserve_gb,
                    mode_override="transformer-block",
                )
                latent = _denoise_one(
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
            else:
                raise
        latent_batches.append(latent)

    timing["denoise_seconds"] = time.perf_counter() - denoise_started

    del prompt_embeds, prompt_mask, negative_embeds, negative_mask
    report("Releasing Krea denoiser", 0.90, total_denoise_steps, total_denoise_steps)
    _release_transformer(pipe, device)

    report("Decoding Krea image", 0.92, total_denoise_steps, total_denoise_steps)
    decode_started = time.perf_counter()
    pipe.vae = vae
    if max(width, height) > 1536:
        try:
            vae.enable_tiling()
        except Exception:
            pass

    for index, latents in enumerate(latent_batches):
        images = _decode_latents(
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
            **load_info,
            "offload": execution_mode,
            "initial_offload": initial_execution_mode,
            "fallback_reason": fallback_reason,
            "text_encoder_mode": encode_mode,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "hardware": runtime_hardware,
            "transformer_storage_gb": round(transformer_storage_gb, 2),
            "activation_reserve_gb": round(reserve_gb, 2),
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
