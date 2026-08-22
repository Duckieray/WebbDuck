"""Cheap structural inspection for single-file image checkpoints.

The catalog must not classify a multi-gigabyte safetensors file from its folder
name alone.  Safetensors exposes all tensor names/dtypes in its JSON header, so
we can recognize formats without materializing model weights during discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.backends.krea2_weights import classify_comfy_quant, parse_comfy_quant_payload


UNKNOWN_ARCHITECTURE = "unknown"
_MAX_HEADER_BYTES = 64 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return {}
            header_length = int.from_bytes(raw_length, "little", signed=False)
            if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
                return {}
            raw_header = handle.read(header_length)
        value = json.loads(raw_header.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _strip_prefix(key: str) -> str:
    value = str(key)
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _read_comfy_quant_configs(path: Path, keys: list[str]) -> list[dict[str, Any]]:
    quant_keys = [key for key in keys if key.endswith(".comfy_quant")]
    if not quant_keys:
        return []
    try:
        from safetensors import safe_open
    except Exception:
        return []

    configs: list[dict[str, Any]] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            # A handful of descriptors is enough to classify the file. Avoid
            # needlessly decoding hundreds of duplicate per-layer descriptors.
            for key in quant_keys[:16]:
                parsed = parse_comfy_quant_payload(handle.get_tensor(key))
                if parsed:
                    configs.append(parsed)
    except Exception:
        return configs
    return configs


def inspect_single_image_checkpoint(path: Path) -> tuple[str, dict[str, Any]]:
    """Return architecture + internal detection metadata for one safetensors file."""
    path = Path(path)
    header = _read_safetensors_header(path)
    if not header:
        return UNKNOWN_ARCHITECTURE, {"method": "invalid_safetensors", "confidence": "none"}

    tensor_entries = {key: value for key, value in header.items() if key != "__metadata__" and isinstance(value, dict)}
    normalized_keys = {_strip_prefix(key) for key in tensor_entries}
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}

    is_krea2 = (
        "txtfusion.projector.weight" in normalized_keys
        and any(key.startswith("blocks.0.attn.wq.") for key in normalized_keys)
        and "first.weight" in normalized_keys
    )
    if not is_krea2:
        return UNKNOWN_ARCHITECTURE, {"method": "tensor_keys", "confidence": "none"}

    quantization = "none"
    quant_configs = _read_comfy_quant_configs(path, list(tensor_entries))
    quant_kinds = {classify_comfy_quant(config) for config in quant_configs}
    stem = path.stem.lower()
    if "convrot" in quant_kinds or "convrot" in stem:
        quantization = "convrot"
    elif "fp8" in quant_kinds:
        quantization = "fp8_scaled"
    else:
        dtypes = {str(entry.get("dtype") or "").upper() for entry in tensor_entries.values()}
        if any(dtype.startswith("F8") for dtype in dtypes) or "fp8" in stem:
            quantization = "fp8"
        elif "BF16" in dtypes:
            quantization = "bf16"
        elif "F16" in dtypes:
            quantization = "fp16"

    metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items()).lower()
    variant_text = f"{stem} {metadata_text}"
    variant = "turbo" if ("turbo" in variant_text or "distill" in variant_text) else "base"

    return "krea2", {
        "method": "tensor_keys",
        "confidence": "high",
        "variant": variant,
        "quantization": quantization,
    }
