"""Cheap structural inspection for single-file image checkpoints.

The catalog must not classify a multi-gigabyte safetensors file from its folder
name alone. Safetensors exposes tensor names/dtypes in its JSON header, so we
can recognize formats without materializing model weights during discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.quantization import classify_comfy_quant, parse_comfy_quant_payload


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
            for key in quant_keys[:16]:
                parsed = parse_comfy_quant_payload(handle.get_tensor(key))
                if parsed:
                    configs.append(parsed)
    except Exception:
        return configs
    return configs


def _metadata_quant_configs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("_quantization_metadata")
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
    return [dict(value) for value in layers.values() if isinstance(value, dict)][:32]


def _detect_flux2_klein(normalized_keys: set[str], stem: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Recognize a FLUX.2 (Klein) transformer single-file checkpoint.

    FLUX.2 Klein transformers expose fused single-stream blocks
    (``single_blocks.N.linear1/linear2`` -> in diffusers ``to_qkv_mlp_proj``) and
    dual text/image stream modulation that FLUX.1 lacks. The presence of BOTH
    ``double_stream_modulation_img`` and ``double_stream_modulation_txt`` is a
    strong, family-exclusive fingerprint. We additionally require the shared
    transformer skeleton (``img_in``/``txt_in``/``time_in``/``final_layer`` and
    ``double_blocks``/``single_blocks``) so arbitrary modulation tensors don't
    false-positive.

    The bare top-level ``double_blocks.*`` naming (no ``transformer.`` prefix) is
    the canonical FLUX.2 export layout; the ``transformer.``-prefixed variant is
    accepted through the same normalization.
    """
    has_skeleton = (
        "img_in.weight" in normalized_keys
        and "txt_in.weight" in normalized_keys
        and "double_blocks.0.img_attn.qkv.weight" in normalized_keys
        and "single_blocks.0.linear1.weight" in normalized_keys
    )
    is_klein = (
        "double_stream_modulation_img.lin.weight" in normalized_keys
        and "double_stream_modulation_txt.lin.weight" in normalized_keys
        and has_skeleton
    )
    if not is_klein:
        return None
    return {"method": "tensor_keys", "confidence": "high", "family": "flux2"}


def inspect_single_image_checkpoint(path: Path) -> tuple[str, dict[str, Any]]:
    """Return architecture + internal detection metadata for one safetensors file."""
    path = Path(path)
    header = _read_safetensors_header(path)
    if not header:
        return UNKNOWN_ARCHITECTURE, {"method": "invalid_safetensors", "confidence": "none"}

    tensor_entries = {
        key: value
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }
    normalized_keys = {_strip_prefix(key) for key in tensor_entries}
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    stem = path.stem.lower()

    is_krea2 = (
        "txtfusion.projector.weight" in normalized_keys
        and any(key.startswith("blocks.0.attn.wq.") for key in normalized_keys)
        and "first.weight" in normalized_keys
    )

    quantization = "none"
    quant_configs = [
        *_read_comfy_quant_configs(path, list(tensor_entries)),
        *_metadata_quant_configs(metadata),
    ]
    quant_kinds = {classify_comfy_quant(config) for config in quant_configs}
    if "convrot" in quant_kinds or "convrot" in stem:
        quantization = "convrot"
    elif "fp8" in quant_kinds:
        quantization = "fp8_scaled"
    elif "unsupported" in quant_kinds:
        quantization = "unsupported"
    else:
        dtypes = {str(entry.get("dtype") or "").upper() for entry in tensor_entries.values()}
        if any(dtype.startswith("F8") for dtype in dtypes) or "fp8" in stem:
            quantization = "fp8"
        elif any(dtype in {"I8", "U8"} for dtype in dtypes):
            quantization = "unsupported"
        elif "BF16" in dtypes:
            quantization = "bf16"
        elif "F16" in dtypes:
            quantization = "fp16"

    metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items()).lower()
    variant_text = f"{stem} {metadata_text}"
    variant = "turbo" if ("turbo" in variant_text or "distill" in variant_text or "tdm" in variant_text) else "base"

    if is_krea2:
        return "krea2", {
            "method": "tensor_keys",
            "confidence": "high",
            "variant": variant,
            "quantization": quantization,
        }

    flux2 = _detect_flux2_klein(normalized_keys, stem, metadata)
    if flux2:
        flux2["quantization"] = quantization
        flux2["variant"] = variant
        flux2["parameter_size"] = _flux_parameter_size(stem)
        return "flux", flux2

    return UNKNOWN_ARCHITECTURE, {"method": "tensor_keys", "confidence": "none"}


def _flux_parameter_size(stem: str) -> str | None:
    """Best-effort parameter size (4B/9B) from a FLUX checkpoint filename.

    Sizes appear as ``9b``/``4b`` after a path separator (``-9b``, ``_9b``),
    directly appended to a family token (``Klein9b``), or touching other non-digit
    characters (``x4b``). A digit must not directly precede the size digit so
    revision-style runs like ``v14`` are not misread as a size.
    """
    import re

    match = re.search(r"(?<![0-9])[49]b(?![0-9])", stem, re.IGNORECASE)
    return match.group(0).upper() if match else None
