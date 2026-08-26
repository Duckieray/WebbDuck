"""Architecture-neutral helpers for checkpoint quantization metadata."""

from __future__ import annotations

import json
from typing import Any


_SUPPORTED_FP8_FORMATS = {
    "float8_e4m3fn",
    "float8_e5m2",
    "fp8",
    "fp8_scaled",
}


def parse_comfy_quant_payload(raw: Any) -> dict[str, Any]:
    """Decode a uint8 JSON tensor used by Comfy-compatible quantized checkpoints."""
    if raw is None:
        return {}
    try:
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if isinstance(values, list):
            payload = bytes(int(value) for value in values)
        elif isinstance(values, (bytes, bytearray)):
            payload = bytes(values)
        else:
            return {}
        parsed = json.loads(payload)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify_comfy_quant(config: dict[str, Any]) -> str:
    """Classify quant metadata without choosing an architecture-specific loader."""
    fmt = str(config.get("format") or "").strip().lower()
    if bool(config.get("convrot")) or fmt.startswith("convrot"):
        return "convrot"
    if fmt in _SUPPORTED_FP8_FORMATS or fmt.startswith("float8"):
        return "fp8"
    return "unsupported" if fmt else "unknown"
