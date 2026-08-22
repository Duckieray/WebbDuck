"""Krea 2 single-transformer checkpoint inspection and key mapping.

Community Krea 2 checkpoints commonly use the original/ComfyUI module names
(`blocks.*`, `txtfusion.*`, `first`, `last`, ...), while Diffusers exposes
`Krea2Transformer2DModel` names.  Keep that conversion backend-local so the
model catalog and UI remain architecture agnostic.
"""

from __future__ import annotations

import json
from typing import Any


_SUPPORTED_FP8_FORMATS = {
    "float8_e4m3fn",
    "float8_e5m2",
    "fp8",
    "fp8_scaled",
}


def strip_krea2_prefix(key: str) -> str:
    value = str(key)
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _map_attention_and_mlp(key: str) -> str:
    replacements = (
        (".attn.wq.", ".attn.to_q."),
        (".attn.wk.", ".attn.to_k."),
        (".attn.wv.", ".attn.to_v."),
        (".attn.gate.", ".attn.to_gate."),
        (".attn.wo.", ".attn.to_out.0."),
        (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
        (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
        (".mlp.gate.", ".ff.gate."),
        (".mlp.up.", ".ff.up."),
        (".mlp.down.", ".ff.down."),
        (".prenorm.scale", ".norm1.weight"),
        (".postnorm.scale", ".norm2.weight"),
    )
    out = key
    for source, target in replacements:
        out = out.replace(source, target)
    return out


def map_krea2_source_key(source_key: str) -> tuple[str | None, str | None]:
    """Map one original/ComfyUI Krea key to Diffusers and optional reshape mode.

    Quantization sidecar tensors are intentionally ignored here.  The runtime
    consumes them together with the associated weight tensor.
    """
    key = strip_krea2_prefix(source_key)
    if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
        return None, None

    direct_prefixes = (
        ("first.", "img_in."),
        ("tmlp.0.", "time_embed.linear_1."),
        ("tmlp.2.", "time_embed.linear_2."),
        ("tproj.1.", "time_mod_proj."),
        ("txtmlp.0.scale", "txt_in.norm.weight"),
        ("txtmlp.1.", "txt_in.linear_1."),
        ("txtmlp.3.", "txt_in.linear_2."),
        ("txtfusion.projector.", "text_fusion.projector."),
        ("last.norm.scale", "final_layer.norm.weight"),
        ("last.linear.", "final_layer.linear."),
    )
    for source, target in direct_prefixes:
        if key == source:
            return target, None
        if key.startswith(source):
            return target + key[len(source) :], None

    if key == "last.modulation.lin":
        return "final_layer.scale_shift_table", "two_rows"

    if key.startswith("blocks."):
        mapped = "transformer_blocks." + key[len("blocks.") :]
        if mapped.endswith(".mod.lin"):
            return mapped[: -len(".mod.lin")] + ".scale_shift_table", "six_rows"
        return _map_attention_and_mlp(mapped), None

    for source_prefix, target_prefix in (
        ("txtfusion.layerwise_blocks.", "text_fusion.layerwise_blocks."),
        ("txtfusion.refiner_blocks.", "text_fusion.refiner_blocks."),
    ):
        if key.startswith(source_prefix):
            mapped = target_prefix + key[len(source_prefix) :]
            return _map_attention_and_mlp(mapped), None

    return None, None


def parse_comfy_quant_payload(raw: Any) -> dict[str, Any]:
    """Decode the uint8 JSON tensor used by ComfyUI quantized checkpoints."""
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
    """Return fp8 / convrot / unsupported for a Comfy quant descriptor."""
    fmt = str(config.get("format") or "").strip().lower()
    if bool(config.get("convrot")):
        return "convrot"
    if fmt in _SUPPORTED_FP8_FORMATS or fmt.startswith("float8"):
        return "fp8"
    return "unsupported" if fmt else "unknown"
