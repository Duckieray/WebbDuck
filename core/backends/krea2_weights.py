"""Krea 2 original/ComfyUI to Diffusers transformer key mapping.

Community Krea 2 checkpoints commonly use the original/ComfyUI module names
(`blocks.*`, `txtfusion.*`, `first`, `last`, ...), while Diffusers exposes
`Krea2Transformer2DModel` names. Keep this conversion backend-local so model
discovery remains independent from execution.
"""

from __future__ import annotations


def strip_krea2_prefix(key: str) -> str:
    value = str(key)
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


# Diffusers exposes the text-fusion stacks as flat ``nn.ModuleList`` attributes
# (``layerwise_blocks`` / ``refiner_blocks``). Kohya exports flatten the compound
# name into ``layerwise.blocks`` / ``refiner.blocks`` (its key format replaces
# every dot with an underscore), and some original/ComfyUI-era exports ship the
# split spelling natively. Both spellings must land on the flat Diffusers name.
_TEXT_FUSION_BLOCK_SPELLINGS = (
    (".layerwise.blocks.", ".layerwise_blocks."),
    (".refiner.blocks.", ".refiner_blocks."),
    (".layerwise.blocks", ".layerwise_blocks"),
    (".refiner.blocks", ".refiner_blocks"),
)


def normalize_krea2_text_fusion_blocks(key: str) -> str:
    """Collapse split ``layerwise.blocks``/``refiner.blocks`` text-fusion names.

    Operates on the ``txtfusion.`` or already-rewritten ``text_fusion.`` tail
    and returns the flat ``layerwise_blocks``/``refiner_blocks`` spelling
    Diffusers' ``Krea2TextFusion`` exposes. Shared by the checkpoint, user-LoRA,
    and identity-LoRA key converters so a compound module name cannot resolve
    differently depending on which converter saw it.
    """
    value = str(key)
    for source, target in _TEXT_FUSION_BLOCK_SPELLINGS:
        if source in value:
            value = value.replace(source, target)
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
    """Map one original/ComfyUI Krea key to Diffusers and optional reshape mode."""
    key = strip_krea2_prefix(source_key)
    if key.endswith(
        (
            ".comfy_quant",
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
            ".pre_quant_scale",
        )
    ):
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
        ("txtfusion.layerwise.blocks.", "text_fusion.layerwise_blocks."),
        ("txtfusion.refiner.blocks.", "text_fusion.refiner_blocks."),
    ):
        if key.startswith(source_prefix):
            mapped = target_prefix + key[len(source_prefix) :]
            return _map_attention_and_mlp(mapped), None

    return None, None
