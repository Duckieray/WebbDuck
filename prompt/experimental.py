"""SDXL conditioning dispatch and refiner helpers."""

import torch

from webbduck.core.perf import stage_timer
from webbduck.prompt.conditioning import (
    build_sdxl_conditioning,
    encode_long_prompt,
    _expand_textual_inversion_prompt,
)


def build_sdxl_refiner_conditioning(pipe, prompt, prompt_2, negative):
    """Build conditioning for SDXL refiners (CLIP-G only), with long-prompt chunking."""
    device = pipe.device
    pos_text = _expand_textual_inversion_prompt(pipe, pipe.tokenizer_2, prompt_2 or prompt or "")
    neg_text = _expand_textual_inversion_prompt(pipe, pipe.tokenizer_2, negative or "")

    prompt_embeds, pooled_prompt_embeds = encode_long_prompt(
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        pos_text,
        device,
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_long_prompt(
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        neg_text,
        device,
    )

    if negative_prompt_embeds.shape[1] < prompt_embeds.shape[1]:
        pad = torch.zeros(
            negative_prompt_embeds.shape[0],
            prompt_embeds.shape[1] - negative_prompt_embeds.shape[1],
            negative_prompt_embeds.shape[2],
            device=negative_prompt_embeds.device,
            dtype=negative_prompt_embeds.dtype,
        )
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)
    elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
        pad = torch.zeros(
            prompt_embeds.shape[0],
            negative_prompt_embeds.shape[1] - prompt_embeds.shape[1],
            prompt_embeds.shape[2],
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds


def build_sdxl_conditioning_dispatch(*, pipe, prompt, prompt_2, negative):
    """Standard SDXL conditioning dispatch (long-prompt chunking is always enabled)."""
    with stage_timer("conditioning_seconds"):
        return build_sdxl_conditioning(
            pipe=pipe,
            prompt=prompt,
            prompt_2=prompt_2,
            negative=negative,
        )
