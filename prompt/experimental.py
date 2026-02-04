"""Experimental two-pass prompt splitting and conditioning."""

import torch
import logging
from webbduck.prompt.management import truncate_to_tokens
from webbduck.prompt.conditioning import build_sdxl_conditioning

log = logging.getLogger(__name__)


def split_prompt_for_two_pass(tokenizer, text, max_tokens=77):
    """
    Split text into base_prompt (first max_tokens) and late_prompt (remainder).
    
    Returns:
        base_prompt: First max_tokens tokens
        late_prompt: Remaining tokens (or None if fits in max_tokens)
    """
    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]

    if len(tokens) <= max_tokens:
        return text, None

    base_tokens = tokens[:max_tokens]
    late_tokens = tokens[max_tokens:]

    base_prompt = tokenizer.decode(
        base_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    late_prompt = tokenizer.decode(
        late_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return base_prompt, late_prompt


def merge_chunk_embeddings(embeds: list[torch.Tensor], mode: str = "weighted"):
    """
    Merge multiple embedding chunks with weighting strategy.
    
    Args:
        embeds: List of [B, 77, D] tensors
        mode: 'mean' or 'weighted'
    """
    if len(embeds) == 1:
        return embeds[0]

    if mode == "mean":
        return torch.stack(embeds, dim=0).mean(dim=0)

    if mode == "weighted":
        weights = torch.linspace(0.2, 1.0, steps=len(embeds))
        weights = weights / weights.sum()

        stacked = torch.stack(embeds, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0)

    raise ValueError(f"Unknown merge mode: {mode}")


def encode_chunks_experimental(tokenizer, text_encoder, chunks, device):
    """Experimental CLIP-L encoder with semantic chunk blending."""
    chunk_embeds = []

    for chunk in chunks:
        tokens = tokenizer(
            chunk,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = text_encoder(
                tokens.input_ids,
                return_dict=True,
                output_hidden_states=True,
            )

        chunk_embeds.append(out.hidden_states[-2])

    return merge_chunk_embeddings(chunk_embeds, mode="weighted")


def encode_chunks_experimental_2(tokenizer, text_encoder_2, chunks, device):
    """Experimental CLIP-G encoder with semantic chunk blending."""
    chunk_embeds = []

    for chunk in chunks:
        tokens = tokenizer(
            chunk,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = text_encoder_2(
                tokens.input_ids,
                return_dict=True,
                output_hidden_states=True,
            )

        chunk_embeds.append(out.hidden_states[-2])

    return merge_chunk_embeddings(chunk_embeds, mode="weighted")


def build_sdxl_conditioning_experimental(pipe, prompt, prompt_2, negative):
    """
    Build SDXL conditioning with experimental long-prompt support.
    
    WARNING: This uses chunked encoding with weighted blending.
    Only active when experimental_compress is enabled.
    """
    log.warning("⚠️ Using experimental prompt conditioning")
    
    from webbduck.prompt.management import chunk_prompt
    
    device = pipe.device

    # Positive prompt (chunked)
    chunks = chunk_prompt(pipe.tokenizer, prompt)
    chunks_2 = chunk_prompt(pipe.tokenizer_2, prompt_2 or prompt)

    token_embeds = encode_chunks_experimental(
        pipe.tokenizer,
        pipe.text_encoder,
        chunks,
        device,
    )

    token_embeds_2 = encode_chunks_experimental_2(
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        chunks_2,
        device,
    )

    prompt_embeds = torch.cat([token_embeds, token_embeds_2], dim=-1)

    # Pooled prompt (first chunk only)
    pooled_text = truncate_to_tokens(
        pipe.tokenizer_2,
        prompt_2 or prompt,
        max_tokens=77,
    )

    with torch.no_grad():
        pooled_prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                pooled_text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).text_embeds

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(
        prompt_embeds.shape[0], 1
    )

    # Negative prompt (chunked)
    neg_chunks = chunk_prompt(pipe.tokenizer, negative)

    neg_token_embeds = encode_chunks_experimental(
        pipe.tokenizer,
        pipe.text_encoder,
        neg_chunks,
        device,
    )

    neg_token_embeds_2 = encode_chunks_experimental_2(
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        neg_chunks,
        device,
    )

    negative_prompt_embeds = torch.cat(
        [neg_token_embeds, neg_token_embeds_2],
        dim=-1,
    )

    neg_pooled_text = truncate_to_tokens(
        pipe.tokenizer_2,
        negative,
        max_tokens=77,
    )

    with torch.no_grad():
        negative_pooled_prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                neg_pooled_text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).text_embeds

    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
        negative_prompt_embeds.shape[0], 1
    )

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def build_sdxl_refiner_conditioning(pipe, prompt, prompt_2, negative):
    """Build conditioning for true SDXL refiners (CLIP-G only)."""
    device = pipe.device

    # Positive
    text = truncate_to_tokens(
        pipe.tokenizer_2,
        prompt_2 or prompt,
        max_tokens=77,
    )

    with torch.no_grad():
        prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).last_hidden_state

        pooled_prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).text_embeds

    # Negative
    neg_text = truncate_to_tokens(
        pipe.tokenizer_2,
        negative,
        max_tokens=77,
    )

    with torch.no_grad():
        negative_prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                neg_text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).last_hidden_state

        negative_pooled_prompt_embeds = pipe.text_encoder_2(
            pipe.tokenizer_2(
                neg_text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        ).text_embeds

    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def build_sdxl_conditioning_dispatch(*, pipe, prompt, prompt_2, negative, experimental: bool = False):
    """
    Route to standard or experimental conditioning based on flag.
    
    Args:
        experimental: If True, use experimental long-prompt logic
    """
    if experimental:
        return build_sdxl_conditioning_experimental(
            pipe=pipe,
            prompt=prompt,
            prompt_2=prompt_2,
            negative=negative,
        )

    return build_sdxl_conditioning(
        pipe=pipe,
        prompt=prompt,
        prompt_2=prompt_2,
        negative=negative,
    )