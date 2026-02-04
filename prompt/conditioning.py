"""Standard SDXL prompt conditioning."""

import torch
from webbduck.prompt.management import chunk_prompt, truncate_to_tokens

MAX_TOKENS = 77


def encode_chunks_text_encoder(tokenizer, text_encoder, chunks, device):
    """Returns CLIP-L token embeddings: [B, 77, 768]"""
    embeds = []

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

        embeds.append(out.hidden_states[-2])

    tokens = torch.cat(embeds, dim=1)

    if tokens.shape[1] < 77:
        pad = torch.zeros(
            tokens.shape[0],
            77 - tokens.shape[1],
            tokens.shape[2],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = torch.cat([tokens, pad], dim=1)

    return tokens[:, :77, :]


def encode_chunks_text_encoder_2(tokenizer, text_encoder_2, chunks, device):
    """Returns CLIP-G token embeddings: [B, 77, 1280]"""
    embeds = []

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
                output_hidden_states=True,
                return_dict=True,
            )

        embeds.append(out.hidden_states[-2])

    tokens = torch.cat(embeds, dim=1)

    if tokens.shape[1] < 77:
        pad = torch.zeros(
            tokens.shape[0],
            77 - tokens.shape[1],
            tokens.shape[2],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = torch.cat([tokens, pad], dim=1)

    return tokens[:, :77, :]


def build_sdxl_conditioning(pipe, prompt, prompt_2, negative):
    """Build standard SDXL conditioning embeddings."""
    device = pipe.device

    # Positive prompt
    chunks = chunk_prompt(pipe.tokenizer, prompt)
    chunks_2 = chunk_prompt(pipe.tokenizer, prompt_2 or prompt)

    token_embeds = encode_chunks_text_encoder(
        pipe.tokenizer,
        pipe.text_encoder,
        chunks,
        device,
    )

    token_embeds_2 = encode_chunks_text_encoder_2(
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        chunks_2,
        device,
    )

    prompt_embeds = torch.cat([token_embeds, token_embeds_2], dim=-1)

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

    # Negative prompt
    neg_chunks = chunk_prompt(pipe.tokenizer, negative)

    neg_token_embeds = encode_chunks_text_encoder(
        pipe.tokenizer,
        pipe.text_encoder,
        neg_chunks,
        device,
    )

    neg_token_embeds_2 = encode_chunks_text_encoder_2(
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