"""Standard SDXL prompt conditioning with long-prompt chunk concatenation."""

import torch


def _tokenize_without_special(tokenizer, text: str) -> list[int]:
    raw = tokenizer(
        text or "",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    if not raw:
        return []
    if isinstance(raw[0], list):
        return [int(x) for x in raw[0]]
    return [int(x) for x in raw]


def _chunk_token_ids(tokenizer, token_ids: list[int]) -> torch.Tensor:
    max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
    max_length = max(3, max_length)
    payload = max_length - 2

    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if bos is None:
        bos = eos if eos is not None else 0
    if eos is None:
        eos = bos if bos is not None else 0
    if pad is None:
        pad = eos

    chunks = []
    if token_ids:
        for i in range(0, len(token_ids), payload):
            chunks.append(token_ids[i:i + payload])
    else:
        chunks.append([])

    padded = []
    for chunk in chunks:
        seq = [int(bos)] + [int(t) for t in chunk] + [int(eos)]
        if len(seq) < max_length:
            seq.extend([int(pad)] * (max_length - len(seq)))
        else:
            seq = seq[:max_length]
        padded.append(seq)

    return torch.tensor(padded, dtype=torch.long)


def _extract_seq(out) -> torch.Tensor:
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) >= 2:
        return hidden_states[-2]
    return out.last_hidden_state


def _extract_pooled(out, seq: torch.Tensor) -> torch.Tensor:
    text_embeds = getattr(out, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    return seq[:, 0, :]


def _pad_seq_to_length(seq: torch.Tensor, target_length: int) -> torch.Tensor:
    current = int(seq.shape[1])
    if current >= target_length:
        return seq
    pad = torch.zeros(
        seq.shape[0],
        target_length - current,
        seq.shape[2],
        dtype=seq.dtype,
        device=seq.device,
    )
    return torch.cat([seq, pad], dim=1)


def encode_long_prompt(tokenizer, text_encoder, text: str, device):
    """Encode arbitrarily long text by chunking CLIP token windows and concatenating seq embeddings."""
    token_ids = _tokenize_without_special(tokenizer, text)
    chunks = _chunk_token_ids(tokenizer, token_ids).to(device)

    seq_parts = []
    pooled_last = None
    with torch.no_grad():
        for i in range(chunks.shape[0]):
            out = text_encoder(
                chunks[i:i + 1],
                return_dict=True,
                output_hidden_states=True,
            )
            seq = _extract_seq(out)
            seq_parts.append(seq)
            pooled_last = _extract_pooled(out, seq)

    seq_concat = torch.cat(seq_parts, dim=1)
    return seq_concat, pooled_last


def build_sdxl_conditioning(pipe, prompt, prompt_2, negative):
    """Build SDXL conditioning embeddings with long-prompt chunking for both encoders."""
    device = pipe.device
    pos_text_2 = prompt_2 or prompt

    pos_seq_1, _ = encode_long_prompt(pipe.tokenizer, pipe.text_encoder, prompt or "", device)
    pos_seq_2, pos_pool = encode_long_prompt(pipe.tokenizer_2, pipe.text_encoder_2, pos_text_2 or "", device)

    neg_seq_1, _ = encode_long_prompt(pipe.tokenizer, pipe.text_encoder, negative or "", device)
    neg_seq_2, neg_pool = encode_long_prompt(pipe.tokenizer_2, pipe.text_encoder_2, negative or "", device)

    pos_pair_len = max(int(pos_seq_1.shape[1]), int(pos_seq_2.shape[1]))
    neg_pair_len = max(int(neg_seq_1.shape[1]), int(neg_seq_2.shape[1]))
    common_len = max(pos_pair_len, neg_pair_len)

    pos_seq_1 = _pad_seq_to_length(pos_seq_1, common_len)
    pos_seq_2 = _pad_seq_to_length(pos_seq_2, common_len)
    neg_seq_1 = _pad_seq_to_length(neg_seq_1, common_len)
    neg_seq_2 = _pad_seq_to_length(neg_seq_2, common_len)

    prompt_embeds = torch.cat([pos_seq_1, pos_seq_2], dim=-1)
    negative_prompt_embeds = torch.cat([neg_seq_1, neg_seq_2], dim=-1)

    return (
        prompt_embeds,
        pos_pool,
        negative_prompt_embeds,
        neg_pool,
    )
