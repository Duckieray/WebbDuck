"""Tests for SDXL long-prompt conditioning."""

from types import SimpleNamespace

import torch

from webbduck.prompt.conditioning import build_sdxl_conditioning, _chunk_token_ids


class _FakeTokenizer:
    def __init__(self, model_max_length=77, bos=101, eos=102, pad=0, offset=0):
        self.model_max_length = model_max_length
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad
        self._offset = int(offset)

    def _to_ids(self, text: str):
        ids = []
        for part in (text or "").split():
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                ids.append(self._offset + int(digits) + 1)
            else:
                ids.append(self._offset + 1)
        return ids

    def __call__(
        self,
        text,
        truncation=False,
        add_special_tokens=True,
        return_tensors=None,
        padding=False,
        max_length=None,
        **_kwargs,
    ):
        ids = self._to_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation and max_length is not None:
            ids = ids[: int(max_length)]

        if padding == "max_length" and max_length is not None:
            if len(ids) < int(max_length):
                ids = ids + [self.pad_token_id] * (int(max_length) - len(ids))

        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        return {"input_ids": ids}


class _FakeEncoder:
    def __init__(self, hidden_size: int, with_text_embeds: bool):
        self.hidden_size = hidden_size
        self.with_text_embeds = with_text_embeds

    def __call__(self, input_ids, return_dict=True, output_hidden_states=True):
        seq = input_ids.float().unsqueeze(-1).repeat(1, 1, self.hidden_size)
        out = {
            "hidden_states": [torch.zeros_like(seq), seq],
            "last_hidden_state": seq,
        }
        if self.with_text_embeds:
            out["text_embeds"] = input_ids.float().sum(dim=1, keepdim=True).repeat(1, self.hidden_size)
        return SimpleNamespace(**out)


class _FakePipe:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = _FakeTokenizer(offset=0)
        self.tokenizer_2 = _FakeTokenizer(offset=1000)
        self.text_encoder = _FakeEncoder(hidden_size=4, with_text_embeds=False)
        self.text_encoder_2 = _FakeEncoder(hidden_size=6, with_text_embeds=True)


def test_long_prompt_conditioning_expands_sequence_length():
    pipe = _FakePipe()
    prompt = " ".join(f"p{i}" for i in range(180))
    negative = "low quality"

    prompt_embeds, pooled, neg_embeds, neg_pooled = build_sdxl_conditioning(
        pipe=pipe,
        prompt=prompt,
        prompt_2="",
        negative=negative,
    )

    assert prompt_embeds.shape[0] == 1
    assert prompt_embeds.shape[-1] == 10  # 4 + 6
    assert prompt_embeds.shape[1] > 77
    assert neg_embeds.shape[1] == prompt_embeds.shape[1]
    assert pooled.shape == (1, 6)
    assert neg_pooled.shape == (1, 6)


def test_pooled_embedding_uses_last_chunk_of_encoder2():
    pipe = _FakePipe()
    prompt = " ".join(f"p{i}" for i in range(165))
    negative = " ".join(f"n{i}" for i in range(95))

    _, pooled, _, neg_pooled = build_sdxl_conditioning(
        pipe=pipe,
        prompt=prompt,
        prompt_2="",
        negative=negative,
    )

    prompt_ids = pipe.tokenizer_2(prompt, truncation=False, add_special_tokens=False)["input_ids"]
    prompt_chunks = _chunk_token_ids(pipe.tokenizer_2, prompt_ids)
    expected_prompt = float(prompt_chunks[-1].sum().item())

    neg_ids = pipe.tokenizer_2(negative, truncation=False, add_special_tokens=False)["input_ids"]
    neg_chunks = _chunk_token_ids(pipe.tokenizer_2, neg_ids)
    expected_neg = float(neg_chunks[-1].sum().item())

    assert torch.allclose(pooled, torch.full((1, 6), expected_prompt))
    assert torch.allclose(neg_pooled, torch.full((1, 6), expected_neg))
