"""Tests for SDXL embedding loading helpers."""

from pathlib import Path

import torch

from core import pipeline as pipeline_mod


class _FakePipe:
    def __init__(self):
        self.tokenizer = object()
        self.text_encoder = object()
        self.tokenizer_2 = object()
        self.text_encoder_2 = object()
        self.calls = []

    def load_textual_inversion(self, payload, **kwargs):
        self.calls.append((payload, kwargs))


def test_dual_clip_embedding_loads_both_encoders(monkeypatch):
    def fake_load_file(_path, device="cpu"):
        return {
            "clip_l": torch.zeros((4, 768), dtype=torch.float32),
            "clip_g": torch.zeros((4, 1280), dtype=torch.float32),
        }

    monkeypatch.setattr(pipeline_mod, "load_file", fake_load_file)

    pipe = _FakePipe()
    ok = pipeline_mod._load_sdxl_dual_clip_embedding(
        pipe,
        Path("/tmp/example.safetensors"),
        "emb_token",
    )

    assert ok is True
    assert len(pipe.calls) == 2

    first_payload, first_kwargs = pipe.calls[0]
    second_payload, second_kwargs = pipe.calls[1]

    assert list(first_payload.keys()) == ["emb_token"]
    assert list(second_payload.keys()) == ["emb_token"]
    assert first_kwargs["tokenizer"] is pipe.tokenizer
    assert first_kwargs["text_encoder"] is pipe.text_encoder
    assert second_kwargs["tokenizer"] is pipe.tokenizer_2
    assert second_kwargs["text_encoder"] is pipe.text_encoder_2


def test_dual_clip_embedding_loads_pt_export(monkeypatch):
    def fake_torch_load(_path, map_location="cpu"):
        return {
            "clip_l": torch.zeros((2, 768), dtype=torch.float32),
            "clip_g": torch.zeros((2, 1280), dtype=torch.float32),
        }

    monkeypatch.setattr(pipeline_mod.torch, "load", fake_torch_load)

    pipe = _FakePipe()
    ok = pipeline_mod._load_sdxl_dual_clip_embedding(
        pipe,
        Path("/tmp/example.pt"),
        "emb_token",
    )

    assert ok is True
    assert len(pipe.calls) == 2


def test_dual_clip_embedding_skips_unsupported_extension():
    pipe = _FakePipe()
    ok = pipeline_mod._load_sdxl_dual_clip_embedding(
        pipe,
        Path("/tmp/example.ckpt"),
        "emb_token",
    )

    assert ok is False
    assert pipe.calls == []
