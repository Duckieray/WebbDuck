"""Tests for best-effort LoRA trigger auto-detection.

Covers :func:`models.registry.detect_lora_trigger` extracting a trained
trigger word from safetensors ``__metadata__`` headers.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

from models.registry import detect_lora_trigger


def _write_lora(tmp_path: Path, name: str, metadata: dict) -> Path:
    path = tmp_path / name
    save_file({"down.weight": torch.zeros(4, 4)}, str(path), metadata=metadata)
    return path


def test_trigger_from_tag_frequency(tmp_path):
    lora = _write_lora(
        tmp_path,
        "test.safetensors",
        {"ss_tag_frequency": '{"1_CLASS": {"MASTURBATE": 1}}'},
    )
    assert detect_lora_trigger(lora) == "MASTURBATE"


def test_trigger_from_multiple_tag_counts_picks_highest(tmp_path):
    lora = _write_lora(
        tmp_path,
        "test.safetensors",
        {"ss_tag_frequency": '{"1_X": {"rare": 1, "common": 20}}'},
    )
    assert detect_lora_trigger(lora) == "common"


def test_trigger_from_explicit_field_wins(tmp_path):
    lora = _write_lora(
        tmp_path,
        "test.safetensors",
        {
            "ss_trigger_word": "explicit-trigger",
            "ss_tag_frequency": '{"1_X": {"other": 1}}',
        },
    )
    assert detect_lora_trigger(lora) == "explicit-trigger"


def test_trigger_from_notes(tmp_path):
    lora = _write_lora(
        tmp_path,
        "test.safetensors",
        {"notes": "Example Prompt: Trigger word: g0th1c, then some rest"},
    )
    assert detect_lora_trigger(lora) == "g0th1c"


def test_no_trigger_returns_none(tmp_path):
    lora = _write_lora(
        tmp_path,
        "test.safetensors",
        {"ss_tag_frequency": '{"1_": {"": 1}}'},
    )
    assert detect_lora_trigger(lora) is None


def test_no_metadata_returns_none(tmp_path):
    lora = _write_lora(tmp_path, "test.safetensors", {})
    assert detect_lora_trigger(lora) is None
