"""Tests for runtime compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from webbduck.core.runtime import resolve_runtime_profile, runtime_error_hint
from webbduck.models import registry as model_registry


def test_forced_cpu_overrides_dtype_to_float32(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_DEVICE", "cpu")
    monkeypatch.setenv("WEBBDUCK_DTYPE", "float16")

    profile = resolve_runtime_profile()

    assert profile.device == "cpu"
    assert profile.dtype == torch.float32
    assert profile.dtype_name == "float32"
    assert any("forcing float32" in note for note in profile.notes)



def test_runtime_error_hint_for_kernel_mismatch():
    exc = RuntimeError("CUDA error: no kernel image is available for execution on the device")
    hint = runtime_error_hint(exc)
    assert hint is not None
    assert "kernel compatibility" in hint.lower()



def test_runtime_error_hint_for_non_kernel_error():
    exc = RuntimeError("unrelated error")
    assert runtime_error_hint(exc) is None


def test_model_registry_root_defaults_to_repo_root():
    expected_root = Path(__file__).resolve().parent.parent
    assert model_registry.ROOT == expected_root


def test_merge_model_registry_creates_parent_directory(monkeypatch, tmp_path):
    models_file = tmp_path / "nested" / "checkpoint" / "sdxl" / "models.json"
    monkeypatch.setattr(model_registry, "MODELS_FILE", models_file)

    registry = model_registry.merge_model_registry(
        {"demo-model": {"path": "demo-path", "source": "local"}},
        {},
        persist=True,
    )

    assert "demo-model" in registry
    assert models_file.exists()
    saved = json.loads(models_file.read_text())
    assert saved == {"demo-model": {"defaults": {}}}
