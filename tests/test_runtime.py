"""Tests for runtime compatibility helpers."""

from __future__ import annotations

import torch

from core.runtime import resolve_runtime_profile, runtime_error_hint


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
