from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

from core.backends import krea2_native_fp8 as native


def test_post_gemm_channel_scale_matches_weight_dequantization_math():
    x = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.5, -1.0]], dtype=torch.float32)
    qweight = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 0.5, 1.5]],
        dtype=torch.float32,
    )
    scale = torch.tensor([0.25, 1.75], dtype=torch.float32)

    expected = F.linear(x, qweight * scale.reshape(-1, 1))
    factored = native._apply_output_scale(F.linear(x, qweight), scale, 2)

    assert torch.allclose(factored, expected)


def test_post_gemm_scalar_scale_matches_weight_dequantization_math():
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    qweight = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float32)
    scale = torch.tensor(0.5, dtype=torch.float32)

    expected = F.linear(x, qweight * scale)
    factored = native._apply_output_scale(F.linear(x, qweight), scale, 2)

    assert torch.allclose(factored, expected)


def test_native_kernel_accepts_only_scalar_or_output_channel_scale():
    assert native._supported_output_scale(None, 8)
    assert native._supported_output_scale(torch.ones(()), 8)
    assert native._supported_output_scale(torch.ones(8), 8)
    assert native._supported_output_scale(torch.ones(8, 1), 8)
    assert not native._supported_output_scale(torch.ones(4), 8)
    assert not native._supported_output_scale(torch.ones(8, 2), 8)


def test_native_fp8_auto_mode_is_cuda_only(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_NATIVE_FP8", raising=False)
    assert native._hardware_allows_native_fp8({"accelerator": "cuda", "fp8_storage": True})
    assert not native._hardware_allows_native_fp8({"accelerator": "rocm", "fp8_storage": True})
    assert not native._hardware_allows_native_fp8({"accelerator": "cpu", "fp8_storage": True})
    assert not native._hardware_allows_native_fp8({"accelerator": "cuda", "fp8_storage": False})


def test_native_fp8_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_NATIVE_FP8", "off")
    assert not native._hardware_allows_native_fp8({"accelerator": "cuda", "fp8_storage": True})


def test_native_forward_has_no_per_layer_host_sync():
    source = inspect.getsource(native._native_forward)
    assert ".item(" not in source
    assert "torch._scaled_mm" in source
    assert "use_fast_accum=True" in source


def test_adaptive_worker_enables_allocator_and_native_kernel_before_generation():
    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "backends"
        / "krea2_worker_adaptive.py"
    ).read_text(encoding="utf-8")

    allocator = 'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")'
    assert allocator in source
    assert source.index(allocator) < source.index("import torch")
    assert "native_fp8.enable_for_hardware(hardware, safe.base)" in source
    assert 'runtime["native_fp8"] = native_fp8.stats()' in source
