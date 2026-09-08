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


# --------------------------------------------------------------------------------------
# FP8 + identity-edit LoRA residual (Phase 4)
# --------------------------------------------------------------------------------------

def _fp8_linear_worker_module():
    from core.backends import krea2_worker as worker

    return worker


def test_linear_lora_coefficient():
    from core.backends.krea2_worker import _linear_lora_coefficient

    assert _linear_lora_coefficient(64, None, 1.0) == 1.0
    assert _linear_lora_coefficient(64, 32, 1.0) == 0.5
    assert _linear_lora_coefficient(64, 32, 2.0) == 1.0
    assert _linear_lora_coefficient(128, None, 0.0) == 0.0


def test_scaled_fp8_linear_keeps_qweight_storage_under_lora():
    worker = _fp8_linear_worker_module()
    qweight = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float32)
    scale = torch.tensor(0.5, dtype=torch.float32)
    linear = worker.ScaledFP8Linear(2, 2, qweight=qweight, scale=scale)
    qweight_before = linear.qweight.clone()

    linear.install_lora(
        torch.randn(4, 2),
        torch.randn(2, 4),
        alpha=torch.tensor(2.0),
        lora_scale=1.0,
    )

    assert linear.lora_rank == 4
    assert torch.equal(linear.qweight, qweight_before)
    assert tuple(linear.lora_a.shape) == (4, 2)
    assert tuple(linear.lora_b.shape) == (2, 4)


def test_scaled_fp8_linear_applies_lora_residual_on_top_of_base():
    worker = _fp8_linear_worker_module()
    torch.manual_seed(0)
    in_features, out_features, rank = 4, 3, 2
    qweight = torch.randn(out_features, in_features)
    scale = torch.tensor(0.5)
    x = torch.randn(5, in_features)

    linear = worker.ScaledFP8Linear(
        in_features, out_features, qweight=qweight, scale=scale
    )
    base = F.linear(x, qweight * scale)

    assert torch.allclose(linear(x), base)

    a = torch.randn(rank, in_features)
    b = torch.randn(out_features, rank)
    coef = 0.25  # alpha=rank? use alpha=1, lora_scale=0.5 -> (1/2)*0.5 = 0.25
    linear.install_lora(a, b, alpha=torch.tensor(1.0), lora_scale=0.5)

    expected = base + coef * F.linear(F.linear(x, a), b)
    assert torch.allclose(linear(x), expected, atol=1e-5)

    # alpha=rank + scale 1.0 => coefficient 1.0 exactly.
    linear2 = worker.ScaledFP8Linear(
        in_features, out_features, qweight=qweight, scale=scale
    )
    linear2.install_lora(a, b, alpha=None, lora_scale=1.0)
    assert torch.allclose(
        linear2(x), base + F.linear(F.linear(x, a), b), atol=1e-5
    )


def test_apply_identity_lora_residuals_handles_dense_and_fp8():
    worker = _fp8_linear_worker_module()
    import torch.nn as nn

    torch.manual_seed(1)
    in_f, out_f, rank = 4, 3, 2

    dense = nn.Linear(in_f, out_f)
    fp8 = worker.ScaledFP8Linear(
        in_f, out_f, qweight=torch.randn(out_f, in_f) * 0.5, scale=torch.tensor(0.5)
    )

    class TwoBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_a = dense
            self.proj_b = fp8

    root = TwoBlock()
    dense_weight_before = dense.weight.clone()

    a = torch.randn(rank, in_f)
    b = torch.randn(out_f, rank)
    converted = {
        "proj_a.lora_A.weight": a,
        "proj_a.lora_B.weight": b,
        "proj_a.lora_alpha": torch.tensor(rank),
        "proj_b.lora_A.weight": a,
        "proj_b.lora_B.weight": b,
        "proj_b.lora_alpha": torch.tensor(rank),
    }

    info = worker.apply_identity_lora_residuals(root, converted_state=converted)
    assert info["applied_modules"] == 2

    # Dense module got the delta fused into its weight.
    assert not torch.equal(dense.weight, dense_weight_before)
    assert torch.allclose(
        dense.weight, dense_weight_before + torch.mm(b, a) * 1.0, atol=1e-6
    )
    # FP8 module keeps its qweight storage and applies the residual at forward.
    assert fp8.lora_rank == rank
    x = torch.randn(3, in_f)
    expected_dense = F.linear(x, dense.weight, dense.bias)
    expected_fp8 = F.linear(x, fp8.qweight * fp8.scale.reshape(-1, 1)) + F.linear(
        F.linear(x, a), b
    )
    assert torch.allclose(root.proj_a(x), expected_dense, atol=1e-6)
    assert torch.allclose(root.proj_b(x), expected_fp8, atol=1e-6)


def test_lora_scale_zero_disables_adapter():
    worker = _fp8_linear_worker_module()
    torch.manual_seed(2)
    in_f, out_f, rank = 4, 3, 2
    qweight = torch.randn(out_f, in_f)
    x = torch.randn(5, in_f)
    linear = worker.ScaledFP8Linear(
        in_f, out_f, qweight=qweight, scale=torch.tensor(0.5)
    )
    base = linear(x)
    a = torch.randn(rank, in_f)
    b = torch.randn(out_f, rank)
    linear.install_lora(a, b, alpha=None, lora_scale=0.0)
    assert linear.lora_scale == 0.0
    assert torch.equal(linear(x), base)


def test_apply_identity_lora_residuals_rejects_unknown_module():
    worker = _fp8_linear_worker_module()
    import torch.nn as nn

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.ReLU()

    root = Root()
    with __import__("pytest").raises(Exception, match="non-linear|unknown|shape"):
        worker.apply_identity_lora_residuals(
            root,
            converted_state={
                "block.lora_A.weight": torch.randn(2, 4),
                "block.lora_B.weight": torch.randn(4, 2),
            },
        )


def test_native_forward_keeps_lora_residual_path():
    source = inspect.getsource(native._native_forward)
    assert "_linear_lora_coefficient" in source
    assert "lora_a" in source
    assert "lora_rank" in source
