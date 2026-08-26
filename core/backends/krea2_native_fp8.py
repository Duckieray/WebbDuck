"""Native FP8 inference kernel for WebbDuck's scaled Krea linears.

Krea community checkpoints can store a linear weight as FP8 qweight plus a
scalar or per-output-channel dequantization scale. The original WebbDuck
storage wrapper preserved the FP8 checkpoint in memory but converted the whole
weight to BF16/FP16 on *every* ``F.linear`` call. For Krea's large transformer
that repeated dequantization is both a latency and transient-VRAM penalty.

This module keeps the same checkpoint math while using PyTorch's native FP8
scaled GEMM when the installed CUDA runtime proves that operation works:

1. dynamically quantize the current BF16/FP16 activation with one per-tensor
   scale;
2. multiply it directly by the stored FP8 qweight using ``torch._scaled_mm``;
3. apply the checkpoint's scalar/per-output weight scale to the GEMM output;
4. add the original bias.

The post-GEMM output scaling is algebraically equivalent to multiplying every
row of qweight by its stored output-channel scale before the matmul, but avoids
Blackwell's currently problematic per-row ``_scaled_mm`` scaling path.

This is an optional optimization. Unsupported hardware, unsupported scale
shapes, kernel probe failures, or non-FP8 inputs automatically retain the
existing storage-only implementation. CUDA OOMs are re-raised so the hardened
Krea offload ladder can still downgrade the request safely.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_ORIGINAL_FORWARD: Any | None = None
_PATCHED_CLASS: type | None = None
_RUNTIME_ENABLED = False
_PROBE_ERROR: str | None = None
_NATIVE_CALLS = 0
_FALLBACK_CALLS = 0
_NATIVE_MODULES: set[int] = set()
_FALLBACK_MODULES: set[int] = set()


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def _mode() -> str:
    return str(os.getenv("WEBBDUCK_KREA2_NATIVE_FP8", "auto")).strip().lower()


def _hardware_allows_native_fp8(hardware: dict[str, Any]) -> bool:
    mode = _mode()
    if mode in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    accelerator = str(hardware.get("accelerator") or "cpu").lower()
    if accelerator != "cuda" and mode not in {"force", "forced"}:
        return False
    return bool(hardware.get("fp8_storage", True))


def _supported_output_scale(scale: torch.Tensor | None, out_features: int) -> bool:
    if scale is None:
        return True
    return int(scale.numel()) in {1, int(out_features)}


def _apply_output_scale(
    output: torch.Tensor,
    scale: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    if scale is None:
        return output
    scale = scale.to(device=output.device, dtype=output.dtype)
    if int(scale.numel()) == 1:
        return output * scale.reshape(())
    if int(scale.numel()) == int(out_features):
        return output * scale.reshape(1, int(out_features))
    raise RuntimeError(
        f"Krea FP8 output scale has {scale.numel()} values; expected 1 or {out_features}."
    )


def _probe_scaled_mm() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    scaled_mm = getattr(torch, "_scaled_mm", None)
    if not callable(scaled_mm):
        return False, "torch._scaled_mm is unavailable"
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return False, "torch.float8_e4m3fn is unavailable"

    try:
        device = torch.device("cuda")
        a_hp = torch.randn((16, 16), device=device, dtype=torch.bfloat16)
        amax = a_hp.abs().amax().float().clamp_min(torch.finfo(torch.float32).tiny)
        fp8_max = float(torch.finfo(fp8_dtype).max)
        dequant_a = (amax / fp8_max).reshape(1, 1)
        quant_multiplier = (1.0 / dequant_a).to(dtype=a_hp.dtype)
        a_fp8 = torch.clamp(a_hp * quant_multiplier, -fp8_max, fp8_max).to(fp8_dtype)
        b_fp8 = torch.randn((16, 16), device=device, dtype=torch.bfloat16).to(fp8_dtype)
        one = torch.ones((1, 1), device=device, dtype=torch.float32)
        out = scaled_mm(
            a_fp8,
            b_fp8.t(),
            scale_a=dequant_a,
            scale_b=one,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        if tuple(out.shape) != (16, 16) or not bool(torch.isfinite(out).all()):
            return False, "native FP8 probe returned invalid output"
        del a_hp, amax, dequant_a, quant_multiplier, a_fp8, b_fp8, one, out
        torch.cuda.empty_cache()
        return True, None
    except Exception as exc:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False, str(exc or exc.__class__.__name__)


def reset_stats() -> None:
    global _NATIVE_CALLS, _FALLBACK_CALLS
    _NATIVE_CALLS = 0
    _FALLBACK_CALLS = 0
    _NATIVE_MODULES.clear()
    _FALLBACK_MODULES.clear()


def enable_for_hardware(hardware: dict[str, Any], base_module: Any) -> dict[str, Any]:
    """Patch ScaledFP8Linear only when the exact runtime passes an FP8 probe."""
    global _ORIGINAL_FORWARD, _PATCHED_CLASS, _RUNTIME_ENABLED, _PROBE_ERROR

    reset_stats()
    if not _hardware_allows_native_fp8(hardware):
        _RUNTIME_ENABLED = False
        _PROBE_ERROR = "native FP8 disabled by hardware/profile"
        return stats()

    ok, error = _probe_scaled_mm()
    _RUNTIME_ENABLED = bool(ok)
    _PROBE_ERROR = error
    if not ok:
        return stats()

    cls = base_module.ScaledFP8Linear
    if _PATCHED_CLASS is not cls:
        _ORIGINAL_FORWARD = cls.forward
        cls.forward = _native_forward
        _PATCHED_CLASS = cls
    return stats()


def _native_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    global _NATIVE_CALLS, _FALLBACK_CALLS

    original = _ORIGINAL_FORWARD
    if original is None:
        raise RuntimeError("Krea native FP8 forward was installed without an original forward")

    disabled = bool(getattr(self, "_webbduck_native_fp8_disabled", False))
    qweight = getattr(self, "qweight", None)
    scale = getattr(self, "scale", None)
    eligible = (
        _RUNTIME_ENABLED
        and not disabled
        and isinstance(qweight, torch.Tensor)
        and x.is_cuda
        and qweight.device == x.device
        and x.dtype in {torch.bfloat16, torch.float16}
        and str(qweight.dtype).startswith("torch.float8")
        and int(getattr(self, "in_features", 0)) % 16 == 0
        and int(getattr(self, "out_features", 0)) % 16 == 0
        and _supported_output_scale(scale, int(getattr(self, "out_features", 0)))
        and callable(getattr(torch, "_scaled_mm", None))
    )
    if not eligible:
        _FALLBACK_CALLS += 1
        _FALLBACK_MODULES.add(id(self))
        return original(self, x)

    try:
        original_shape = tuple(x.shape)
        x2d = x.reshape(-1, int(self.in_features)).contiguous()
        fp8_dtype = qweight.dtype
        fp8_max = float(torch.finfo(fp8_dtype).max)

        # Keep dynamic activation quantization fully on the accelerator. Host
        # reads such as ``amax.item()`` would synchronize every linear layer and
        # destroy the throughput benefit of the native GEMM path.
        amax = x2d.abs().amax().float()
        amax = torch.nan_to_num(
            amax,
            nan=1.0,
            posinf=fp8_max,
            neginf=fp8_max,
        ).clamp_min(torch.finfo(torch.float32).tiny)
        dequant_a = (amax / fp8_max).reshape(1, 1)
        quant_multiplier = (1.0 / dequant_a).to(dtype=x.dtype)
        x_fp8 = torch.clamp(x2d * quant_multiplier, -fp8_max, fp8_max).to(fp8_dtype)
        scale_b = torch.ones((1, 1), device=x.device, dtype=torch.float32)
        output = torch._scaled_mm(
            x_fp8,
            qweight.t(),
            scale_a=dequant_a,
            scale_b=scale_b,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        del dequant_a, quant_multiplier, x_fp8, scale_b

        output = _apply_output_scale(output, scale, int(self.out_features))
        bias = getattr(self, "bias", None)
        if bias is not None:
            output = output + bias.to(device=x.device, dtype=x.dtype).reshape(1, -1)

        _NATIVE_CALLS += 1
        _NATIVE_MODULES.add(id(self))
        return output.reshape(*original_shape[:-1], int(self.out_features))
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise
        # Kernel/layout incompatibilities should be paid only once per module.
        # Keep correctness by returning to the existing dequantize+F.linear path.
        setattr(self, "_webbduck_native_fp8_disabled", True)
        _FALLBACK_CALLS += 1
        _FALLBACK_MODULES.add(id(self))
        return original(self, x)


def stats() -> dict[str, Any]:
    return {
        "enabled": bool(_RUNTIME_ENABLED),
        "mode": _mode(),
        "probe_error": _PROBE_ERROR,
        "native_calls": int(_NATIVE_CALLS),
        "fallback_calls": int(_FALLBACK_CALLS),
        "native_modules": len(_NATIVE_MODULES),
        "fallback_modules": len(_FALLBACK_MODULES),
        "kernel": "torch._scaled_mm-per-tensor-activation" if _RUNTIME_ENABLED else None,
    }
