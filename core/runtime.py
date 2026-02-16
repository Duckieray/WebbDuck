"""Runtime device/dtype selection and GPU compatibility probing."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch


_KERNEL_COMPAT_MARKERS = (
    "no kernel image is available for execution on the device",
    "device kernel image is invalid",
    "cuda error",
)


@dataclass
class RuntimeProfile:
    device: str
    dtype: torch.dtype
    dtype_name: str
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    cuda_capability: Optional[tuple[int, int]] = None
    total_vram_gb: Optional[float] = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dtype"] = self.dtype_name
        return data


def _is_true(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_kernel_compat_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _KERNEL_COMPAT_MARKERS)


def _probe_cuda_kernel_support() -> tuple[bool, Optional[str]]:
    """Run tiny CUDA kernels to verify runtime compatibility.

    Includes an embedding op since some incompatible builds fail there first.
    """
    try:
        x = torch.tensor([1.0], device="cuda")
        _ = (x + 1.0).item()

        emb = torch.nn.Embedding(16, 8, device="cuda")
        idx = torch.tensor([1, 3, 5], device="cuda", dtype=torch.long)
        _ = emb(idx)

        torch.cuda.synchronize()
        return True, None
    except Exception as exc:
        return False, str(exc)


def resolve_runtime_profile(logger=None) -> RuntimeProfile:
    """Resolve runtime device/dtype with compatibility checks and safe fallbacks."""
    notes: list[str] = []

    forced_device = (os.getenv("WEBBDUCK_DEVICE", "") or "").strip().lower()
    forced_dtype = (os.getenv("WEBBDUCK_DTYPE", "") or "").strip().lower()
    strict_device = _is_true(os.getenv("WEBBDUCK_STRICT_DEVICE"))

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    if forced_device in {"cpu", "cuda"}:
        device = forced_device

    if device == "cuda" and not cuda_available:
        msg = "WEBBDUCK_DEVICE=cuda was requested but CUDA is not available"
        if strict_device:
            raise RuntimeError(msg)
        notes.append(f"{msg}; falling back to CPU")
        device = "cpu"

    cuda_name = None
    cuda_cap = None
    total_vram_gb = None

    if cuda_available:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = None
        try:
            major, minor = torch.cuda.get_device_capability(0)
            cuda_cap = (int(major), int(minor))
        except Exception:
            cuda_cap = None
        try:
            total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        except Exception:
            total_vram_gb = None

    if device == "cuda":
        ok, err = _probe_cuda_kernel_support()
        if not ok:
            msg = f"CUDA runtime kernel probe failed: {err}"
            if strict_device:
                raise RuntimeError(msg)
            notes.append(f"{msg}; falling back to CPU")
            device = "cpu"

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    bf16_supported = False
    if cuda_available:
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False

    if forced_dtype in dtype_map:
        dtype = dtype_map[forced_dtype]
    elif device == "cuda":
        dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        dtype = torch.float32

    if device == "cuda" and dtype == torch.bfloat16 and not bf16_supported:
        notes.append("bfloat16 requested but not supported on this GPU; using float16")
        dtype = torch.float16

    if device == "cpu" and dtype != torch.float32:
        notes.append("CPU mode selected; forcing float32 for compatibility")
        dtype = torch.float32

    profile = RuntimeProfile(
        device=device,
        dtype=dtype,
        dtype_name=str(dtype).replace("torch.", ""),
        cuda_available=cuda_available,
        cuda_device_name=cuda_name,
        cuda_capability=cuda_cap,
        total_vram_gb=total_vram_gb,
        notes=tuple(notes),
    )

    if logger is not None:
        logger.info(
            "Runtime profile: device=%s dtype=%s cuda_available=%s gpu=%s cc=%s vram_gb=%s",
            profile.device,
            profile.dtype_name,
            profile.cuda_available,
            profile.cuda_device_name,
            profile.cuda_capability,
            profile.total_vram_gb,
        )
        for note in profile.notes:
            logger.warning("Runtime note: %s", note)

    return profile


def runtime_error_hint(exc: Exception) -> Optional[str]:
    """Return user-facing hint when an exception indicates CUDA kernel mismatch."""
    if not _is_kernel_compat_error(exc):
        return None
    return (
        "Detected CUDA kernel compatibility failure. "
        "Install a PyTorch build compatible with your GPU architecture, "
        "or set WEBBDUCK_DEVICE=cpu as a fallback."
    )
