"""Stdlib-first hardware profiling shared by runtime setup and isolated workers.

The public WebbDuck generation contract is model/checkpoint driven. Hardware is
an internal execution concern: runtime preparation and backends use this module
to choose compatible acceleration stacks and memory strategies automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


@dataclass(frozen=True)
class HostHardwareProfile:
    accelerator: str
    vendor: str
    name: str
    total_vram_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    source: str = "fallback"

    @property
    def torch_index(self) -> str | None:
        """Best PyTorch 2.12 wheel channel for the detected accelerator."""
        if self.accelerator == "cuda":
            # PyTorch 2.12 recommends CUDA 12.6 for Pascal/Volta and CUDA 13+
            # for newer NVIDIA architectures. CUDA 13.0 is the stable default.
            if self.compute_capability and self.compute_capability < (7, 5):
                return "cu126"
            return "cu130"
        if self.accelerator == "rocm":
            return "rocm7.2"
        if self.accelerator == "cpu":
            return "cpu"
        # macOS/MPS wheels are installed from PyPI without a platform index.
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.compute_capability is not None:
            payload["compute_capability"] = list(self.compute_capability)
        payload["torch_index"] = self.torch_index
        return payload


def _run_capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _parse_compute_capability(raw: str) -> tuple[int, int] | None:
    try:
        major, minor = str(raw).strip().split(".", 1)
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _detect_nvidia() -> HostHardwareProfile | None:
    raw = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return None
    first = raw.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        total_vram_gb = float(parts[1]) / 1024.0
    except ValueError:
        total_vram_gb = 0.0
    capability = _parse_compute_capability(parts[2]) if len(parts) >= 3 else None
    return HostHardwareProfile(
        accelerator="cuda",
        vendor="nvidia",
        name=parts[0] or "NVIDIA GPU",
        total_vram_gb=total_vram_gb,
        compute_capability=capability,
        source="nvidia-smi",
    )


def _detect_rocm() -> HostHardwareProfile | None:
    raw = _run_capture(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if raw:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = {}
        if isinstance(payload, dict) and payload:
            card = next((value for value in payload.values() if isinstance(value, dict)), {})
            name = "AMD GPU"
            total_bytes = 0.0
            for key, value in card.items():
                lowered = str(key).lower()
                if "card series" in lowered or "card model" in lowered or "product" in lowered:
                    if value:
                        name = str(value)
                if "vram total" in lowered:
                    try:
                        total_bytes = float(value)
                    except (TypeError, ValueError):
                        pass
            return HostHardwareProfile(
                accelerator="rocm",
                vendor="amd",
                name=name,
                total_vram_gb=total_bytes / 1024**3 if total_bytes else 0.0,
                source="rocm-smi",
            )

    # A minimal sysfs fallback still identifies AMD hardware even when ROCm
    # tooling is not on PATH. Runtime readiness will later report if ROCm itself
    # is unavailable.
    for vendor_path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor == "0x1002":
            return HostHardwareProfile(
                accelerator="rocm",
                vendor="amd",
                name="AMD GPU",
                source="sysfs",
            )
    return None


def detect_host_hardware(force: str | None = None) -> HostHardwareProfile:
    """Detect the host accelerator without importing PyTorch."""
    requested = str(force or os.getenv("WEBBDUCK_ACCELERATOR") or "auto").strip().lower()
    if requested in {"nvidia", "cuda"}:
        return _detect_nvidia() or HostHardwareProfile("cuda", "nvidia", "NVIDIA GPU", source="forced")
    if requested in {"amd", "rocm"}:
        return _detect_rocm() or HostHardwareProfile("rocm", "amd", "AMD GPU", source="forced")
    if requested == "cpu":
        return HostHardwareProfile("cpu", "cpu", "CPU", source="forced")
    if requested in {"mps", "apple"}:
        return HostHardwareProfile("mps", "apple", "Apple GPU", source="forced")

    if platform.system() == "Darwin":
        return HostHardwareProfile("mps", "apple", "Apple GPU", source="platform")
    nvidia = _detect_nvidia()
    if nvidia is not None:
        return nvidia
    rocm = _detect_rocm()
    if rocm is not None:
        return rocm
    return HostHardwareProfile("cpu", "cpu", "CPU", source="fallback")


def memory_tier(total_vram_gb: float) -> str:
    value = max(0.0, float(total_vram_gb))
    if value <= 0:
        return "cpu"
    if value < 8:
        return "tiny"
    if value < 12:
        return "low"
    if value < 18:
        return "constrained"
    if value < 26:
        return "mid"
    if value < 40:
        return "high"
    return "ultra"


def detect_torch_hardware(torch_module: Any | None = None) -> dict[str, Any]:
    """Return live accelerator capabilities from an already-installed PyTorch."""
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    torch = torch_module
    profile: dict[str, Any] = {
        "accelerator": "cpu",
        "vendor": "cpu",
        "name": "CPU",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "compute_capability": None,
        "bf16": False,
        "fp8_storage": False,
        "stream_prefetch": False,
        "memory_tier": "cpu",
    }

    try:
        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        mps_available = False

    if bool(torch.cuda.is_available()):
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        profile["accelerator"] = "rocm" if is_rocm else "cuda"
        profile["vendor"] = "amd" if is_rocm else "nvidia"
        try:
            profile["name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            profile["name"] = "AMD GPU" if is_rocm else "NVIDIA GPU"
        try:
            free, total = torch.cuda.mem_get_info(0)
            profile["free_vram_gb"] = float(free) / 1024**3
            profile["total_vram_gb"] = float(total) / 1024**3
        except Exception:
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                profile["total_vram_gb"] = float(total) / 1024**3
            except Exception:
                pass
        if not is_rocm:
            try:
                capability = torch.cuda.get_device_capability(0)
                profile["compute_capability"] = [int(capability[0]), int(capability[1])]
            except Exception:
                pass
        try:
            profile["bf16"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            profile["bf16"] = False
        try:
            probe = torch.empty(1, device="cuda", dtype=torch.float8_e4m3fn)
            del probe
            profile["fp8_storage"] = True
        except Exception:
            profile["fp8_storage"] = False
        # Diffusers' CUDA-stream prefetch path is proven on CUDA. ROCm uses the
        # torch.cuda compatibility API too, but stays on the conservative
        # synchronous path until a ROCm hardware smoke test validates streams.
        profile["stream_prefetch"] = not is_rocm
    elif mps_available:
        profile.update(
            {
                "accelerator": "mps",
                "vendor": "apple",
                "name": "Apple GPU",
            }
        )

    profile["memory_tier"] = memory_tier(float(profile["total_vram_gb"]))
    return profile
