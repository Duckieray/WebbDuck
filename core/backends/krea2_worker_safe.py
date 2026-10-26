"""Safety/compatibility wrapper for the Krea 2 isolated worker.

The main Krea worker owns checkpoint loading and the model-specific execution
primitives. This wrapper keeps the hardware policy easy to audit while we
validate constrained-GPU behavior on real machines:

* conditioning is explicitly cast to the selected inference dtype before it is
  reused by Diffusers, preventing accidental FP32 latents/attention;
* constrained GPUs reserve additional transient memory before choosing resident
  execution;
* CUDA OOM recovery uses fresh denoiser instances and progressively safer
  profiles so stale offload hooks cannot contaminate a retry.

Once this policy has enough hardware coverage it can be folded back into the
base worker without changing the public generation API.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backends import krea2_worker as base
from runtime_hardware import detect_torch_hardware


def _selected_dtype(hardware: dict[str, Any]) -> torch.dtype:
    device = base._torch_device(hardware)
    if device == "cuda":
        return torch.bfloat16 if bool(hardware.get("bf16")) else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# The safety policy below now lives in the base worker, which is the single
# generation loop. Alias it here rather than keeping a second copy: these helpers
# used to be duplicated, and the duplication is exactly how img2img and the face
# detailer ended up implemented in one loop and skipped by the other.
_coerce_conditioning_dtype = base._coerce_conditioning_dtype
_memory_reserve_gb = base._memory_reserve_gb
_retry_profiles = base._retry_profiles
_dispose_pipe = base._dispose_pipe
_fresh_denoiser = base._fresh_denoiser


def _run(request: dict, output_dir: Path, progress_path: Path | None = None) -> dict:
    """Delegate to the canonical worker loop.

    This module used to *replace* ``base._run`` with a text-to-image-only copy of
    the loop. That silently disabled every feature added to the base worker
    afterwards - Krea 2 img2img and the face/skin detailer never ran in
    production, because this wrapper is the entry the host actually launches.

    The safety policy this module existed to provide (conditioning dtype coercion,
    the desktop-aware activation reserve, post-placement headroom validation and
    the progressive CUDA OOM ladder) has been folded into the base worker, so the
    canonical loop now carries it. Delegating keeps a single generation loop, so
    a feature can no longer be implemented in one and skipped by the other.
    """
    return base._run(request, output_dir, progress_path)



if __name__ == "__main__":
    raise SystemExit(base.main())
