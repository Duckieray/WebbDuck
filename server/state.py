"""Runtime state tracking."""

import torch
import time

state = {
    "stage": "Idle",
    "progress": 0.0,
    "vram": {
        "used": 0.0,
        "total": 0.0,
    },
    "step": 0,
    "total_steps": 0,
    "last_update": time.time(),
    "error_detail": None,
}


def update_stage(stage: str):
    """Update current processing stage.

    Clears ``error_detail`` only when transitioning from a terminal/error
    stage to an active-processing stage (e.g. when a new job starts).
    """
    state["stage"] = stage
    state["last_update"] = time.time()


def update_progress(p: float, step: int = 0, total_steps: int = 0):
    """Update progress (0.0 to 1.0) and step counts."""
    state["progress"] = float(p)
    state["step"] = step
    state["total_steps"] = total_steps
    state["last_update"] = time.time()


def update_error_detail(detail: str | None):
    """Set or clear the error_detail field."""
    state["error_detail"] = detail
    state["last_update"] = time.time()


def update_vram():
    """Update device-wide VRAM usage stats.

    Generation backends such as Krea run in isolated Python processes. PyTorch's
    ``memory_allocated()`` only sees allocations owned by the current process,
    which made the WebbDuck UI report 0.0 GB while an isolated worker was using
    the GPU. ``mem_get_info()`` reports global free/total device memory, so the
    derived used value includes isolated backend processes too.
    """
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        state["vram"] = {
            "used": max(0.0, float(total_bytes - free_bytes) / 1024**3),
            "total": float(total_bytes) / 1024**3,
        }
        state["last_update"] = time.time()
        return
    except Exception:
        pass

    # Keep a local-allocator fallback for CUDA builds where mem_get_info is not
    # available or the driver query fails.
    try:
        state["vram"] = {
            "used": torch.cuda.memory_allocated() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
        state["last_update"] = time.time()
    except Exception:
        # Keep the server responsive even if the CUDA runtime is partially unusable.
        pass


def snapshot():
    """Get current state snapshot."""
    update_vram()
    return dict(state)
