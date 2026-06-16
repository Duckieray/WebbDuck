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
    """Update current processing stage."""
    if stage != "Error":
        state["error_detail"] = None
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
    """Update VRAM usage stats."""
    if not torch.cuda.is_available():
        return
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
