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
    "last_update": time.time(),
}


def update_stage(stage: str):
    """Update current processing stage."""
    state["stage"] = stage
    state["last_update"] = time.time()


def update_progress(p: float):
    """Update progress (0.0 to 1.0)."""
    state["progress"] = float(p)
    state["last_update"] = time.time()


def update_vram():
    """Update VRAM usage stats."""
    if not torch.cuda.is_available():
        return
    state["vram"] = {
        "used": torch.cuda.memory_allocated() / 1024**3,
        "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }
    state["last_update"] = time.time()


def snapshot():
    """Get current state snapshot."""
    update_vram()
    return dict(state)