"""Queue pause/resume coordination for the GPU worker.

Pausing does not interrupt the in-flight generation job (the diffusers loops
are not safely pre-emptible). Instead the worker finishes the current job,
releases the GPU (unloads resident models), and blocks before starting the
next queued job. Resuming wakes the worker and lets the queue drain again.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

_paused = False
_resume_event: asyncio.Event | None = None


def _resume_signal() -> asyncio.Event:
    """Lazily create the resume event so it binds to the running loop."""

    global _resume_event
    if _resume_event is None:
        _resume_event = asyncio.Event()
        _resume_event.set()
    return _resume_event


def is_queue_paused() -> bool:
    """Return True while the generation queue is paused."""

    return _paused


def set_queue_paused(paused: bool) -> bool:
    """Pause or resume the queue. Returns the resulting paused state."""

    global _paused
    _paused = bool(paused)
    event = _resume_signal()
    if _paused:
        event.clear()
    else:
        event.set()
    return _paused


async def wait_for_queue_resume() -> None:
    """Block until the queue is resumed (returns immediately when running)."""

    await _resume_signal().wait()


def release_gpu_for_pause() -> bool:
    """Best-effort release of host-resident GPU models while paused.

    Isolated backends (SDXL/FLUX/Krea/Qwen) run as job-scoped subprocesses, so
    their VRAM returns to the driver when the job exits. This clears the
    long-lived host holders (captioners and any in-process pipeline) so the GPU
    is genuinely idle while the queue is paused.
    """

    released = False

    try:
        from core.pipeline import pipeline_manager

        if getattr(pipeline_manager, "pipe", None) is not None:
            pipeline_manager.unload_all()
            released = True
    except Exception:
        log.exception("Failed to unload generation pipeline while pausing queue")

    try:
        from core.captioner import unload_captioners

        unload_captioners()
        released = True
    except Exception:
        log.exception("Failed to unload captioners while pausing queue")

    return released
