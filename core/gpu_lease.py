"""In-process GPU lease coordination between WebbDuck core and plugins.

Provides a simple token-based lease system so that GPU-heavy runtimes (WebbDuck
generation, DuckMotion, IP-Adapter captioning, etc.) do not step on each other.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Any

log = logging.getLogger("webbduck.gpu_lease")

# Default idle timeout before a lease is considered stale (seconds).
# Set via WEBBDUCK_LEASE_IDLE_TIMEOUT or DUCKMOTION_LEASE_IDLE_TIMEOUT env var.
_LEASE_IDLE_TIMEOUT = float(os.getenv("WEBBDUCK_LEASE_IDLE_TIMEOUT") or os.getenv("DUCKMOTION_LEASE_IDLE_TIMEOUT") or "300.0")

_LEASE: dict[str, Any] = {
    "held": False,
    "owner": None,
    "owner_kind": None,
    "label": None,
    "token": None,
    "acquired_at": None,
    "job_id": None,
    "last_progress_at": None,
}
_LEASE_LOCK = threading.RLock()
_CONDITION = threading.Condition(_LEASE_LOCK)


def acquire_gpu_lease_blocking(
    *,
    owner: str = "unknown",
    owner_kind: str = "unknown",
    label: str = "",
    job_id: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Block until the GPU lease is acquired or *timeout_seconds* elapses.

    Returns ``{"acquired": True, "lease": {...}}`` on success or
    ``{"acquired": False, "lease": {...}, "waited_seconds": ...}`` on timeout.
    """
    deadline = (time.monotonic() + timeout_seconds) if (timeout_seconds is not None and timeout_seconds > 0) else None

    with _CONDITION:
        while _LEASE["held"]:
            # Auto-release stale leases that have stopped sending heartbeats.
            last_progress = _LEASE.get("last_progress_at")
            if last_progress is not None:
                idle = time.time() - last_progress
                if idle > _LEASE_IDLE_TIMEOUT:
                    held_by = f"{_LEASE.get('owner')}/{_LEASE.get('label')}"
                    log.warning(
                        "Auto-releasing stale GPU lease held by %s (idle %.0fs > %.0fs)",
                        held_by, idle, _LEASE_IDLE_TIMEOUT,
                    )
                    _clear_lease_locked()
                    _CONDITION.notify_all()
                    break
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                waited = _LEASE.get("acquired_at")
                waited = (time.time() - waited) if waited else None
                return {"acquired": False, "lease": dict(_LEASE), "waited_seconds": waited}
            _CONDITION.wait(timeout=remaining)
        _set_lease_locked(owner=owner, owner_kind=owner_kind, label=label, job_id=job_id)
        return {"acquired": True, "lease": dict(_LEASE)}


async def wait_for_gpu_lease_async(
    *,
    owner: str = "unknown",
    owner_kind: str = "unknown",
    label: str = "",
    job_id: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Async equivalent of :func:`acquire_gpu_lease_blocking`."""
    loop = asyncio.get_running_loop()
    from functools import partial as _partial
    return await loop.run_in_executor(
        None,
        _partial(
            acquire_gpu_lease_blocking,
            owner=owner,
            owner_kind=owner_kind,
            label=label,
            job_id=job_id,
            timeout_seconds=timeout_seconds,
        ),
    )


def release_gpu_lease(*, token: str) -> bool:
    """Release the lease identified by *token*.

    Returns ``True`` if the lease was released, ``False`` if *token* does not
    match the current lease holder.
    """
    with _CONDITION:
        if not _LEASE["held"]:
            return False
        if str(_LEASE.get("token") or "") != str(token or ""):
            return False
        _clear_lease_locked()
        _CONDITION.notify_all()
        return True


def get_gpu_lease() -> dict[str, Any]:
    """Return a snapshot of the current lease state."""
    with _LEASE_LOCK:
        return dict(_LEASE)


def _set_lease_locked(*, owner: str, owner_kind: str, label: str, job_id: str | None) -> None:
    token = uuid.uuid4().hex
    _LEASE.update(
        {
            "held": True,
            "owner": str(owner),
            "owner_kind": str(owner_kind),
            "label": str(label),
            "token": token,
            "acquired_at": time.time(),
            "job_id": str(job_id) if job_id else None,
        }
    )
    log.info("GPU lease acquired: owner=%s kind=%s label=%s token=%s", owner, owner_kind, label, token[:8])


def _clear_lease_locked() -> None:
    _LEASE.update(
        {
            "held": False,
            "owner": None,
            "owner_kind": None,
            "label": None,
            "token": None,
            "acquired_at": None,
            "job_id": None,
            "last_progress_at": None,
        }
    )


def lease_heartbeat(*, token: str) -> bool:
    """Update the progress timestamp for the current lease holder.

    Call this periodically from the lease holder while doing GPU work.
    The lease system will auto-release leases that go idle past
    ``WEBBDUCK_LEASE_IDLE_TIMEOUT`` (default 300s).
    Returns ``True`` if the heartbeat was accepted.
    """
    with _LEASE_LOCK:
        if not _LEASE["held"]:
            return False
        if str(_LEASE.get("token") or "") != str(token or ""):
            return False
        _LEASE["last_progress_at"] = time.time()
        return True
