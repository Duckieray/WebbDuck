"""Shared in-process GPU lease coordination for WebbDuck and web plugins.

This module provides a lightweight mutual exclusion mechanism for GPU-heavy
workloads that run inside the same WebbDuck process (for example core image
generation and local web plugins such as DuckMotion).
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class GPULease:
    token: str
    owner: str
    owner_kind: str
    label: str
    job_id: str | None
    acquired_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "owner": self.owner,
            "owner_kind": self.owner_kind,
            "label": self.label,
            "job_id": self.job_id,
            "acquired_at": self.acquired_at,
        }


_LEASE_LOCK = threading.RLock()
_LEASE: GPULease | None = None


def get_gpu_lease() -> dict[str, Any]:
    with _LEASE_LOCK:
        lease = _LEASE
        return {
            "held": lease is not None,
            "lease": lease.to_dict() if lease is not None else None,
        }


def try_acquire_gpu_lease(
    *,
    owner: str,
    owner_kind: str,
    label: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Attempt to acquire the shared GPU lease without blocking."""
    global _LEASE
    now = time.time()
    owner_value = str(owner or "").strip() or "unknown"
    kind_value = str(owner_kind or "").strip() or "unknown"
    label_value = str(label or "").strip() or kind_value
    job_value = str(job_id).strip() if job_id is not None else None

    with _LEASE_LOCK:
        if _LEASE is None:
            lease = GPULease(
                token=uuid.uuid4().hex,
                owner=owner_value,
                owner_kind=kind_value,
                label=label_value,
                job_id=job_value,
                acquired_at=now,
            )
            _LEASE = lease
            return {"acquired": True, "lease": lease.to_dict()}

        # Re-entrant for exact owner/job pair (same holder refreshing state).
        if (
            _LEASE.owner == owner_value
            and _LEASE.owner_kind == kind_value
            and (_LEASE.job_id or None) == (job_value or None)
        ):
            return {"acquired": True, "lease": _LEASE.to_dict(), "reentrant": True}

        return {"acquired": False, "lease": _LEASE.to_dict()}


def release_gpu_lease(*, token: str | None = None, owner: str | None = None) -> bool:
    """Release the shared GPU lease if the caller matches the current holder."""
    global _LEASE
    token_value = str(token or "").strip()
    owner_value = str(owner or "").strip()

    with _LEASE_LOCK:
        if _LEASE is None:
            return False
        if token_value and _LEASE.token == token_value:
            _LEASE = None
            return True
        if owner_value and _LEASE.owner == owner_value:
            _LEASE = None
            return True
        return False


async def wait_for_gpu_lease_async(
    *,
    owner: str,
    owner_kind: str,
    label: str,
    job_id: str | None = None,
    poll_seconds: float = 0.15,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Acquire the shared GPU lease using async polling."""
    started = time.time()
    while True:
        attempt = try_acquire_gpu_lease(
            owner=owner,
            owner_kind=owner_kind,
            label=label,
            job_id=job_id,
        )
        if attempt.get("acquired"):
            return attempt
        if timeout_seconds is not None:
            elapsed = time.time() - started
            if elapsed >= max(0.0, float(timeout_seconds)):
                lease = attempt.get("lease") if isinstance(attempt, dict) else None
                return {
                    "acquired": False,
                    "timeout": True,
                    "waited_seconds": elapsed,
                    "lease": lease if isinstance(lease, dict) else None,
                }
        await asyncio.sleep(max(0.02, float(poll_seconds)))


def acquire_gpu_lease_blocking(
    *,
    owner: str,
    owner_kind: str,
    label: str,
    job_id: str | None = None,
    poll_seconds: float = 0.15,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Acquire the shared GPU lease with blocking polling (thread-friendly)."""
    started = time.time()
    while True:
        attempt = try_acquire_gpu_lease(
            owner=owner,
            owner_kind=owner_kind,
            label=label,
            job_id=job_id,
        )
        if attempt.get("acquired"):
            return attempt
        if timeout_seconds is not None:
            elapsed = time.time() - started
            if elapsed >= max(0.0, float(timeout_seconds)):
                lease = attempt.get("lease") if isinstance(attempt, dict) else None
                return {
                    "acquired": False,
                    "timeout": True,
                    "waited_seconds": elapsed,
                    "lease": lease if isinstance(lease, dict) else None,
                }
        time.sleep(max(0.02, float(poll_seconds)))
