"""Lightweight per-generation performance timing utilities."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict


_local = threading.local()


def _metrics() -> Dict[str, float]:
    metrics = getattr(_local, "metrics", None)
    if metrics is None:
        metrics = {}
        _local.metrics = metrics
    return metrics


def reset_metrics() -> None:
    """Reset timing metrics for the current thread."""
    _local.metrics = {}


def add_duration(name: str, seconds: float) -> None:
    """Accumulate a stage duration (in seconds)."""
    key = str(name or "").strip()
    if not key:
        return
    value = float(max(0.0, seconds))
    metrics = _metrics()
    metrics[key] = float(metrics.get(key, 0.0)) + value


@contextmanager
def stage_timer(name: str):
    """Context manager that adds elapsed time to a named metric."""
    start = time.perf_counter()
    try:
        yield
    finally:
        add_duration(name, time.perf_counter() - start)


def snapshot_metrics() -> Dict[str, float]:
    """Return a JSON-safe snapshot of current metrics."""
    metrics = _metrics()
    return {k: round(float(v), 6) for k, v in metrics.items()}
