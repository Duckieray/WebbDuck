"""Shared helpers for resolving isolated image-runtime interpreters."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_runtime_python(runtime: str, env_var: str) -> str:
    """Resolve the Python interpreter for an isolated generation runtime.

    An explicit ``WEBBDUCK_<RUNTIME>_PYTHON`` override wins.  Otherwise fall
    back to the standard per-runtime layout under ``WEBBDUCK_RUNTIME_HOME``
    (default ``~/.local/share/webbduck/runtimes/<runtime>/``).
    """
    configured = str(os.getenv(env_var) or "").strip()
    if configured:
        return configured
    runtime_home = Path(
        os.getenv("WEBBDUCK_RUNTIME_HOME", "~/.local/share/webbduck/runtimes")
    ).expanduser()
    suffix = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return str(runtime_home / runtime / suffix)