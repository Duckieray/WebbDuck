"""Lightweight probes for isolated model-runtime interpreters.

These checks intentionally import runtime libraries/classes without constructing
or loading model weights. They are safe to call from readiness/doctor surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Iterable


# Every currently installed image runtime reads safetensors checkpoints either
# directly or through Diffusers. Probe it centrally so a stale venv cannot be
# advertised as ready and then fail only after the user presses Generate.
_SHARED_REQUIRED_SYMBOLS = (("safetensors", "safe_open"),)


def probe_python_runtime(
    python_executable: str,
    required_symbols: Iterable[tuple[str, str]],
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    executable = Path(str(python_executable or "")).expanduser()
    if not executable.exists():
        return {
            "ready": False,
            "python": str(executable),
            "reason": "Configured runtime Python does not exist.",
        }

    requested = [*list(_SHARED_REQUIRED_SYMBOLS), *list(required_symbols)]
    requirements: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for module, symbol in requested:
        key = (str(module), str(symbol))
        if key in seen:
            continue
        seen.add(key)
        requirements.append([key[0], key[1]])

    script = r'''
import importlib
import json
import sys

requirements = json.loads(sys.argv[1])
out = {"ready": True, "python": sys.executable, "missing": []}
try:
    import torch
    out["torch_version"] = getattr(torch, "__version__", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        try:
            out["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            out["gpu_name"] = None
        try:
            out["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception:
            out["cuda_capability"] = None
        try:
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        except Exception:
            out["vram_gb"] = None
except Exception as exc:
    out["ready"] = False
    out["missing"].append({"module": "torch", "error": str(exc)})

try:
    import diffusers
    out["diffusers_version"] = getattr(diffusers, "__version__", None)
except Exception as exc:
    out["ready"] = False
    out["missing"].append({"module": "diffusers", "error": str(exc)})

for module_name, symbol_name in requirements:
    try:
        module = importlib.import_module(module_name)
        getattr(module, symbol_name)
    except Exception as exc:
        out["ready"] = False
        out["missing"].append({
            "module": module_name,
            "symbol": symbol_name,
            "error": str(exc),
        })

if out["ready"] and not out.get("cuda_available"):
    out["ready"] = False
    out["reason"] = "Runtime imports succeed, but CUDA is not available in this interpreter."
elif out["missing"]:
    out["reason"] = "One or more required runtime imports are unavailable."
print(json.dumps(out))
'''

    try:
        completed = subprocess.run(
            [str(executable), "-c", script, json.dumps(requirements)],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except Exception as exc:
        return {
            "ready": False,
            "python": str(executable),
            "reason": f"Runtime probe failed to start: {exc}",
        }

    stdout = (completed.stdout or "").strip().splitlines()
    if not stdout:
        detail = (completed.stderr or "").strip()
        return {
            "ready": False,
            "python": str(executable),
            "reason": f"Runtime probe returned no JSON (exit {completed.returncode}).",
            "detail": detail[-4000:],
        }

    try:
        payload = json.loads(stdout[-1])
    except Exception:
        return {
            "ready": False,
            "python": str(executable),
            "reason": f"Runtime probe returned invalid JSON (exit {completed.returncode}).",
            "detail": "\n".join(stdout[-10:])[-4000:],
        }

    payload["python"] = str(payload.get("python") or executable)
    payload["exit_code"] = int(completed.returncode)
    if completed.returncode != 0:
        payload["ready"] = False
        payload.setdefault("reason", "Runtime probe process exited unsuccessfully.")
    return payload
