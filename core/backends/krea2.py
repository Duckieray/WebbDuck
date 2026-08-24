"""Krea 2 image backend executed in an isolated Python runtime."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from core.backends.base import GenerationBackend, backend_resolver
from core.backends.runtime_probe import probe_python_runtime
from core.exceptions import GenerationCancelledError
from core.provider_credentials import resolve_provider_token
from models.model_descriptor import ModelDescriptor


def _runtime_python() -> str:
    configured = str(os.getenv("WEBBDUCK_KREA2_PYTHON") or "").strip()
    if configured:
        return configured
    runtime_home = Path(
        os.getenv("WEBBDUCK_RUNTIME_HOME", "~/.local/share/webbduck/runtimes")
    ).expanduser()
    suffix = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return str(runtime_home / "krea2" / suffix)


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    token, _source = resolve_provider_token("huggingface")
    if token:
        env["HF_TOKEN"] = token
    return env


def _worker_error(result: dict[str, Any], log_lines: list[str], returncode: int | None) -> RuntimeError:
    detail = str(result.get("error") or "Krea 2 runtime failed")
    worker_trace = str(result.get("traceback") or "").strip()
    logs = "\n".join(log_lines[-20:]).strip()
    combined = "\n".join(part for part in (detail, worker_trace, logs) if part)
    lowered = combined.lower()

    # Local single-file Krea checkpoints still use the official Krea repository
    # for non-transformer components. Hugging Face intentionally reports gated
    # repositories using the same generic repository-not-found wording, which is
    # otherwise very misleading to a WebbDuck user.
    if (
        "krea/krea-2-" in lowered
        and (
            "not a valid model identifier" in lowered
            or "gated repo" in lowered
            or "restricted" in lowered
            or "401" in lowered
            or "403" in lowered
        )
    ):
        detail = (
            "Krea 2 support components are gated on Hugging Face. Accept the Krea 2 "
            "Community License for the selected Raw/Turbo model and configure a Hugging "
            "Face token in WebbDuck Settings. The local checkpoint itself does not need "
            "to be re-downloaded."
        )
        if worker_trace:
            detail = f"{detail}\n{worker_trace}"
    else:
        detail = combined

    if returncode not in (None, 0):
        detail = f"Krea 2 runtime exited with code {returncode}.\n{detail}"
    return RuntimeError(detail.strip())


class Krea2DiffusersBackend(GenerationBackend):
    backend_id = "krea2_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "krea2"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        return probe_python_runtime(
            _runtime_python(),
            (
                ("diffusers", "Krea2Pipeline"),
                ("diffusers", "Krea2Transformer2DModel"),
                ("accelerate", "init_empty_weights"),
            ),
        )

    def generate(
        self,
        descriptor: ModelDescriptor,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[list[Image.Image], int]:
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before Krea 2 runtime start")

        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        if not descriptor.supported:
            quant = str(descriptor.detection.get("quantization") or "unknown")
            raise RuntimeError(
                f"Krea checkpoint '{descriptor.name}' is recognized but its {quant} single-file "
                "format is not runnable by the installed Krea backend."
            )

        defaults = descriptor.defaults or {}
        raw_seed = settings.get("seed")
        seed = int(raw_seed) if raw_seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        payload = {
            "model_path": descriptor.path,
            "model_format": descriptor.format,
            "variant": descriptor.detection.get("variant"),
            "quantization": descriptor.detection.get("quantization"),
            "prompt": prompt,
            "width": int(settings.get("width") or defaults.get("width") or 1024),
            "height": int(settings.get("height") or defaults.get("height") or 1024),
            "steps": int(settings.get("steps") or defaults.get("steps") or 28),
            "guidance": float(settings.get("cfg") if settings.get("cfg") is not None else defaults.get("cfg", 4.5)),
            "num_images": max(1, int(settings.get("num_images") or 1)),
            "seed": seed,
        }

        python_exe = _runtime_python()
        worker = Path(__file__).with_name("krea2_worker.py")
        timeout_seconds = max(30.0, float(os.getenv("WEBBDUCK_KREA2_TIMEOUT_SECONDS", "1800")))

        with tempfile.TemporaryDirectory(prefix="webbduck_krea2_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    [
                        python_exe,
                        str(worker),
                        "--request", str(request_path),
                        "--result", str(result_path),
                        "--output-dir", str(tmp),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=_worker_environment(),
                )
                started = time.monotonic()
                while proc.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise GenerationCancelledError("Krea 2 generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(f"Krea 2 runtime timed out after {int(timeout_seconds)} seconds")
                    time.sleep(0.2)

            logs = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:] if log_path.exists() else []
            if not result_path.exists():
                raise _worker_error({}, logs, proc.returncode)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise _worker_error(result, logs, proc.returncode)

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("Krea 2 runtime returned no images")
            return images, int(result.get("seed", seed))


_backend = Krea2DiffusersBackend()


def ensure_registered() -> Krea2DiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
