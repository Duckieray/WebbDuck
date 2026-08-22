"""Krea 2 image backend executed in an isolated Python runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from core.backends.base import GenerationBackend, backend_resolver
from core.exceptions import GenerationCancelledError
from models.model_descriptor import ModelDescriptor


class Krea2DiffusersBackend(GenerationBackend):
    backend_id = "krea2_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "krea2"

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

        defaults = descriptor.defaults or {}
        seed = int(settings.get("seed") or int(time.time_ns() & 0xFFFFFFFF))
        payload = {
            "model_path": descriptor.path,
            "prompt": prompt,
            "width": int(settings.get("width") or defaults.get("width") or 1024),
            "height": int(settings.get("height") or defaults.get("height") or 1024),
            "steps": int(settings.get("steps") or defaults.get("steps") or 8),
            "guidance": float(settings.get("cfg") if settings.get("cfg") is not None else defaults.get("cfg", 0.0)),
            "num_images": max(1, int(settings.get("num_images") or 1)),
            "seed": seed,
        }

        python_exe = os.getenv("WEBBDUCK_KREA2_PYTHON") or sys.executable
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
                        "--request",
                        str(request_path),
                        "--result",
                        str(result_path),
                        "--output-dir",
                        str(tmp),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
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

            logs = log_path.read_text(encoding="utf-8").splitlines()[-80:] if log_path.exists() else []
            if not result_path.exists():
                raise RuntimeError(
                    f"Krea 2 runtime exited without a result (code {proc.returncode}).\n" + "\n".join(logs[-20:])
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (str(result.get("error") or "Krea 2 runtime failed") + "\n" + "\n".join(logs[-20:])).strip()
                )

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
