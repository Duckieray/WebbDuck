"""Qwen-Image-2512 backend executed in an isolated Diffusers runtime."""

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
from core.backends.runtime_probe import probe_python_runtime
from core.exceptions import GenerationCancelledError
from models.model_descriptor import ModelDescriptor


class QwenImageDiffusersBackend(GenerationBackend):
    backend_id = "qwen_image_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "qwen_image"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        python_exe = os.getenv("WEBBDUCK_QWEN_IMAGE_PYTHON") or sys.executable
        return probe_python_runtime(
            python_exe,
            (("diffusers", "QwenImagePipeline"),),
        )

    def generate(
        self,
        descriptor: ModelDescriptor,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[list[Image.Image], int]:
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before Qwen Image runtime start")

        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        defaults = descriptor.defaults or {}
        raw_seed = settings.get("seed")
        seed = int(raw_seed) if raw_seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        payload = {
            "model_path": descriptor.path,
            "prompt": prompt,
            "negative_prompt": str(settings.get("negative_prompt") or ""),
            "width": int(settings.get("width") or defaults.get("width") or 1328),
            "height": int(settings.get("height") or defaults.get("height") or 1328),
            "steps": int(settings.get("steps") or defaults.get("steps") or 50),
            "true_cfg_scale": float(
                settings.get("cfg")
                if settings.get("cfg") is not None
                else defaults.get("cfg", 4.0)
            ),
            "num_images": max(1, int(settings.get("num_images") or 1)),
            "seed": seed,
        }

        python_exe = os.getenv("WEBBDUCK_QWEN_IMAGE_PYTHON") or sys.executable
        worker = Path(__file__).with_name("qwen_image_worker.py")
        timeout_seconds = max(60.0, float(os.getenv("WEBBDUCK_QWEN_IMAGE_TIMEOUT_SECONDS", "3600")))

        with tempfile.TemporaryDirectory(prefix="webbduck_qwen_image_") as tmp_raw:
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
                )
                started = time.monotonic()
                while proc.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise GenerationCancelledError("Qwen Image generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(
                            f"Qwen Image runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    time.sleep(0.25)

            logs = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                if log_path.exists()
                else []
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"Qwen Image runtime exited without a result (code {proc.returncode}).\n"
                    + "\n".join(logs[-20:])
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (
                        str(result.get("error") or "Qwen Image runtime failed")
                        + "\n"
                        + "\n".join(logs[-20:])
                    ).strip()
                )

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("Qwen Image runtime returned no images")
            return images, int(result.get("seed", seed))


_backend = QwenImageDiffusersBackend()


def ensure_registered() -> QwenImageDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
