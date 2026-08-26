"""FLUX image backend executed in an isolated Python runtime."""

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


_DEFAULT_FLUX2_9B_COMPONENT_MODEL = "black-forest-labs/FLUX.2-klein-9B"


class FluxDiffusersBackend(GenerationBackend):
    """Run FLUX-family Diffusers pipelines outside WebbDuck's core environment."""

    backend_id = "flux_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "flux"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        python_exe = os.getenv("WEBBDUCK_FLUX_PYTHON") or sys.executable
        required = [("diffusers", "Flux2KleinPipeline")]
        if descriptor.format == "gguf":
            required.extend(
                [
                    ("diffusers", "Flux2Transformer2DModel"),
                    ("diffusers", "GGUFQuantizationConfig"),
                    ("gguf", "GGUFReader"),
                ]
            )
        payload = probe_python_runtime(python_exe, required)
        if descriptor.format == "gguf":
            payload["model_format"] = "gguf"
            payload["component_model"] = str(
                descriptor.detection.get("component_model")
                or os.getenv("WEBBDUCK_FLUX2_COMPONENT_MODEL")
                or _DEFAULT_FLUX2_9B_COMPONENT_MODEL
            )
            if not payload.get("ready"):
                payload["repair"] = "python tools/prepare_model_runtimes.py flux"
        return payload

    def generate(
        self,
        descriptor: ModelDescriptor,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[list[Image.Image], int]:
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before FLUX runtime start")

        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        defaults = descriptor.defaults or {}
        seed_raw = settings.get("seed")
        seed = int(seed_raw) if seed_raw is not None else int(time.time_ns() & 0xFFFFFFFF)
        component_model = str(
            os.getenv("WEBBDUCK_FLUX2_COMPONENT_MODEL")
            or descriptor.detection.get("component_model")
            or _DEFAULT_FLUX2_9B_COMPONENT_MODEL
        )
        payload = {
            "model_path": descriptor.path,
            "model_format": descriptor.format,
            "component_model": component_model,
            "detection": dict(descriptor.detection or {}),
            "prompt": prompt,
            "input_image": str(settings.get("image") or "").strip() or None,
            "width": int(settings.get("width") or defaults.get("width") or 1024),
            "height": int(settings.get("height") or defaults.get("height") or 1024),
            "steps": int(settings.get("steps") or defaults.get("steps") or 4),
            "guidance": float(
                settings.get("cfg")
                if settings.get("cfg") is not None
                else defaults.get("cfg", 1.0)
            ),
            "num_images": max(1, int(settings.get("num_images") or 1)),
            "seed": seed,
        }

        python_exe = os.getenv("WEBBDUCK_FLUX_PYTHON") or sys.executable
        worker = Path(__file__).with_name("flux_worker.py")
        timeout_seconds = max(30.0, float(os.getenv("WEBBDUCK_FLUX_TIMEOUT_SECONDS", "1800")))

        with tempfile.TemporaryDirectory(prefix="webbduck_flux_") as tmp_raw:
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
                        raise GenerationCancelledError("FLUX generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(
                            f"FLUX runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    time.sleep(0.2)

            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            if not result_path.exists():
                detail = "\n".join(log_lines[-20:])
                raise RuntimeError(
                    f"FLUX runtime exited without a result (code {proc.returncode}).\n{detail}"
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                detail = str(result.get("error") or "FLUX runtime failed")
                logs = "\n".join(log_lines[-20:])
                raise RuntimeError(f"{detail}\n{logs}".strip())

            runtime = result.get("runtime")
            if isinstance(runtime, dict):
                settings["flux_runtime"] = runtime

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("FLUX runtime returned no images")
            return images, int(result.get("seed", seed))


_backend = FluxDiffusersBackend()


def ensure_registered() -> FluxDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
