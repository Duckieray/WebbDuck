"""MOSS-Image backend executed in an isolated Diffusers runtime.

MOSS-Image support components (Qwen3.5-9B text encoder + FLUX.2 VAE + the
remote ``pipeline_moss_image`` module) come from a Diffusers repository. Any
MOSS-Image-based single-file DiT can be overlaid on those components, so the
backend never hardcodes one published checkpoint.
"""

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
from core.backends.runtime_utils import resolve_runtime_python
from core.exceptions import GenerationCancelledError
from models.model_descriptor import ModelDescriptor


DEFAULT_COMPONENT_MODEL = "OpenMOSS-Team/MOSS-Image-Base"
DIMENSION_MULTIPLE = 16


def _component_model(descriptor: ModelDescriptor) -> str:
    return str(
        os.getenv("WEBBDUCK_MOSS_IMAGE_COMPONENT_MODEL")
        or descriptor.detection.get("component_model")
        or DEFAULT_COMPONENT_MODEL
    ).strip()


def _snap(value: int, multiple: int = DIMENSION_MULTIPLE) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


_MOSS_LORA_ARCHES = {"moss", "moss_image", "moss_image_diffusers"}


def _resolve_loras(raw_loras: Any) -> list[dict[str, Any]]:
    """Resolve Studio LoRA names to files the isolated worker can load.

    Studio submits LoRAs by registry name, exactly like the SDXL/FLUX paths. Keep
    that contract and send explicit paths/weights so the worker does not depend
    on the host registry.
    """
    if not isinstance(raw_loras, list) or not raw_loras:
        return []

    from models.registry import LORA_REGISTRY

    resolved: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_loras):
        if isinstance(raw, str):
            name = raw.strip()
            requested_weight = None
        elif isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("model") or "").strip()
            requested_weight = raw.get("weight", raw.get("strength"))
        else:
            raise ValueError(f"Invalid LoRA entry at index {index}: expected name or object")

        if not name:
            raise ValueError(f"Invalid LoRA entry at index {index}: missing name")
        registry_entry = LORA_REGISTRY.get(name)
        if registry_entry is None:
            raise ValueError(f"Unknown LoRA: {name}")

        arch = str(registry_entry.get("arch") or "").strip().lower()
        # Unclassified adapters are allowed through; the worker validates the
        # adapter against the DiT tensor layout and fails with the real cause.
        if arch and arch not in _MOSS_LORA_ARCHES:
            raise ValueError(
                f"LoRA '{name}' targets {arch}, not MOSS-Image."
            )

        path = Path(str(registry_entry.get("path") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file not found for '{name}': {path}")

        weight_raw = requested_weight
        if weight_raw is None:
            weight_raw = registry_entry.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LoRA weight for '{name}': {weight_raw!r}") from exc

        resolved.append(
            {
                "name": name,
                "path": str(path),
                "weight": weight,
                "trigger": str(registry_entry.get("trigger") or "").strip() or None,
            }
        )
    return resolved


def _inject_lora_triggers(prompt: str, loras: list[dict[str, Any]]) -> str:
    """Append trigger words from selected LoRAs to the text prompt.

    MOSS-Image conditions on a Qwen3.5 text encoder, so plain words work;
    ``(token:weight)`` syntax does not and would only feed punctuation to the
    encoder. The Studio weight slider controls the adapter strength separately.
    """
    prompt = str(prompt or "").strip()
    triggers: list[str] = []
    for entry in loras:
        trigger = str(entry.get("trigger") or "").strip()
        if trigger and trigger not in triggers:
            triggers.append(trigger)
    missing = [trigger for trigger in triggers if trigger not in prompt]
    if not missing:
        return prompt
    suffix = ", ".join(missing)
    return f"{prompt}, {suffix}" if prompt else suffix


def _report_progress(
    callback: Any,
    stage: str,
    progress: float,
    step: int = 0,
    total_steps: int = 0,
) -> None:
    if not callable(callback):
        return
    try:
        callback(stage, progress, step, total_steps)
    except Exception:
        pass


def _forward_worker_progress(
    progress_path: Path,
    callback: Any,
    previous: str | None,
) -> str | None:
    if not callable(callback) or not progress_path.exists():
        return previous
    try:
        raw = progress_path.read_text(encoding="utf-8").strip()
        if not raw or raw == previous:
            return previous
        event = json.loads(raw)
        _report_progress(
            callback,
            str(event.get("stage") or "Generating with MOSS-Image"),
            float(event.get("progress") or 0.0),
            int(event.get("step") or 0),
            int(event.get("total_steps") or 0),
        )
        return raw
    except Exception:
        return previous


class MossImageDiffusersBackend(GenerationBackend):
    """Run MOSS-Image text-to-image pipelines outside WebbDuck's core env."""

    backend_id = "moss_image_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return (
            descriptor.backend == self.backend_id
            and descriptor.architecture == "moss_image"
        )

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        python_exe = resolve_runtime_python("moss_image", "WEBBDUCK_MOSS_IMAGE_PYTHON")
        payload = probe_python_runtime(
            python_exe,
            (
                ("diffusers", "AutoencoderKLFlux2"),
                ("transformers", "Qwen3_5Model"),
                ("accelerate", "init_empty_weights"),
                ("safetensors", "safe_open"),
            ),
        )
        payload["model_format"] = descriptor.format
        payload["component_model"] = _component_model(descriptor)
        if not payload.get("ready"):
            payload["repair_hint"] = (
                "Repair/update this isolated runtime with "
                "`python tools/prepare_model_runtimes.py moss_image`, then restart WebbDuck."
            )
        return payload

    def generate(
        self,
        descriptor: ModelDescriptor,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[list[Image.Image], int]:
        cancel_event = kwargs.get("cancel_event")
        progress_callback = kwargs.get("progress_callback")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before MOSS-Image runtime start")

        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        resolved_loras = _resolve_loras(settings.get("loras", []))
        prompt = _inject_lora_triggers(prompt, resolved_loras)

        defaults = descriptor.defaults or {}
        raw_seed = settings.get("seed")
        seed = int(raw_seed) if raw_seed is not None else int(time.time_ns() & 0xFFFFFFFF)

        payload = {
            "model_path": descriptor.path,
            "model_format": descriptor.format,
            "component_model": _component_model(descriptor),
            "prompt": prompt,
            "negative_prompt": str(settings.get("negative_prompt") or ""),
            "width": _snap(int(settings.get("width") or defaults.get("width") or 1024)),
            "height": _snap(int(settings.get("height") or defaults.get("height") or 1024)),
            "steps": int(settings.get("steps") or defaults.get("steps") or 50),
            "guidance_scale": float(
                settings.get("cfg")
                if settings.get("cfg") is not None
                else defaults.get("cfg", 3.0)
            ),
            "num_images": max(1, int(settings.get("num_images") or 1)),
            "seed": seed,
            "loras": resolved_loras,
        }

        python_exe = resolve_runtime_python("moss_image", "WEBBDUCK_MOSS_IMAGE_PYTHON")
        worker = Path(__file__).with_name("moss_image_worker.py")
        timeout_seconds = max(60.0, float(os.getenv("WEBBDUCK_MOSS_IMAGE_TIMEOUT_SECONDS", "3600")))

        with tempfile.TemporaryDirectory(prefix="webbduck_moss_image_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            progress_path = tmp / "progress.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            _report_progress(progress_callback, "Starting MOSS-Image runtime", 0.05)
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    [
                        python_exe,
                        str(worker),
                        "--request", str(request_path),
                        "--result", str(result_path),
                        "--output-dir", str(tmp),
                        "--progress", str(progress_path),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                started = time.monotonic()
                last_progress: str | None = None
                while proc.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise GenerationCancelledError("MOSS-Image generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        raise RuntimeError(
                            f"MOSS-Image runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    last_progress = _forward_worker_progress(
                        progress_path, progress_callback, last_progress
                    )
                    time.sleep(0.25)
                _forward_worker_progress(progress_path, progress_callback, last_progress)

            logs = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                if log_path.exists()
                else []
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"MOSS-Image runtime exited without a result (code {proc.returncode}).\n"
                    + "\n".join(logs[-20:])
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (
                        str(result.get("error") or "MOSS-Image runtime failed")
                        + "\n"
                        + "\n".join(logs[-20:])
                    ).strip()
                )

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("MOSS-Image runtime returned no images")
            _report_progress(progress_callback, "MOSS-Image complete", 1.0)
            return images, int(result.get("seed", seed))


_backend = MossImageDiffusersBackend()


def ensure_registered() -> MossImageDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
