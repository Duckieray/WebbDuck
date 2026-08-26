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
from core.backends.flux_lora import inject_lora_trigger, lora_trigger_phrase, resolve_flux_loras
from core.backends.runtime_probe import probe_python_runtime
from core.exceptions import GenerationCancelledError
from core.provider_credentials import resolve_provider_token
from models.model_descriptor import ModelDescriptor


_DEFAULT_FLUX2_9B_COMPONENT_MODEL = "black-forest-labs/FLUX.2-klein-9B"


def _huggingface_token() -> tuple[str, str | None]:
    """Resolve WebbDuck Settings/env credentials, then honor normal HF CLI auth."""
    token, source = resolve_provider_token("huggingface")
    if token:
        return token, source
    try:
        from huggingface_hub import get_token

        token = str(get_token() or "").strip()
    except Exception:
        token = ""
    return (token, "huggingface-cli") if token else ("", None)


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    token, _source = _huggingface_token()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    return env


def _component_access_error(
    component_model: str,
    exc: Exception,
    *,
    token_configured: bool,
) -> RuntimeError:
    text = str(exc or exc.__class__.__name__)
    lowered = text.lower()
    repo_url = f"https://huggingface.co/{component_model}"

    gated_markers = (
        "gated",
        "403",
        "forbidden",
        "restricted",
        "not a valid model identifier",
        "repository not found",
    )
    if any(marker in lowered for marker in gated_markers):
        if token_configured:
            return RuntimeError(
                f"FLUX.2 Klein 9B support components are gated on Hugging Face and the "
                f"configured token is not authorized for {component_model}. Open {repo_url}, "
                "accept the FLUX Non-Commercial License / access terms with the same Hugging "
                "Face account that owns the token, then retry. Your local GGUF does not need "
                "to be re-downloaded."
            )
        return RuntimeError(
            f"FLUX.2 Klein 9B support components are gated on Hugging Face. Open {repo_url}, "
            "accept the FLUX Non-Commercial License / access terms, then configure that "
            "account's Hugging Face token in WebbDuck Settings. Your local GGUF does not "
            "need to be re-downloaded."
        )

    if "401" in lowered or "unauthorized" in lowered or "invalid token" in lowered:
        return RuntimeError(
            f"Hugging Face rejected the credentials needed for {component_model}. Replace "
            "the Hugging Face token in WebbDuck Settings and make sure the FLUX.2 Klein 9B "
            f"access terms have been accepted at {repo_url}."
        )

    if "offline" in lowered or "localentrynotfound" in lowered or "local entry" in lowered:
        return RuntimeError(
            f"FLUX.2 Klein 9B needs support components from {component_model}, but they are "
            "not available in the local Hugging Face cache while offline. Cache the licensed "
            "support-component repository first or set WEBBDUCK_FLUX2_COMPONENT_MODEL to a "
            "complete local Diffusers component directory."
        )

    return RuntimeError(
        f"Unable to verify FLUX.2 Klein 9B support components from {component_model}: {text}"
    )


def _preflight_component_access(component_model: str) -> None:
    """Verify a GGUF checkpoint can obtain its non-transformer FLUX components."""
    local = Path(component_model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise RuntimeError(
                "WEBBDUCK_FLUX2_COMPONENT_MODEL must point to a complete local Diffusers "
                f"directory, not a file: {local}"
            )
        required = (
            local / "model_index.json",
            local / "transformer" / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "The configured local FLUX.2 component directory is incomplete. Missing: "
                + ", ".join(missing)
            )
        return

    token, _source = _huggingface_token()
    try:
        from huggingface_hub import hf_hub_download

        for filename in ("model_index.json", "transformer/config.json"):
            hf_hub_download(
                repo_id=component_model,
                filename=filename,
                token=token or None,
            )
    except Exception as exc:
        raise _component_access_error(
            component_model,
            exc,
            token_configured=bool(token),
        ) from exc


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
            str(event.get("stage") or "Generating with FLUX.2"),
            float(event.get("progress") or 0.0),
            int(event.get("step") or 0),
            int(event.get("total_steps") or 0),
        )
        return raw
    except Exception:
        return previous


class FluxDiffusersBackend(GenerationBackend):
    """Run FLUX-family Diffusers pipelines outside WebbDuck's core environment."""

    backend_id = "flux_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "flux"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        python_exe = os.getenv("WEBBDUCK_FLUX_PYTHON") or sys.executable
        required = [
            ("diffusers", "Flux2KleinPipeline"),
            ("peft", "LoraConfig"),
        ]
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
            token, source = _huggingface_token()
            payload["huggingface_credentials_configured"] = bool(token)
            payload["huggingface_credentials_source"] = source
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
        progress_callback = kwargs.get("progress_callback")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before FLUX runtime start")

        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        resolved_loras = resolve_flux_loras(settings.get("loras", []))
        trigger_phrase = lora_trigger_phrase(resolved_loras)
        prompt = inject_lora_trigger(prompt, trigger_phrase)
        if resolved_loras:
            settings["prompt"] = prompt
            settings["lora_trigger_phrase"] = trigger_phrase

        defaults = descriptor.defaults or {}
        seed_raw = settings.get("seed")
        seed = int(seed_raw) if seed_raw is not None else int(time.time_ns() & 0xFFFFFFFF)
        component_model = str(
            os.getenv("WEBBDUCK_FLUX2_COMPONENT_MODEL")
            or descriptor.detection.get("component_model")
            or _DEFAULT_FLUX2_9B_COMPONENT_MODEL
        )
        if descriptor.format == "gguf":
            _report_progress(progress_callback, "Checking FLUX.2 access", 0.03)
            _preflight_component_access(component_model)

        payload = {
            "model_path": descriptor.path,
            "model_format": descriptor.format,
            "component_model": component_model,
            "detection": dict(descriptor.detection or {}),
            "prompt": prompt,
            "loras": resolved_loras,
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
            progress_path = tmp / "progress.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            _report_progress(progress_callback, "Starting FLUX.2 runtime", 0.05)
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
                        "--progress",
                        str(progress_path),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=_worker_environment(),
                )
                started = time.monotonic()
                last_progress: str | None = None
                while proc.poll() is None:
                    last_progress = _forward_worker_progress(
                        progress_path,
                        progress_callback,
                        last_progress,
                    )
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
                _forward_worker_progress(progress_path, progress_callback, last_progress)

            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
            if not result_path.exists():
                detail = "\n".join(log_lines[-20:])
                raise RuntimeError(
                    f"FLUX runtime exited without a result (code {proc.returncode}).\n{detail}"
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                detail = str(result.get("error") or "FLUX runtime failed")
                worker_trace = str(result.get("traceback") or "").strip()
                logs = "\n".join(log_lines[-20:])
                combined = "\n".join(part for part in (detail, worker_trace, logs) if part)
                raise RuntimeError(combined.strip())

            runtime = result.get("runtime")
            if isinstance(runtime, dict):
                settings["flux_runtime"] = runtime

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("FLUX runtime returned no images")
            _report_progress(progress_callback, "FLUX.2 complete", 1.0)
            return images, int(result.get("seed", seed))


_backend = FluxDiffusersBackend()


def ensure_registered() -> FluxDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
