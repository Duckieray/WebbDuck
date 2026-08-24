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


def _component_source(variant: str) -> str:
    explicit = str(os.getenv("WEBBDUCK_KREA2_COMPONENT_MODEL") or "").strip()
    if explicit:
        return explicit
    if variant == "turbo":
        return str(os.getenv("WEBBDUCK_KREA2_TURBO_COMPONENT_MODEL") or "krea/Krea-2-Turbo")
    return str(os.getenv("WEBBDUCK_KREA2_BASE_COMPONENT_MODEL") or "krea/Krea-2-Raw")


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
    return env


def _component_access_error(component_source: str, exc: Exception, *, token_configured: bool) -> RuntimeError:
    text = str(exc or exc.__class__.__name__)
    lowered = text.lower()
    repo_url = f"https://huggingface.co/{component_source}"

    if "403" in lowered or "gated" in lowered or "restricted" in lowered:
        if token_configured:
            return RuntimeError(
                f"Krea 2 support components are gated on Hugging Face and the configured token is not "
                f"authorized for {component_source}. Accept the Krea 2 Community License for that model "
                f"at {repo_url}, then retry. The selected local checkpoint does not need to be re-downloaded."
            )
        return RuntimeError(
            f"Krea 2 support components are gated on Hugging Face. Accept the Krea 2 Community License "
            f"for {component_source} at {repo_url}, then configure a Hugging Face token in WebbDuck "
            "Settings. The selected local checkpoint does not need to be re-downloaded."
        )

    if "401" in lowered or "unauthorized" in lowered or "invalid token" in lowered:
        return RuntimeError(
            f"Hugging Face rejected the credentials needed for Krea 2 support components ({component_source}). "
            "Replace the Hugging Face token in WebbDuck Settings and make sure the Krea 2 Community License "
            f"has been accepted at {repo_url}."
        )

    if "offline" in lowered or "localentrynotfound" in lowered or "local entry" in lowered:
        return RuntimeError(
            f"Krea 2 needs support components from {component_source}, but they are not available in the "
            "local Hugging Face cache while offline. Cache the licensed support-component repository first "
            "or set WEBBDUCK_KREA2_COMPONENT_MODEL to a complete local Diffusers component directory."
        )

    return RuntimeError(
        f"Unable to verify Krea 2 support components from {component_source}: {text}"
    )


def _preflight_component_access(component_source: str) -> None:
    """Verify a single-file Krea checkpoint can obtain its non-transformer assets.

    This intentionally downloads at most tiny JSON configuration files. It does
    not fetch model weights. The purpose is to detect gated/license/token failures
    before the expensive Krea transformer scaffold/overlay path starts.
    """
    local = Path(component_source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise RuntimeError(
                "WEBBDUCK_KREA2_COMPONENT_MODEL must point to a complete local Diffusers directory, "
                f"not a file: {local}"
            )
        required = (
            local / "model_index.json",
            local / "transformer" / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "The configured local Krea 2 component directory is incomplete. Missing: "
                + ", ".join(missing)
            )
        return

    token, _source = _huggingface_token()
    try:
        from huggingface_hub import hf_hub_download

        for filename in ("model_index.json", "transformer/config.json"):
            hf_hub_download(
                repo_id=component_source,
                filename=filename,
                token=token or None,
            )
    except Exception as exc:
        raise _component_access_error(
            component_source,
            exc,
            token_configured=bool(token),
        ) from exc


def _worker_error(result: dict[str, Any], log_lines: list[str], returncode: int | None) -> RuntimeError:
    detail = str(result.get("error") or "Krea 2 runtime failed")
    worker_trace = str(result.get("traceback") or "").strip()
    logs = "\n".join(log_lines[-20:]).strip()
    combined = "\n".join(part for part in (detail, worker_trace, logs) if part)
    lowered = combined.lower()

    # Keep this as a worker-side fallback in case access disappears after the
    # host preflight. Known gating failures stay concise in the UI; the worker
    # traceback remains in its job log for debugging.
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
            "Krea 2 support components are gated on Hugging Face. Accept the Krea 2 Community License "
            "for the selected Raw/Turbo support model and configure an authorized Hugging Face token in "
            "WebbDuck Settings. The selected local checkpoint does not need to be re-downloaded."
        )
    else:
        detail = combined

    if returncode not in (None, 0):
        detail = f"Krea 2 runtime exited with code {returncode}.\n{detail}"
    return RuntimeError(detail.strip())


def _report_progress(callback: Any, stage: str, progress: float, step: int = 0, total_steps: int = 0) -> None:
    if not callable(callback):
        return
    try:
        callback(stage, progress, step, total_steps)
    except Exception:
        # UI progress must never be allowed to fail inference.
        pass


def _forward_worker_progress(progress_path: Path, callback: Any, previous: str | None) -> str | None:
    if not callable(callback) or not progress_path.exists():
        return previous
    try:
        raw = progress_path.read_text(encoding="utf-8").strip()
        if not raw or raw == previous:
            return previous
        event = json.loads(raw)
        _report_progress(
            callback,
            str(event.get("stage") or "Generating with Krea 2"),
            float(event.get("progress") or 0.0),
            int(event.get("step") or 0),
            int(event.get("total_steps") or 0),
        )
        return raw
    except Exception:
        return previous


class Krea2DiffusersBackend(GenerationBackend):
    backend_id = "krea2_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "krea2"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        payload = probe_python_runtime(
            _runtime_python(),
            (
                ("diffusers", "Krea2Pipeline"),
                ("diffusers", "Krea2Transformer2DModel"),
                ("accelerate", "init_empty_weights"),
            ),
        )
        if not payload.get("ready"):
            payload["repair_hint"] = (
                "Repair/update this isolated runtime with "
                "`python tools/prepare_model_runtimes.py krea2`, then restart WebbDuck."
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
        variant = str(descriptor.detection.get("variant") or "base").lower()
        component_source = _component_source(variant)
        if descriptor.format == "single":
            _report_progress(progress_callback, "Checking Krea components", 0.03)
            _preflight_component_access(component_source)

        payload = {
            "model_path": descriptor.path,
            "model_format": descriptor.format,
            "variant": variant,
            "quantization": descriptor.detection.get("quantization"),
            "component_source": component_source,
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
            progress_path = tmp / "progress.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            _report_progress(progress_callback, "Starting Krea runtime", 0.05)
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
                        raise GenerationCancelledError("Krea 2 generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(f"Krea 2 runtime timed out after {int(timeout_seconds)} seconds")
                    time.sleep(0.2)
                _forward_worker_progress(progress_path, progress_callback, last_progress)

            logs = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:] if log_path.exists() else []
            if not result_path.exists():
                raise _worker_error({}, logs, proc.returncode)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise _worker_error(result, logs, proc.returncode)

            timing = result.get("timing")
            if isinstance(timing, dict):
                perf = settings.setdefault("performance_timing", {})
                if isinstance(perf, dict):
                    for key, value in timing.items():
                        try:
                            perf[f"krea_{key}"] = round(float(value), 6)
                        except (TypeError, ValueError):
                            continue

            runtime = result.get("runtime")
            if isinstance(runtime, dict):
                settings["krea_runtime"] = runtime

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
