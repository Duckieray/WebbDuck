"""SDXL backend exposed through WebbDuck's generic generation contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from PIL import Image

from core.backends.base import GenerationBackend, backend_resolver
from core.backends.runtime_probe import probe_python_runtime
from core.exceptions import GenerationCancelledError
from models.model_descriptor import ModelDescriptor


def _hydrate_execution_registry() -> None:
    """Populate the mature SDXL stack from the canonical model catalog.

    The backend-owned SDXL implementation still uses a name->entry mapping for
    checkpoint/refiner/asset lookup. That mapping is no longer a discovery
    authority: rebuild its SDXL model entries from the architecture-neutral
    runtime catalog immediately before execution.
    """
    from models.catalog import runtime_registry
    from models.registry import MODEL_REGISTRY

    canonical = runtime_registry()
    sdxl_entries: dict[str, dict[str, Any]] = {}
    for name, raw_entry in canonical.items():
        entry = dict(raw_entry)
        if str(entry.get("arch") or entry.get("architecture") or "").lower() != "sdxl":
            continue
        entry["path"] = Path(str(entry.get("path") or ""))
        sdxl_entries[str(name)] = entry

    for name in list(MODEL_REGISTRY):
        if str(MODEL_REGISTRY.get(name, {}).get("arch") or "").lower() == "sdxl":
            MODEL_REGISTRY.pop(name, None)
    MODEL_REGISTRY.update(sdxl_entries)


def _runtime_python() -> str:
    configured = str(os.getenv("WEBBDUCK_SDXL_PYTHON") or "").strip()
    if configured:
        return configured
    runtime_home = Path(
        os.getenv("WEBBDUCK_RUNTIME_HOME", "~/.local/share/webbduck/runtimes")
    ).expanduser()
    suffix = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return str(runtime_home / "sdxl" / suffix)


def _apply_progress_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    try:
        from server.state import update_progress, update_stage

        if payload.get("stage"):
            update_stage(str(payload["stage"]))
        if payload.get("progress") is not None:
            update_progress(
                float(payload.get("progress") or 0.0),
                step=int(payload.get("step") or 0),
                total_steps=int(payload.get("total_steps") or 0),
            )
    except Exception:
        pass


def _worker_error(result: dict[str, Any], log_lines: list[str], returncode: int | None) -> RuntimeError:
    detail = str(result.get("error") or "SDXL runtime failed")
    worker_trace = str(result.get("traceback") or "").strip()
    logs = "\n".join(log_lines[-20:]).strip()
    suffix = "\n".join(part for part in (worker_trace, logs) if part)
    if suffix:
        detail = f"{detail}\n{suffix}"
    if returncode not in (None, 0):
        detail = f"SDXL runtime exited with code {returncode}.\n{detail}"
    return RuntimeError(detail.strip())


class SDXLDiffusersBackend(GenerationBackend):
    """Run the mature SDXL stack in its own Python environment."""

    backend_id = "sdxl_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "sdxl"

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "Checkpoint is not handled by this backend."}
        payload = probe_python_runtime(
            _runtime_python(),
            (
                ("diffusers", "StableDiffusionXLPipeline"),
                ("diffusers", "StableDiffusionXLImg2ImgPipeline"),
                ("diffusers", "StableDiffusionXLInpaintPipeline"),
                ("transformers", "CLIPTokenizer"),
            ),
        )
        if not payload.get("ready"):
            payload["repair_hint"] = (
                "Repair/update this isolated runtime with "
                "`python tools/prepare_model_runtimes.py sdxl`, then restart WebbDuck."
            )
        return payload

    def generate(
        self,
        descriptor: ModelDescriptor,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[list[Image.Image], int]:
        selected = str(settings.get("base_model") or "")
        if selected and selected != descriptor.name:
            raise ValueError(
                f"Selected checkpoint changed during routing: {selected!r} != {descriptor.name!r}"
            )

        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled before SDXL runtime start")

        python_exe = _runtime_python()
        worker = Path(__file__).with_name("sdxl_worker.py")
        repo_root = Path(__file__).resolve().parents[2]
        timeout_seconds = max(30.0, float(os.getenv("WEBBDUCK_SDXL_TIMEOUT_SECONDS", "3600")))

        with tempfile.TemporaryDirectory(prefix="webbduck_sdxl_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            progress_path = tmp / "progress.json"
            log_path = tmp / "worker.log"
            request_path.write_text(
                json.dumps({"action": "generate", "settings": settings}),
                encoding="utf-8",
            )

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
                    cwd=str(repo_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                started = time.monotonic()
                last_progress_mtime = -1
                while proc.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise GenerationCancelledError("SDXL generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(
                            f"SDXL runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    try:
                        progress_mtime = progress_path.stat().st_mtime_ns
                    except OSError:
                        progress_mtime = -1
                    if progress_mtime != last_progress_mtime:
                        last_progress_mtime = progress_mtime
                        _apply_progress_file(progress_path)
                    time.sleep(0.2)

            _apply_progress_file(progress_path)
            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
            if not result_path.exists():
                raise _worker_error({}, log_lines, proc.returncode)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise _worker_error(result, log_lines, proc.returncode)

            updates = result.get("metadata_updates")
            if isinstance(updates, dict):
                settings.update(updates)

            images: list[Image.Image] = []
            for raw_path in result.get("images") or []:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB").copy())
            if not images:
                raise RuntimeError("SDXL runtime returned no images")
            return images, int(result.get("seed", settings.get("seed") or 0))

    def tokenize(self, descriptor: ModelDescriptor, text: str, *, which: str = "prompt") -> dict[str, Any]:
        if not self.can_handle(descriptor):
            raise ValueError("Checkpoint is not handled by the SDXL backend")
        self.require_ready(descriptor)

        worker = Path(__file__).with_name("sdxl_worker.py")
        repo_root = Path(__file__).resolve().parents[2]
        timeout_seconds = max(10.0, float(os.getenv("WEBBDUCK_SDXL_TOKENIZE_TIMEOUT_SECONDS", "60")))
        with tempfile.TemporaryDirectory(prefix="webbduck_sdxl_tokenize_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            request_path.write_text(
                json.dumps(
                    {
                        "action": "tokenize",
                        "model_name": descriptor.name,
                        "text": str(text or ""),
                        "which": str(which or "prompt"),
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    _runtime_python(),
                    str(worker),
                    "--request",
                    str(request_path),
                    "--result",
                    str(result_path),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if not result_path.exists():
                detail = (completed.stdout or "") + "\n" + (completed.stderr or "")
                raise RuntimeError(
                    f"SDXL tokenizer runtime returned no result (code {completed.returncode}).\n{detail[-4000:]}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(str(result.get("error") or "SDXL tokenizer runtime failed"))
            tokens = int(result.get("tokens") or 0)
            limit = max(1, int(result.get("limit") or 77))
            return {
                "tokens": tokens,
                "limit": limit,
                "over": max(0, tokens - limit),
                "available": True,
            }

    def unload(self) -> None:
        # Workers are job-scoped processes. Their CUDA/model state disappears
        # when the process exits, so there is nothing resident in the host.
        return None


_backend = SDXLDiffusersBackend()


def ensure_registered() -> SDXLDiffusersBackend:
    """Idempotently register the installed SDXL backend."""
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
