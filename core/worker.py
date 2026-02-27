"""GPU worker for processing generation and upscale jobs."""

import asyncio
import gc
import logging
import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

from webbduck.core.generation import run_generation
from webbduck.core.exceptions import GenerationCancelledError
from webbduck.core.gpu_lease import release_gpu_lease, wait_for_gpu_lease_async
from webbduck.core.runtime import runtime_error_hint
from webbduck.server.storage import save_images, append_session_entry
from webbduck.server.events import broadcast_state
from webbduck.server.state import update_stage, update_progress, snapshot
from webbduck.models.upscaler import get_upsampler

log = logging.getLogger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "out of memory",
        "cuda out of memory",
        "cannot allocate memory",
        "oom",
    )
    return any(m in text for m in markers)


def _is_kernel_compat_error(exc: Exception) -> bool:
    return runtime_error_hint(exc) is not None


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            use_ipc_collect = str(os.getenv("WEBBDUCK_USE_IPC_COLLECT", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if use_ipc_collect:
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _prepare_gpu_for_webbduck_job() -> None:
    """Best-effort VRAM cleanup before WebbDuck generation starts.

    If DuckMotion is installed and previously loaded a Wan pipeline in-process,
    proactively unload it now that WebbDuck owns the GPU lease.
    """
    t0 = time.perf_counter()
    unloaded = False
    for module_name, module in list(sys.modules.items()):
        if module_name != "webbduck_webplugin_duckmotion":
            continue
        unload = getattr(module, "_unload_pipeline", None)
        if callable(unload):
            try:
                unload()
                unloaded = True
                log.info("Unloaded DuckMotion pipeline before WebbDuck job startup.")
            except Exception:
                log.exception("Failed to unload DuckMotion pipeline before WebbDuck job.")
        break

    _cleanup_memory()
    elapsed = time.perf_counter() - t0
    if elapsed > 2.0:
        log.warning("WebbDuck GPU preflight cleanup took %.2fs (duckmotion_unloaded=%s).", elapsed, unloaded)


async def _acquire_worker_gpu_lease(job):
    update_stage("Waiting for GPU")
    await broadcast_state(snapshot())
    timeout_raw = str(os.getenv("WEBBDUCK_GPU_LEASE_WAIT_SECONDS", "180")).strip()
    try:
        timeout_seconds = max(5.0, float(timeout_raw))
    except Exception:
        timeout_seconds = 180.0
    attempt = await wait_for_gpu_lease_async(
        owner="webbduck-core",
        owner_kind="webbduck",
        label=str(job.get("type") or "generation"),
        job_id=str(job.get("job_id") or "") or None,
        timeout_seconds=timeout_seconds,
    )
    if not attempt.get("acquired"):
        lease = attempt.get("lease") if isinstance(attempt, dict) else None
        holder = ""
        if isinstance(lease, dict):
            holder = str(lease.get("owner") or "").strip()
        holder_txt = f" Current holder: {holder}." if holder else ""
        waited = attempt.get("waited_seconds")
        waited_txt = f"{float(waited):.1f}s" if isinstance(waited, (int, float)) else f"{timeout_seconds:.1f}s"
        raise RuntimeError(
            "GPU lease wait timed out after "
            f"{waited_txt}.{holder_txt} "
            "If this persists, restart WebbDuck and verify no plugin job is still running."
        )
    lease = attempt.get("lease") if isinstance(attempt, dict) else None
    if isinstance(lease, dict):
        token = str(lease.get("token") or "").strip()
        return token or None
    return None


def _release_worker_gpu_lease(token):
    if not token:
        return
    try:
        release_gpu_lease(token=token)
    except Exception:
        pass


def _build_oom_message(exc: Exception) -> str:
    base = (
        "Generation failed due to memory exhaustion. "
        "Try reducing batch size/resolution, closing other GPU or RAM-heavy apps, "
        "or waiting for queued work to finish before retrying."
    )
    return f"{base} Original error: {exc}"


def _build_kernel_compat_message(exc: Exception) -> str:
    base = (
        "Generation failed due to GPU kernel compatibility. "
        "This usually means the installed PyTorch/CUDA build or selected dtype "
        "is incompatible with your GPU architecture. "
        "Try setting WEBBDUCK_DTYPE=float16 and installing a PyTorch wheel channel "
        "compatible with your GPU generation (for example stable cu124 or nightly cu126/cu128)."
    )
    hint = runtime_error_hint(exc)
    if hint:
        base = f"{base} {hint}"
    return f"{base} Original error: {exc}"


async def run_upscale(job, cancel_event=None):
    """Execute upscale task."""
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("Upscale cancelled before start")
    update_stage("Upscaling")
    update_progress(0.0)
    await broadcast_state(snapshot())

    scale = int(job.get("scale", 2))
    image_path = Path(job["image"])

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    upsampler = get_upsampler(scale)

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with torch.inference_mode():
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Upscale cancelled")
        upscaled_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)

    update_progress(0.9)
    await broadcast_state(snapshot())

    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(upscaled_rgb)

    out_path = image_path.with_name(f"{image_path.stem}_upscaled.png")
    out_img.save(out_path)

    update_progress(1.0)
    update_stage("Idle")
    await broadcast_state(snapshot())

    return str(out_path)


async def gpu_worker(queue):
    """Main GPU worker loop - processes generation and upscale jobs."""
    loop = asyncio.get_running_loop()

    while True:
        job = await queue.get()
        on_start = job.get("on_start")
        on_finish = job.get("on_finish")
        cancel_event = job.get("cancel_event")

        if callable(on_start):
            try:
                on_start(job)
            except Exception:
                pass
        
        if job["type"] == "upscale":
            lease_token = None
            try:
                lease_token = await _acquire_worker_gpu_lease(job)
                result = await run_upscale(job, cancel_event=cancel_event)
                job["future"].set_result({"image": result})
                if callable(on_finish):
                    try:
                        on_finish(job, True, None)
                    except Exception:
                        pass
            except Exception as e:
                _cleanup_memory()
                if isinstance(e, GenerationCancelledError):
                    update_stage("Cancelled")
                    update_progress(0.0)
                    await broadcast_state(snapshot())
                    if not job["future"].done():
                        job["future"].set_result({"status": "cancelled", "job_id": job.get("job_id")})
                    if callable(on_finish):
                        try:
                            on_finish(job, False, "__cancelled__")
                        except Exception:
                            pass
                    continue

                update_stage("Error")
                await broadcast_state(snapshot())
                if _is_kernel_compat_error(e):
                    msg = _build_kernel_compat_message(e)
                    update_stage("GPU Compatibility Error")
                    update_progress(0.0)
                    await broadcast_state(snapshot())
                    log.exception("GPU compatibility error during upscale job %s", job.get("job_id"))
                    job["future"].set_exception(RuntimeError(msg))
                elif _is_oom_error(e):
                    msg = _build_oom_message(e)
                    update_stage("OOM (Memory)")
                    update_progress(0.0)
                    await broadcast_state(snapshot())
                    log.exception("OOM during upscale job %s", job.get("job_id"))
                    job["future"].set_exception(RuntimeError(msg))
                else:
                    job["future"].set_exception(e)
                if callable(on_finish):
                    try:
                        on_finish(job, False, str(e))
                    except Exception:
                        pass
            finally:
                _release_worker_gpu_lease(lease_token)
                queue.task_done()
            continue

        lease_token = None
        try:
            lease_token = await _acquire_worker_gpu_lease(job)
            _prepare_gpu_for_webbduck_job()
            update_stage("Preparing")
            update_progress(0.02)
            await broadcast_state(snapshot())
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelledError("Generation cancelled before denoising")

            gen_started_monotonic = time.perf_counter()
            gen_started_utc = datetime.utcnow().isoformat() + "Z"
            images, seed = await loop.run_in_executor(
                None, run_generation, job["settings"], cancel_event
            )
            gen_finished_monotonic = time.perf_counter()
            gen_finished_utc = datetime.utcnow().isoformat() + "Z"

            # Ensure actual seed is saved in metadata
            job["settings"]["seed"] = seed
            job["settings"]["generation_started_at"] = gen_started_utc
            job["settings"]["generation_finished_at"] = gen_finished_utc
            job["settings"]["generation_elapsed_seconds"] = round(
                max(0.0, gen_finished_monotonic - gen_started_monotonic),
                3,
            )
            num_images = max(1, int(job["settings"].get("num_images", 1)))
            job["settings"]["generation_elapsed_seconds_per_image"] = round(
                float(job["settings"]["generation_elapsed_seconds"]) / float(num_images),
                3,
            )
            perf = job["settings"].get("performance_timing")
            if isinstance(perf, dict) and perf:
                per_image = {}
                for key, val in perf.items():
                    try:
                        per_image[str(key)] = round(float(val) / float(num_images), 6)
                    except Exception:
                        continue
                if per_image:
                    job["settings"]["performance_timing_per_image"] = per_image

            update_stage("Decoding")
            update_progress(0.98)
            await broadcast_state(snapshot())

            update_stage("Saving")
            update_progress(0.995)
            await broadcast_state(snapshot())

            paths = save_images(images, job["settings"])

            update_progress(1.0)
            update_stage("Idle")
            await broadcast_state(snapshot())

            # Sanitize settings for session log
            log_settings = job["settings"].copy()
            if "input_image" in log_settings:
                del log_settings["input_image"]
            if "mask_image" in log_settings:
                del log_settings["mask_image"]

            append_session_entry({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": job["type"],
                "seed": seed,
                "settings": log_settings,
                "images": paths,
            })

            job["future"].set_result({
                "seed": seed,
                "images": paths,
            })
            if callable(on_finish):
                try:
                    on_finish(job, True, None)
                except Exception:
                    pass

        except Exception as e:
            _cleanup_memory()
            if isinstance(e, GenerationCancelledError):
                update_stage("Cancelled")
                update_progress(0.0)
                await broadcast_state(snapshot())
                if not job["future"].done():
                    job["future"].set_result({"status": "cancelled", "job_id": job.get("job_id")})
                if callable(on_finish):
                    try:
                        on_finish(job, False, "__cancelled__")
                    except Exception:
                        pass
                continue

            update_stage("Error")
            await broadcast_state(snapshot())
            if _is_kernel_compat_error(e):
                msg = _build_kernel_compat_message(e)
                update_stage("GPU Compatibility Error")
                update_progress(0.0)
                await broadcast_state(snapshot())
                log.exception("GPU compatibility error during generation job %s", job.get("job_id"))
                job["future"].set_exception(RuntimeError(msg))
            elif _is_oom_error(e):
                msg = _build_oom_message(e)
                update_stage("OOM (Memory)")
                update_progress(0.0)
                await broadcast_state(snapshot())
                log.exception("OOM during generation job %s", job.get("job_id"))
                job["future"].set_exception(RuntimeError(msg))
            else:
                job["future"].set_exception(e)
            if callable(on_finish):
                try:
                    on_finish(job, False, str(e))
                except Exception:
                    pass

        finally:
            _release_worker_gpu_lease(lease_token)
            queue.task_done()
