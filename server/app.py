"""FastAPI application and endpoints."""

import asyncio
import json
import uuid
import time
import os
import random
import threading
from pathlib import Path
import shutil
from fastapi import FastAPI, Form, HTTPException, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webbduck.server.state import snapshot, update_stage, update_progress
from webbduck.server.events import broadcast, broadcast_state, active_sockets
from webbduck.server.storage import (
    save_images,
    BASE,
    to_web_path,
    resolve_web_path,
    ensure_manifest,
    search_manifest_sessions,
    filter_manifest_sessions,
    remove_manifest_image,
    remove_manifest_run,
    favorite_map,
    set_favorite,
    remove_favorite_image,
    remove_favorite_run,
)
from webbduck.server.thumbnails import ensure_thumbnail, get_thumbnail_path
from webbduck.core.worker import gpu_worker
from webbduck.models.registry import (
    MODEL_REGISTRY,
    LORA_REGISTRY,
    EMBEDDING_REGISTRY,
    CHECKPOINT_ROOT,
    LORA_ROOT,
    LORA_FILE,
    EMBEDDING_ROOT,
    EMBEDDING_FILE,
    MODELS_FILE,
    refresh_registries,
)
from webbduck.core.schedulers import SCHEDULERS
from webbduck.core.captioner import (
    list_captioners,
    get_caption_styles,
    is_captioning_available,
    generate_caption,
)
from webbduck.core.web_plugins import (
    build_public_web_plugin_list,
    connect_remote_web_plugin,
    discover_web_plugins,
    disconnect_remote_web_plugin,
    include_web_plugin_routers,
    list_remote_web_plugins,
    mount_web_plugin_assets,
)
from webbduck.core.gpu_lease import get_gpu_lease

INPUTS_DIR = Path("inpaint_input")
INPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI()
THUMB_CONCURRENCY = max(1, int(os.getenv("WEBBDUCK_THUMB_CONCURRENCY", "2")))
thumb_semaphore = asyncio.Semaphore(THUMB_CONCURRENCY)

WEB_PLUGINS = discover_web_plugins()
mount_web_plugin_assets(app, WEB_PLUGINS)
include_web_plugin_routers(app, WEB_PLUGINS)

# Queue for GPU jobs
generation_queue = asyncio.Queue(maxsize=32)
job_registry = {}
active_job_id = None
active_job = None
CATALOG_POLL_SECONDS = max(1.0, float(os.getenv("WEBBDUCK_CATALOG_POLL_SECONDS", "3.0")))
SMART_EXTEND_FEATHER_DEFAULT = 12
SMART_EXTEND_STEP_GROWTH_DEFAULT = 1.25
SMART_EXTEND_AUTO_STEP_DEFAULT = False
SMART_EXTEND_REFINE_DEFAULT = True
SMART_EXTEND_REFINE_EACH_STEP_DEFAULT = True
SMART_EXTEND_REFINE_WIDTH_DEFAULT = 64
SMART_EXTEND_REFINE_STRENGTH_DEFAULT = 0.32
SMART_EXTEND_PYRAMID_TRIGGER_RATIO_DEFAULT = 2.4


class RemotePluginConnectRequest(BaseModel):
    base_url: str


class DeleteImagesRequest(BaseModel):
    paths: list[str]


def _resolve_smart_extend_settings(
    *,
    advanced: bool,
    feather: int,
    auto_step: bool,
    step_growth: float,
    refine: bool,
    refine_width: int,
    refine_strength: float,
    refine_each_step: bool,
    pyramid_trigger_ratio: float,
) -> dict:
    """Return effective smart-extend settings.

    When advanced is disabled, server-enforced defaults are used.
    """
    if not advanced:
        return {
            "smart_extend_advanced": False,
            "smart_extend_feather": SMART_EXTEND_FEATHER_DEFAULT,
            "smart_extend_auto_step": SMART_EXTEND_AUTO_STEP_DEFAULT,
            "smart_extend_step_growth": SMART_EXTEND_STEP_GROWTH_DEFAULT,
            "smart_extend_refine": SMART_EXTEND_REFINE_DEFAULT,
            "smart_extend_refine_width": SMART_EXTEND_REFINE_WIDTH_DEFAULT,
            "smart_extend_refine_strength": SMART_EXTEND_REFINE_STRENGTH_DEFAULT,
            "smart_extend_refine_each_step": SMART_EXTEND_REFINE_EACH_STEP_DEFAULT,
            "smart_extend_pyramid_trigger_ratio": SMART_EXTEND_PYRAMID_TRIGGER_RATIO_DEFAULT,
        }

    return {
        "smart_extend_advanced": True,
        "smart_extend_feather": max(0, int(feather)),
        "smart_extend_auto_step": bool(auto_step),
        "smart_extend_step_growth": max(1.01, float(step_growth)),
        "smart_extend_refine": bool(refine),
        "smart_extend_refine_width": max(1, int(refine_width)),
        "smart_extend_refine_strength": max(0.0, float(refine_strength)),
        "smart_extend_refine_each_step": bool(refine_each_step),
        "smart_extend_pyramid_trigger_ratio": max(1.0, float(pyramid_trigger_ratio)),
    }


def _round_to_8(value, fallback=1024) -> int:
    """Normalize dimensions to nearest multiple of 8."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = int(fallback)
    return max(8, int(round(float(n) / 8.0) * 8))


def normalize_dimensions(width, height) -> tuple[int, int]:
    """Return width/height normalized to multiples of 8."""
    return _round_to_8(width, fallback=1024), _round_to_8(height, fallback=1024)


def summarize_loras(loras) -> list[str]:
    """Create compact LoRA labels for queue metadata."""
    if not isinstance(loras, list):
        return []

    labels = []
    for item in loras:
        if isinstance(item, str):
            labels.append(item)
            continue

        if not isinstance(item, dict):
            continue

        name = item.get("name") or item.get("model")
        if not name:
            continue

        weight_raw = item.get("weight", item.get("strength"))
        if weight_raw is None:
            labels.append(str(name))
            continue

        try:
            weight = float(weight_raw)
            labels.append(f"{name} ({weight:.2f})")
        except (TypeError, ValueError):
            labels.append(str(name))

    return labels[:8]


def summarize_embeddings(embeddings) -> list[str]:
    """Create compact embedding labels for queue metadata."""
    if not isinstance(embeddings, list):
        return []

    labels = []
    for item in embeddings:
        if isinstance(item, str):
            labels.append(item)
            continue
        if not isinstance(item, dict):
            continue

        name = item.get("name") or item.get("model")
        if not name:
            continue
        token = item.get("token")
        labels.append(f"{name} [{token}]" if token else str(name))
    return labels[:8]


def _parse_json_list(raw: str, field_name: str) -> list:
    """Parse list-like JSON form values safely."""
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON for '{field_name}'") from exc

    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail=f"'{field_name}' must be a JSON array")
    return value


def summarize_settings(settings: dict) -> dict:
    """Build a compact metadata summary for queue UI."""
    prompt = settings.get("prompt", "") or ""
    negative = settings.get("negative_prompt", "") or ""
    image_path = settings.get("image") or settings.get("input_image")
    mask_path = settings.get("mask_image")

    input_image_url = None
    if image_path:
        image_name = Path(str(image_path)).name
        if image_name:
            input_image_url = f"/inputs/{image_name}"

    mode = "txt2img"
    if image_path and mask_path:
        mode = "inpaint"
    elif image_path:
        mode = "img2img"

    mode_details = []
    strength = settings.get("strength")
    if strength is not None:
        try:
            mode_details.append(f"denoise {float(strength):.2f}")
        except (TypeError, ValueError):
            pass

    mask_blur = settings.get("mask_blur")
    if mask_blur is not None:
        mode_details.append(f"mask blur {mask_blur}")

    inpaint_fill = settings.get("inpainting_fill")
    if inpaint_fill:
        mode_details.append(f"fill {inpaint_fill}")

    if settings.get("smart_extend"):
        mode_details.append("smart extend")
        if settings.get("smart_extend_advanced"):
            mode_details.append("advanced controls")
        feather = settings.get("smart_extend_feather")
        if feather is not None:
            mode_details.append(f"feather {feather}")
        auto_step = settings.get("smart_extend_auto_step")
        if auto_step is not None:
            mode_details.append("auto-step on" if auto_step else "auto-step off")
        if bool(auto_step):
            step_growth = settings.get("smart_extend_step_growth")
            if step_growth is not None:
                try:
                    mode_details.append(f"growth {float(step_growth):.2f}x")
                except (TypeError, ValueError):
                    pass
        refine = settings.get("smart_extend_refine")
        if refine is not None:
            mode_details.append("seam refine on" if refine else "seam refine off")
        refine_each_step = settings.get("smart_extend_refine_each_step")
        if refine_each_step is not None:
            mode_details.append("refine/step on" if refine_each_step else "refine/step off")
        refine_width = settings.get("smart_extend_refine_width")
        if refine_width is not None:
            mode_details.append(f"seam width {refine_width}")
        refine_strength = settings.get("smart_extend_refine_strength")
        if refine_strength is not None:
            try:
                mode_details.append(f"seam denoise {float(refine_strength):.2f}")
            except (TypeError, ValueError):
                pass
        offset_x = settings.get("smart_extend_offset_x")
        offset_y = settings.get("smart_extend_offset_y")
        if offset_x is not None or offset_y is not None:
            mode_details.append(f"offset x {offset_x or 0}, y {offset_y or 0}")
        pyramid_enabled = settings.get("smart_extend_pyramid_enable")
        if pyramid_enabled is not None:
            mode_details.append("pyramid on" if pyramid_enabled else "pyramid off")
        pyramid_ratio = settings.get("smart_extend_pyramid_trigger_ratio")
        if pyramid_ratio is not None:
            try:
                mode_details.append(f"pyr ratio {float(pyramid_ratio):.2f}x")
            except (TypeError, ValueError):
                pass
        repeat_seed_initializer = settings.get("smart_extend_repeat_seed_initializer")
        if repeat_seed_initializer is not None:
            mode_details.append(f"repeat seed {repeat_seed_initializer}")
        pyramid_initializer = settings.get("smart_extend_pyramid_initializer")
        if pyramid_initializer is not None:
            mode_details.append(f"pyr init {pyramid_initializer}")

    return {
        "prompt": prompt[:160],
        "base_model": settings.get("base_model"),
        "scheduler": settings.get("scheduler"),
        "steps": settings.get("steps"),
        "cfg": settings.get("cfg"),
        "width": settings.get("width"),
        "height": settings.get("height"),
        "num_images": settings.get("num_images"),
        "seed": settings.get("seed"),
        "negative_prompt": negative[:220],
        "loras": summarize_loras(settings.get("loras", [])),
        "embeddings": summarize_embeddings(settings.get("embeddings", [])),
        "mode": mode,
        "has_input_image": bool(settings.get("image") or settings.get("input_image")),
        "has_mask": bool(settings.get("mask_image")),
        "input_image_url": input_image_url,
        "mode_details": mode_details[:14],
    }


def resolve_seed(seed) -> int:
    """Resolve optional seed to a concrete integer for queue visibility/repro."""
    if seed is None:
        return random.randint(0, 2**32 - 1)
    try:
        return int(seed)
    except (TypeError, ValueError):
        return random.randint(0, 2**32 - 1)


def build_queue_payload() -> dict:
    """Build queue payload for API and WebSocket updates."""
    jobs = []
    recent_completed = []
    queued_positions = {}
    for idx, job in enumerate(list(generation_queue._queue), start=1):
        jid = job.get("job_id")
        if jid:
            queued_positions[jid] = idx

    for meta in sorted(job_registry.values(), key=lambda x: x.get("created_at", 0), reverse=True):
        status = meta.get("status")
        if status == "completed":
            item = dict(meta)
            item["queue_position"] = None
            recent_completed.append(item)
            continue
        if status in {"failed", "cancelled"}:
            continue
        item = dict(meta)
        item["queue_position"] = queued_positions.get(item["job_id"])
        jobs.append(item)

    return {
        "active_job_id": active_job_id,
        "queued_count": generation_queue.qsize(),
        "gpu_lease": get_gpu_lease(),
        "jobs": jobs[:100],
        "recent_completed": recent_completed[:50],
    }


async def broadcast_queue_update():
    """Push queue update to connected WebSocket clients."""
    await broadcast({
        "type": "queue",
        "payload": build_queue_payload(),
    })


def schedule_queue_update():
    """Schedule queue update broadcast when in an async loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(broadcast_queue_update())


def _mark_job_start(job):
    global active_job_id, active_job
    job_id = job.get("job_id")
    if not job_id:
        return
    active_job_id = job_id
    active_job = job
    meta = job_registry.get(job_id)
    if meta:
        meta["status"] = "running"
        meta["started_at"] = time.time()
    schedule_queue_update()


def _mark_job_finish(job, success: bool, error: str | None):
    global active_job_id, active_job
    job_id = job.get("job_id")
    if not job_id:
        return
    meta = job_registry.get(job_id)
    if meta:
        if error == "__cancelled__":
            meta["status"] = "cancelled"
        else:
            meta["status"] = "completed" if success else "failed"
        meta["finished_at"] = time.time()
        if error and error != "__cancelled__":
            meta["error"] = error
        if success:
            fut = job.get("future")
            if fut and fut.done():
                try:
                    res = fut.result()
                    if isinstance(res, dict):
                        compact = {}
                        if "seed" in res:
                            compact["seed"] = res["seed"]
                        if "images" in res and isinstance(res["images"], list):
                            compact["images"] = res["images"][:8]
                        if "image" in res:
                            compact["image"] = res["image"]
                        if compact:
                            meta["result"] = compact
                except Exception:
                    pass
    if active_job_id == job_id:
        active_job_id = None
        active_job = None

    # Prevent unbounded growth.
    if len(job_registry) > 300:
        old_keys = sorted(
            job_registry.keys(),
            key=lambda k: job_registry[k].get("created_at", 0)
        )[:100]
        for key in old_keys:
            job_registry.pop(key, None)
    schedule_queue_update()


def queue_position_for(job_id: str) -> int | None:
    for idx, queued in enumerate(list(generation_queue._queue), start=1):
        if queued.get("job_id") == job_id:
            return idx
    return None


async def enqueue(job, wait_for_result: bool = True):
    """Enqueue job. Optionally wait for result."""
    if "cancel_event" not in job or job.get("cancel_event") is None:
        job["cancel_event"] = threading.Event()
    await generation_queue.put(job)
    job_id = job["job_id"]
    meta = job_registry[job_id]
    meta["status"] = "queued"
    meta["queued_at"] = time.time()
    await broadcast_queue_update()

    if not wait_for_result:
        return {
            "status": "queued",
            "job_id": job_id,
            "queue_position": queue_position_for(job_id),
        }

    return await job["future"]


async def vram_sampler():
    """Periodically broadcast VRAM stats."""
    while True:
        await asyncio.sleep(0.5)
        await broadcast_state(snapshot())


def _path_stamp(path: Path):
    if not path.exists():
        return ("missing", str(path))

    stat = path.stat()
    stamp = [("self", path.name, stat.st_mtime_ns, stat.st_size)]

    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            try:
                cstat = child.stat()
                stamp.append((child.name, cstat.st_mtime_ns, cstat.st_size, child.is_dir()))
            except Exception:
                stamp.append((child.name, "err"))

    return tuple(stamp)


def compute_catalog_signature():
    """Cheap signature for watched catalog folders/files."""
    return (
        _path_stamp(CHECKPOINT_ROOT),
        _path_stamp(LORA_ROOT),
        _path_stamp(LORA_FILE),
        _path_stamp(EMBEDDING_ROOT),
        _path_stamp(EMBEDDING_FILE),
        _path_stamp(MODELS_FILE),
    )


async def catalog_watcher():
    """Watch model/LoRA folders and hot-refresh registries."""
    last_signature = compute_catalog_signature()
    while True:
        await asyncio.sleep(CATALOG_POLL_SECONDS)
        current_signature = compute_catalog_signature()
        if current_signature == last_signature:
            continue
        last_signature = current_signature

        try:
            changed = refresh_registries()
            if not changed:
                continue
            await broadcast({
                "type": "catalog",
                "payload": {
                    "models": len(MODEL_REGISTRY),
                    "loras": len(LORA_REGISTRY),
                    "embeddings": len(EMBEDDING_REGISTRY),
                },
            })
        except Exception as exc:
            print(f"[Catalog Watcher] refresh failed: {exc}")


def validate_second_pass(settings):
    """Validate second pass model configuration."""
    second_pass = settings.get("second_pass_model")
    mode = settings.get("second_pass_mode", "auto")

    if not second_pass or second_pass == "None":
        return

    entry = MODEL_REGISTRY.get(second_pass)
    if not entry:
        raise ValueError(f"Unknown second-pass model: {second_pass}")

    if entry.get("arch") != "sdxl":
        raise ValueError(
            f"Second-pass model '{second_pass}' is not SDXL-compatible"
        )

    if mode == "auto":
        return

    if mode == "refiner" and "refiner" not in second_pass.lower():
        raise ValueError(
            "Second Pass Mode is set to 'Refiner', "
            "but the selected model does not appear to be a refiner.\n\n"
            "Switch to 'Img2Img' or 'Auto'."
        )


@app.on_event("startup")
async def startup():
    """Start background tasks."""
    asyncio.create_task(gpu_worker(generation_queue))
    asyncio.create_task(vram_sampler())
    asyncio.create_task(catalog_watcher())
    # Build search manifest lazily in background if missing.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, ensure_manifest)


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve main UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    return ui_path.read_text()


@app.get("/docs/simple-guide")
def simple_guide():
    """Serve the user-friendly guide markdown used by Studio help."""
    guide_path = Path(__file__).parent.parent / "docs" / "SIMPLE_GUIDE.md"
    if not guide_path.exists():
        return {"markdown": ""}
    return {"markdown": guide_path.read_text(encoding="utf-8")}


@app.get("/plugins/web")
def list_web_plugins():
    """List discovered web plugins for dynamic UI tab injection."""
    return {
        "plugins": build_public_web_plugin_list(WEB_PLUGINS),
    }


@app.get("/plugins/web/remote")
def list_remote_plugins():
    """List remote plugins connected by URL."""
    return {
        "plugins": list_remote_web_plugins(),
    }


@app.post("/plugins/web/remote/connect")
def connect_remote_plugin(payload: RemotePluginConnectRequest):
    """Connect a remote plugin by base URL (for example 127.0.0.1:8020)."""
    local_ids = {plugin.plugin_id for plugin in WEB_PLUGINS}
    try:
        plugin = connect_remote_web_plugin(payload.base_url, reserved_ids=local_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to connect remote plugin: {exc}") from exc
    return {"plugin": plugin}


@app.delete("/plugins/web/remote/{plugin_id}")
def disconnect_remote_plugin_route(plugin_id: str):
    """Disconnect and remove a previously connected remote plugin."""
    removed = disconnect_remote_web_plugin(plugin_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Remote plugin not found.")
    return {"removed": True, "id": str(plugin_id).strip().lower()}


app.mount("/ui", StaticFiles(directory=str(Path(__file__).parent.parent / "ui")), name="ui")

# Mount dynamic output directory
from webbduck.server.storage import BASE
app.mount("/outputs", StaticFiles(directory=str(BASE)), name="outputs")
app.mount("/inputs", StaticFiles(directory=str(INPUTS_DIR)), name="inputs")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket connection for live updates."""
    await ws.accept()
    await ws.send_text(json.dumps({
        "type": "queue",
        "payload": build_queue_payload(),
    }))
    active_sockets.add(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        active_sockets.discard(ws)


@app.post("/test")
async def test(
    prompt: str = Form(""),
    prompt_2: str = Form(""),
    negative_prompt: str = Form(""),
    steps: int = Form(30),
    cfg: float = Form(7.5),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(None),
    scheduler: str = Form("UniPC"),
    base_model: str = Form(...),
    second_pass_model: str = Form("None"),
    second_pass_mode: str = Form("auto"),
    loras: str = Form("[]"),
    embeddings: str = Form("[]"),
    strength: float = Form(0.75),
    image: UploadFile = File(None),
    mask: UploadFile = File(None),
    inpainting_fill: str = Form("replace"),
    mask_blur: int = Form(8),
    smart_extend: bool = Form(False),
    smart_extend_advanced: bool = Form(False),
    smart_extend_anchor: str = Form("center"),
    smart_extend_feather: int = Form(SMART_EXTEND_FEATHER_DEFAULT),
    smart_extend_auto_step: bool = Form(SMART_EXTEND_AUTO_STEP_DEFAULT),
    smart_extend_step_growth: float = Form(SMART_EXTEND_STEP_GROWTH_DEFAULT),
    smart_extend_refine: bool = Form(SMART_EXTEND_REFINE_DEFAULT),
    smart_extend_refine_width: int = Form(SMART_EXTEND_REFINE_WIDTH_DEFAULT),
    smart_extend_refine_strength: float = Form(SMART_EXTEND_REFINE_STRENGTH_DEFAULT),
    smart_extend_refine_each_step: bool = Form(SMART_EXTEND_REFINE_EACH_STEP_DEFAULT),
    smart_extend_offset_x: int = Form(None),
    smart_extend_offset_y: int = Form(None),
    smart_extend_pyramid_enable: bool = Form(False),
    smart_extend_pyramid_trigger_ratio: float = Form(SMART_EXTEND_PYRAMID_TRIGGER_RATIO_DEFAULT),
    clip_skip: int = Form(None),
    wait_for_result: bool = Form(True),
):
    """Generate single test image."""
    width, height = normalize_dimensions(width, height)
    lora_list = _parse_json_list(loras, "loras")
    embedding_list = _parse_json_list(embeddings, "embeddings")
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    settings = {
        "base_model": base_model,
        "second_pass_model": second_pass_model,
        "second_pass_mode": second_pass_mode,
        "prompt": prompt,
        "prompt_2": prompt_2,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "num_images": 1,
        "seed": resolve_seed(seed),
        "scheduler": scheduler,
        "loras": lora_list,
        "embeddings": embedding_list,
        "strength": strength,
        "inpainting_fill": inpainting_fill,
        "mask_blur": mask_blur,
        "smart_extend": smart_extend,
        "smart_extend_anchor": smart_extend_anchor,
        "smart_extend_offset_x": smart_extend_offset_x,
        "smart_extend_offset_y": smart_extend_offset_y,
        "smart_extend_pyramid_enable": smart_extend_pyramid_enable,
        "clip_skip": clip_skip,
    }
    settings.update(
        _resolve_smart_extend_settings(
            advanced=smart_extend_advanced,
            feather=smart_extend_feather,
            auto_step=smart_extend_auto_step,
            step_growth=smart_extend_step_growth,
            refine=smart_extend_refine,
            refine_width=smart_extend_refine_width,
            refine_strength=smart_extend_refine_strength,
            refine_each_step=smart_extend_refine_each_step,
            pyramid_trigger_ratio=smart_extend_pyramid_trigger_ratio,
        )
    )

    if image:
        ext = Path(image.filename).suffix
        if not ext:
            ext = ".png"
        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = INPUTS_DIR / unique_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        settings["image"] = str(file_path.absolute())

    if mask:
        unique_mask_name = f"{uuid.uuid4()}_mask.png"
        mask_path = INPUTS_DIR / unique_mask_name
        with open(mask_path, "wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)
        settings["mask_image"] = str(mask_path.absolute())

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": "test",
        "settings": settings,
        "future": future,
        "cancel_event": threading.Event(),
        "on_start": _mark_job_start,
        "on_finish": _mark_job_finish,
    }
    job_registry[job_id] = {
        "job_id": job_id,
        "type": "test",
        "status": "created",
        "created_at": time.time(),
        "settings": summarize_settings(settings),
    }

    try:
        validate_second_pass(settings)
    except ValueError as e:
        return {
            "error": str(e),
            "type": "validation_error"
        }

    return await enqueue(job, wait_for_result=wait_for_result)


@app.post("/generate")
async def generate(
    prompt: str = Form(""),
    prompt_2: str = Form(""),
    negative_prompt: str = Form(""),
    steps: int = Form(30),
    cfg: float = Form(7.5),
    width: int = Form(1024),
    height: int = Form(1024),
    num_images: int = Form(4),
    seed: int = Form(None),
    base_model: str = Form(...),
    second_pass_model: str = Form("None"),
    second_pass_mode: str = Form("auto"),
    loras: str = Form("[]"),
    embeddings: str = Form("[]"),
    scheduler: str = Form("UniPC"),
    strength: float = Form(0.75),
    refinement_strength: float = Form(0.3),
    image: UploadFile = File(None),
    mask: UploadFile = File(None),
    inpainting_fill: str = Form("replace"),
    mask_blur: int = Form(8),
    smart_extend: bool = Form(False),
    smart_extend_advanced: bool = Form(False),
    smart_extend_anchor: str = Form("center"),
    smart_extend_feather: int = Form(SMART_EXTEND_FEATHER_DEFAULT),
    smart_extend_auto_step: bool = Form(SMART_EXTEND_AUTO_STEP_DEFAULT),
    smart_extend_step_growth: float = Form(SMART_EXTEND_STEP_GROWTH_DEFAULT),
    smart_extend_refine: bool = Form(SMART_EXTEND_REFINE_DEFAULT),
    smart_extend_refine_width: int = Form(SMART_EXTEND_REFINE_WIDTH_DEFAULT),
    smart_extend_refine_strength: float = Form(SMART_EXTEND_REFINE_STRENGTH_DEFAULT),
    smart_extend_refine_each_step: bool = Form(SMART_EXTEND_REFINE_EACH_STEP_DEFAULT),
    smart_extend_offset_x: int = Form(None),
    smart_extend_offset_y: int = Form(None),
    smart_extend_repeat_chunked: bool = Form(True),
    smart_extend_repeat_passes: str = Form("auto"),
    smart_extend_repeat_seed_initializer: str = Form("none"),
    smart_extend_pyramid_enable: bool = Form(False),
    smart_extend_pyramid_trigger_ratio: float = Form(SMART_EXTEND_PYRAMID_TRIGGER_RATIO_DEFAULT),
    smart_extend_pyramid_initializer: str = Form("edge_strips"),
    clip_skip: int = Form(None),
    wait_for_result: bool = Form(True),
):
    """Generate batch of images."""
    width, height = normalize_dimensions(width, height)
    lora_list = _parse_json_list(loras, "loras")
    embedding_list = _parse_json_list(embeddings, "embeddings")
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    settings = {
        "base_model": base_model,
        "second_pass_model": second_pass_model,
        "second_pass_mode": second_pass_mode,
        "prompt": prompt,
        "prompt_2": prompt_2,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "num_images": num_images,
        "seed": resolve_seed(seed),
        "loras": lora_list,
        "embeddings": embedding_list,
        "scheduler": scheduler,
        "strength": strength,
        "refinement_strength": refinement_strength,
        "inpainting_fill": inpainting_fill,
        "mask_blur": mask_blur,
        "smart_extend": smart_extend,
        "smart_extend_anchor": smart_extend_anchor,
        "smart_extend_offset_x": smart_extend_offset_x,
        "smart_extend_offset_y": smart_extend_offset_y,
        "smart_extend_repeat_chunked": smart_extend_repeat_chunked,
        "smart_extend_repeat_passes": smart_extend_repeat_passes,
        "smart_extend_repeat_seed_initializer": smart_extend_repeat_seed_initializer,
        "smart_extend_pyramid_enable": smart_extend_pyramid_enable,
        "smart_extend_pyramid_initializer": smart_extend_pyramid_initializer,
        "clip_skip": clip_skip,
    }
    settings.update(
        _resolve_smart_extend_settings(
            advanced=smart_extend_advanced,
            feather=smart_extend_feather,
            auto_step=smart_extend_auto_step,
            step_growth=smart_extend_step_growth,
            refine=smart_extend_refine,
            refine_width=smart_extend_refine_width,
            refine_strength=smart_extend_refine_strength,
            refine_each_step=smart_extend_refine_each_step,
            pyramid_trigger_ratio=smart_extend_pyramid_trigger_ratio,
        )
    )

    if image:
        # Use UUID to prevent file locking issues on Windows
        ext = Path(image.filename).suffix
        if not ext:
            ext = ".png" # default
        unique_name = f"{uuid.uuid4()}{ext}"
        
        file_path = INPUTS_DIR / unique_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        settings["image"] = str(file_path.absolute())

    if mask:
        unique_mask_name = f"{uuid.uuid4()}_mask.png"
        mask_path = INPUTS_DIR / unique_mask_name
        with open(mask_path, "wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)
        settings["mask_image"] = str(mask_path.absolute())

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": "batch",
        "settings": settings,
        "future": future,
        "cancel_event": threading.Event(),
        "on_start": _mark_job_start,
        "on_finish": _mark_job_finish,
    }
    job_registry[job_id] = {
        "job_id": job_id,
        "type": "batch",
        "status": "created",
        "created_at": time.time(),
        "settings": summarize_settings(settings),
    }

    try:
        validate_second_pass(settings)
        return await enqueue(job, wait_for_result=wait_for_result)
    except ValueError as e:
        return {
            "error": str(e),
            "type": "validation_error"
        }
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(trace) # Ensure it shows in server log
        return JSONResponse(
            status_code=500,
            content={"error": f"Generation failed: {str(e)}", "trace": trace}
        )


def _unlink_file(path: Path) -> bool:
    """Delete a file if it exists. Return True when deleted."""
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def _cleanup_run_if_empty(run_dir: Path):
    """Delete run metadata when the last image has been removed."""
    if not run_dir.exists() or not run_dir.is_dir():
        return

    has_any_png = any(run_dir.glob("*.png"))
    if has_any_png:
        return

    _unlink_file(run_dir / "meta.json")
    remove_manifest_run(run_dir.name)
    remove_favorite_run(run_dir.name)


def _delete_image_artifacts(target: Path) -> list[str]:
    """Delete image + sidecars and return deleted web paths."""
    deleted_paths: list[str] = []

    if not target.is_file():
        return deleted_paths

    candidates = [target]
    if target.suffix.lower() == ".png" and not target.name.endswith("_upscaled.png"):
        candidates.append(target.with_name(f"{target.stem}_upscaled.png"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        if _unlink_file(candidate):
            candidate_web = to_web_path(candidate)
            deleted_paths.append(candidate_web)
            remove_manifest_image(candidate_web)
            remove_favorite_image(candidate_web)

        thumb = get_thumbnail_path(candidate)
        if _unlink_file(thumb):
            deleted_paths.append(to_web_path(thumb))

    _cleanup_run_if_empty(target.parent)
    return deleted_paths


@app.post("/delete_image")
async def delete_image(path: str = Form(...)):
    """Delete a single image and related artifacts."""
    try:
        target = resolve_web_path(path)
        if not target.is_file():
            return JSONResponse(status_code=400, content={"error": "Not a file"})

        deleted = _delete_image_artifacts(target)
        print(f"[Info] Deleted {target} ({len(deleted)} artifacts)")
        return {
            "status": "ok",
            "deleted_count": len(deleted),
            "deleted": deleted,
        }
    except Exception as e:
        print(f"[Error] Delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/delete_images")
async def delete_images(payload: DeleteImagesRequest):
    """Batch delete image files and related artifacts."""
    try:
        raw_paths = payload.paths if isinstance(payload.paths, list) else []
        requested: list[str] = []
        seen: set[str] = set()
        for raw in raw_paths:
            if not isinstance(raw, str):
                continue
            path = raw.strip()
            if not path or path in seen:
                continue
            seen.add(path)
            requested.append(path)

        deleted: list[str] = []
        failed: list[dict[str, str]] = []

        for path in requested:
            try:
                target = resolve_web_path(path)
                if not target.is_file():
                    failed.append({"path": path, "error": "Not a file"})
                    continue
                deleted.extend(_delete_image_artifacts(target))
            except Exception as exc:
                failed.append({"path": path, "error": str(exc)})

        return {
            "status": "ok",
            "requested_count": len(requested),
            "deleted_count": len(deleted),
            "failed_count": len(failed),
            "deleted": deleted,
            "failed": failed,
        }
    except Exception as e:
        print(f"[Error] Batch delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/delete_run")
async def delete_run(path: str = Form(...)):
    """Delete an entire generation run folder."""
    try:
        # Resolve the image path first
        target_file = resolve_web_path(path)
        
        # The run directory is the parent of the image file
        run_dir = target_file.parent
        
        # Security: Ensure we are deleting a subdirectory of BASE (outputs)
        # and not BASE itself or something outside.
        if run_dir == BASE or not BASE in run_dir.parents:
             return JSONResponse(status_code=400, content={"error": "Invalid run directory"})

        if not run_dir.exists() or not run_dir.is_dir():
             return JSONResponse(status_code=400, content={"error": "Run directory not found"})

        run_name = run_dir.name
        shutil.rmtree(run_dir)
        remove_manifest_run(run_name)
        remove_favorite_run(run_name)
        print(f"[Info] Deleted run {run_dir}")
        return {"status": "ok"}
    except Exception as e:
        print(f"[Error] Delete run failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/gallery")
def gallery(start: int = 0, limit: int = 50, after: float = 0.0):
    """List generated image runs with pagination."""
    runs = sorted(BASE.iterdir(), reverse=True)
    bounded_start = max(0, int(start))
    bounded_limit = max(1, min(int(limit), 500))

    out = []
    valid_seen = 0
    favorites = favorite_map()

    for r in runs:
        if not r.is_dir():
            continue
        meta_file = r / "meta.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)

            # Fallback for old runs
            if "timestamp" not in meta:
                try:
                    import time
                    from datetime import datetime
                    dt = datetime.strptime(r.name, "%Y-%m-%d_%H-%M-%S")
                    meta["timestamp"] = dt.timestamp()
                except Exception:
                    pass

            # Legacy 'after' filter (optional, mostly for polling)
            if after > 0 and meta.get("timestamp", 0) <= after:
                continue

        except Exception as e:
            print(f"[Error] Failed to load meta for {r.name}: {e}")
            continue

        imgs = []
        variants = {}

        for p in r.glob("*.png"):
            if p.name.endswith("_upscaled.png"):
                original_stem = p.name.replace("_upscaled.png", "")
                original_name = f"{original_stem}.png"
                variants[original_name] = to_web_path(p)
            else:
                imgs.append(to_web_path(p))
        
        def sort_key(x):
            stem = Path(x).stem
            if stem.isdigit():
                return (0, int(stem), "")
            return (1, 0, stem)

        imgs.sort(key=sort_key)
        if not imgs:
            continue

        if valid_seen < bounded_start:
            valid_seen += 1
            continue

        out.append({
            "run": r.name,
            "images": imgs,
            "variants": variants,
            "favorites": {Path(p).name: True for p in imgs if favorites.get(p)},
            "meta": meta
        })
        valid_seen += 1
        if len(out) >= bounded_limit:
            break

    return out


@app.get("/gallery/search")
def gallery_search(q: str, start: int = 0, limit: int = 1000):
    """Search gallery globally via manifest index."""
    q = (q or "").strip()
    if not q:
        return []
    bounded_limit = max(1, min(int(limit), 5000))
    bounded_start = max(0, int(start))
    return search_manifest_sessions(q, start=bounded_start, limit=bounded_limit)


@app.get("/gallery/filter")
def gallery_filter(kind: str, start: int = 0, limit: int = 2000):
    """Filter gallery sessions quickly using manifest index."""
    k = (kind or "").strip().lower()
    if k not in {"hd", "favorites"}:
        return []
    bounded_limit = max(1, min(int(limit), 5000))
    bounded_start = max(0, int(start))
    return filter_manifest_sessions(k, start=bounded_start, limit=bounded_limit)


@app.post("/favorite")
async def favorite_image(path: str = Form(...), favorite: bool = Form(True)):
    """Mark/unmark a generated image as favorite."""
    try:
        target = resolve_web_path(path)
        if not target.exists() or not target.is_file():
            return JSONResponse(status_code=404, content={"error": "Image not found"})
        web_path, fav = set_favorite(to_web_path(target), bool(favorite))
        return {"status": "ok", "path": web_path, "favorite": fav}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/models")
@app.get("/models")
def list_models():
    """List available models."""
    return [
        {
            "name": name,
            "type": info.get("type"),
            "defaults": info.get("defaults", {}),
        }
        for name, info in MODEL_REGISTRY.items()
    ]


@app.get("/second_pass_models")
def list_second_pass_models():
    """List available second pass models."""
    return list(MODEL_REGISTRY.keys())


@app.get("/schedulers")
def list_schedulers():
    """List available schedulers."""
    return list(SCHEDULERS.keys())


@app.get("/models/{base_model:path}/loras")
def list_model_loras(base_model: str):
    if base_model not in MODEL_REGISTRY:
        return []

    model_arch = MODEL_REGISTRY[base_model]["arch"]

    return [
        {
            "name": name,
            "description": cfg.get("description", ""),
            "weight": cfg.get("weight", 1.0),
        }
        for name, cfg in LORA_REGISTRY.items()
        if cfg["arch"] == model_arch
    ]


@app.get("/models/{base_model:path}/embeddings")
def list_model_embeddings(base_model: str):
    if base_model not in MODEL_REGISTRY:
        return []

    model_arch = MODEL_REGISTRY[base_model]["arch"]
    return [
        {
            "name": name,
            "description": cfg.get("description", ""),
            "token": cfg.get("token", name),
        }
        for name, cfg in EMBEDDING_REGISTRY.items()
        if cfg.get("arch") == model_arch
    ]


@app.post("/upscale")
async def upscale(
    image: str = Form(...),
    scale: int = Form(2),
    wait_for_result: bool = Form(True),
):
    """Upscale an image."""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": "upscale",
        "image": str(resolve_web_path(image)),
        "scale": scale,
        "future": future,
        "cancel_event": threading.Event(),
        "on_start": _mark_job_start,
        "on_finish": _mark_job_finish,
    }
    job_registry[job_id] = {
        "job_id": job_id,
        "type": "upscale",
        "status": "created",
        "created_at": time.time(),
        "settings": {"image": image, "scale": scale},
    }

    return await enqueue(job, wait_for_result=wait_for_result)


@app.get("/queue")
def get_queue():
    """List active queue jobs and recent completions."""
    return build_queue_payload()


@app.get("/gpu/lease")
def gpu_lease_status():
    """Inspect shared GPU lease state (WebbDuck + local plugins)."""
    return get_gpu_lease()


@app.get("/runtime/profile")
def runtime_profile():
    """Return resolved runtime device/dtype profile."""
    from webbduck.core.runtime import resolve_runtime_profile

    return {"runtime": resolve_runtime_profile().to_dict()}


@app.post("/queue/cancel")
async def cancel_queue_job(job_id: str = Form(...)):
    """Cancel a queued or running job."""
    meta = job_registry.get(job_id)
    if not meta:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    status = meta.get("status")

    if status in {"running", "cancelling"} and active_job_id == job_id and active_job:
        cancel_event = active_job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        meta["status"] = "cancelling"
        meta["cancel_requested_at"] = time.time()
        await broadcast_queue_update()
        return {"status": "cancelling", "job_id": job_id}

    for queued_job in list(generation_queue._queue):
        if queued_job.get("job_id") != job_id:
            continue

        cancel_event = queued_job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        generation_queue._queue.remove(queued_job)
        generation_queue.task_done()

        fut = queued_job.get("future")
        if fut and not fut.done():
            fut.cancel()

        meta["status"] = "cancelled"
        meta["finished_at"] = time.time()
        await broadcast_queue_update()
        return {"status": "cancelled", "job_id": job_id}

    if status in {"completed", "failed", "cancelled"}:
        return JSONResponse(status_code=409, content={"error": f"Job already {status}"})
    return JSONResponse(status_code=409, content={"error": "Job is no longer cancellable"})


@app.post("/models/unload_all")
async def unload_all_models():
    """Unload all loaded generation and captioning models from memory."""
    if active_job_id is not None:
        return JSONResponse(
            status_code=409,
            content={"error": "Cannot unload while a job is running"},
        )
    lease = get_gpu_lease()
    held = bool(lease.get("held"))
    lease_owner = ((lease.get("lease") or {}) if isinstance(lease, dict) else {}).get("owner")
    if held and lease_owner and str(lease_owner) != "webbduck-core":
        return JSONResponse(
            status_code=409,
            content={"error": f"Cannot unload while GPU is leased by {lease_owner}"},
        )

    from webbduck.core.pipeline import pipeline_manager, get_runtime_profile_dict
    from webbduck.core.captioner import unload_captioners

    update_stage("Unloading models")
    update_progress(0.05)
    await broadcast_state(snapshot())

    try:
        pipeline_manager.unload_all()
        unload_captioners()
    except Exception as exc:
        update_stage("Error")
        await broadcast_state(snapshot())
        return JSONResponse(status_code=500, content={"error": f"Unload failed: {exc}"})

    update_progress(1.0)
    update_stage("Idle")
    await broadcast_state(snapshot())
    return {"status": "unloaded"}


async def _delayed_process_exit(delay_seconds: float = 0.35):
    await asyncio.sleep(max(0.0, float(delay_seconds)))
    os._exit(0)


@app.post("/app/shutdown")
async def shutdown_app():
    """Terminate the WebbDuck process."""
    update_stage("Shutting down")
    update_progress(0.0)
    await broadcast_state(snapshot())
    asyncio.create_task(_delayed_process_exit())
    return {"status": "shutting_down"}


@app.get("/health")
def health():
    """System health check."""
    import torch
    from webbduck.core.pipeline import pipeline_manager, get_runtime_profile_dict
    from webbduck.server.state import snapshot

    cuda_ok = torch.cuda.is_available()
    vram = None

    if cuda_ok:
        vram = {
            "used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            ),
        }

    return {
        "status": "ok",
        "cuda_available": cuda_ok,
        "runtime": get_runtime_profile_dict(),
        "vram": vram,
        "models": {
            "count": len(MODEL_REGISTRY),
            "names": list(MODEL_REGISTRY.keys()),
        },
        "loras": {
            "count": len(LORA_REGISTRY),
            "names": list(LORA_REGISTRY.keys()),
        },
        "embeddings": {
            "count": len(EMBEDDING_REGISTRY),
            "names": list(EMBEDDING_REGISTRY.keys()),
        },
        "queue": {
            "size": generation_queue.qsize(),
            "maxsize": generation_queue.maxsize,
            "active_job_id": active_job_id,
            "tracked_jobs": len(job_registry),
            "gpu_lease": get_gpu_lease(),
        },
        "pipeline": {
            "loaded": pipeline_manager.pipe is not None,
            "base_model": pipeline_manager.key,
            "second_pass_model": pipeline_manager.current_second_pass_model,
            "loras": list(pipeline_manager.current_loras.keys()),
            "embeddings": list(pipeline_manager.current_embeddings.keys()),
        },
        "runtime_state": snapshot(),
    }


@app.post("/tokenize")
async def tokenize_prompt(
    text: str = Form(""),
    base_model: str = Form(...),
    which: str = Form("prompt"),
):
    """Count tokens in prompt."""
    from webbduck.core.pipeline import get_tokenizer
    from webbduck.models.registry import MODEL_REGISTRY
    
    # Get model path directly
    base_entry = MODEL_REGISTRY[base_model]
    base_path = base_entry["path"]
    
    # Use lightweight tokenizer loader
    tokenizer, tokenizer_2 = get_tokenizer(base_path)

    active_tokenizer = (
        tokenizer_2 if which == "prompt_2"
        else tokenizer
    )

    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]

    return {
        "tokens": len(tokens),
        "limit": 77,
        "over": max(0, len(tokens) - 77),
    }


@app.get("/captioners")
def get_captioners():
    """List available captioning plugins."""
    return {
        "available": is_captioning_available(),
        "captioners": list_captioners(),
    }


@app.get("/caption_styles")
def get_caption_styles_endpoint():
    """List available caption styles and their prompts."""
    return get_caption_styles()


@app.post("/caption")
async def caption_image(
    image: UploadFile = File(...),
    style: str = Form("sd_prompt"),
    captioner: str = Form(None),
    max_tokens: int = Form(300),
):
    """Generate a caption for the uploaded image.
    
    This will temporarily unload generation pipelines to free VRAM,
    then reload them after captioning is complete.
    """
    if not is_captioning_available():
        return JSONResponse(
            status_code=503,
            content={
                "error": "No captioner plugins available",
                "message": "Install a captioner plugin in ~/.webbduck/plugins/captioners/",
            }
        )
    
    # Save uploaded image temporarily
    ext = Path(image.filename).suffix if image.filename else ".png"
    unique_name = f"{uuid.uuid4()}_caption{ext}"
    file_path = INPUTS_DIR / unique_name
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Update UI status
        update_stage("Captioning")
        update_progress(0.3)
        await broadcast_state(snapshot())
        
        # Offload generation pipelines to free VRAM for captioning
        def offload_and_caption():
            # Lazy import to avoid circular imports and side effects
            import torch
            import gc
            
            # Only offload if there's actually a pipeline loaded on GPU
            try:
                from webbduck.core.pipeline import pipeline_manager
                if pipeline_manager.pipe is not None and hasattr(pipeline_manager.pipe, 'unet'):
                    # Check if UNet is actually on CUDA before offloading
                    unet_device = next(pipeline_manager.pipe.unet.parameters()).device
                    if unet_device.type == 'cuda':
                        pipeline_manager.pipe.unet.to('cpu')
                        pipeline_manager.pipe.vae.to('cpu')
                        torch.cuda.empty_cache()
                        gc.collect()
            except Exception:
                pass  # If anything fails, just continue with captioning
            
            # Now run captioning
            return generate_caption(
                image_path=file_path,
                style=style,
                captioner_name=captioner,
                max_tokens=max_tokens,
            )
        
        # Run captioning in executor to avoid blocking
        loop = asyncio.get_event_loop()
        caption = await loop.run_in_executor(None, offload_and_caption)
        
        update_stage("Idle")
        update_progress(1.0)
        await broadcast_state(snapshot())
        
        print(f"[Caption] Complete - returning result, no further loading should happen")
        return {"caption": caption, "style": style}
        
    except Exception as e:
        import traceback
        update_stage("Error")
        await broadcast_state(snapshot())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )
    finally:
        # Clean up temp file
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


@app.get("/thumbs/{path:path}")
async def get_thumbnail(path: str):
    """Serve a thumbnail, generating it on demand if needed."""
    try:
        async with thumb_semaphore:
            # Run resizing in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            thumb_path = await loop.run_in_executor(None, ensure_thumbnail, path)
        data = thumb_path.read_bytes()
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Thumbnail error: {e}"})
