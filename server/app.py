"""FastAPI application and endpoints."""

import asyncio
import json
import uuid
import time
import os
from pathlib import Path
import shutil
from fastapi import FastAPI, Form, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webbduck.server.state import snapshot, update_stage, update_progress
from webbduck.server.events import broadcast, broadcast_state, active_sockets
from webbduck.server.storage import save_images, BASE, to_web_path, resolve_web_path
from webbduck.server.thumbnails import ensure_thumbnail
from fastapi.responses import FileResponse
from webbduck.core.worker import gpu_worker
from webbduck.models.registry import MODEL_REGISTRY, LORA_REGISTRY
from webbduck.core.schedulers import SCHEDULERS
from webbduck.core.captioner import (
    list_captioners,
    get_caption_styles,
    is_captioning_available,
    generate_caption,
)

INPUTS_DIR = Path("inpaint_input")
INPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI()
THUMB_CONCURRENCY = max(1, int(os.getenv("WEBBDUCK_THUMB_CONCURRENCY", "2")))
thumb_semaphore = asyncio.Semaphore(THUMB_CONCURRENCY)

# Queue for GPU jobs
generation_queue = asyncio.Queue(maxsize=32)
job_registry = {}
active_job_id = None


def summarize_settings(settings: dict) -> dict:
    """Build a compact metadata summary for queue UI."""
    prompt = settings.get("prompt", "") or ""
    return {
        "prompt": prompt[:160],
        "base_model": settings.get("base_model"),
        "scheduler": settings.get("scheduler"),
        "steps": settings.get("steps"),
        "cfg": settings.get("cfg"),
        "width": settings.get("width"),
        "height": settings.get("height"),
        "num_images": settings.get("num_images"),
        "has_input_image": bool(settings.get("image") or settings.get("input_image")),
        "has_mask": bool(settings.get("mask_image")),
    }


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
    global active_job_id
    job_id = job.get("job_id")
    if not job_id:
        return
    active_job_id = job_id
    meta = job_registry.get(job_id)
    if meta:
        meta["status"] = "running"
        meta["started_at"] = time.time()
    schedule_queue_update()


def _mark_job_finish(job, success: bool, error: str | None):
    global active_job_id
    job_id = job.get("job_id")
    if not job_id:
        return
    meta = job_registry.get(job_id)
    if meta:
        meta["status"] = "completed" if success else "failed"
        meta["finished_at"] = time.time()
        if error:
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


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve main UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    return ui_path.read_text()


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
    base_model: str = Form(...),
    second_pass_model: str = Form("None"),
    second_pass_mode: str = Form("auto"),
    loras: str = Form("[]"),
    experimental_compress: bool = Form(False),
    wait_for_result: bool = Form(True),
):
    """Generate single test image."""
    lora_list = json.loads(loras)
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
        "loras": lora_list,
        "experimental_compress": experimental_compress,
    }

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": "test",
        "settings": settings,
        "future": future,
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

    experimental_compress: bool = Form(False),
    scheduler: str = Form("UniPC"),
    strength: float = Form(0.75),
    refinement_strength: float = Form(0.3),
    image: UploadFile = File(None),
    mask: UploadFile = File(None),
    inpainting_fill: str = Form("replace"),
    mask_blur: int = Form(8),
    wait_for_result: bool = Form(True),
):
    """Generate batch of images."""
    lora_list = json.loads(loras)
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
        "seed": seed,
        "loras": lora_list,
        "experimental_compress": experimental_compress,
        "scheduler": scheduler,
        "strength": strength,
        "refinement_strength": refinement_strength,
        "inpainting_fill": inpainting_fill,
        "mask_blur": mask_blur,
    }

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


@app.post("/delete_image")
async def delete_image(path: str = Form(...)):
    """Delete an image file."""
    try:
        # Security check: ensure path is within BASE
        target = resolve_web_path(path)
        
        # Double check it is actually a file
        if not target.is_file():
             return JSONResponse(status_code=400, content={"error": "Not a file"})

        target.unlink()
        print(f"[Info] Deleted {target}")
        return {"status": "ok"}
    except Exception as e:
        print(f"[Error] Delete failed: {e}")
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

        shutil.rmtree(run_dir)
        print(f"[Info] Deleted run {run_dir}")
        return {"status": "ok"}
    except Exception as e:
        print(f"[Error] Delete run failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/gallery")
def gallery(start: int = 0, limit: int = 50, after: float = 0.0):
    """List generated image runs with pagination."""
    # fast scan of directories
    runs = sorted(BASE.iterdir(), reverse=True)
    
    # Slice the list of folders to avoid processing everything
    # We slice slightly more than limit to account for potentially invalid folders
    # But for simplicity and speed, strict slicing is usually fine if cleanup is good
    if start >= len(runs):
        return []
        
    runs_slice = runs[start : start + limit + 10] # +10 buffer for non-runs
    
    out = []
    count = 0
    
    for r in runs_slice:
        if count >= limit:
            break
            
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

        out.append({
            "run": r.name, 
            "images": imgs, 
            "variants": variants,
            "meta": meta
        })
        count += 1
        
    return out


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


@app.get("/models/{base_model}/loras")
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


@app.post("/queue/cancel")
async def cancel_queue_job(job_id: str = Form(...)):
    """Cancel a queued (not yet running) job."""
    meta = job_registry.get(job_id)
    if not meta:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    if meta.get("status") == "running":
        return JSONResponse(
            status_code=409,
            content={"error": "Job already running; queued cancellation only"}
        )

    for queued_job in list(generation_queue._queue):
        if queued_job.get("job_id") != job_id:
            continue

        generation_queue._queue.remove(queued_job)
        generation_queue.task_done()

        fut = queued_job.get("future")
        if fut and not fut.done():
            fut.cancel()

        meta["status"] = "cancelled"
        meta["finished_at"] = time.time()
        await broadcast_queue_update()
        return {"status": "cancelled", "job_id": job_id}

    return JSONResponse(status_code=409, content={"error": "Job is not queued"})


@app.get("/health")
def health():
    """System health check."""
    import torch
    from webbduck.core.pipeline import pipeline_manager
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
        "vram": vram,
        "models": {
            "count": len(MODEL_REGISTRY),
            "names": list(MODEL_REGISTRY.keys()),
        },
        "loras": {
            "count": len(LORA_REGISTRY),
            "names": list(LORA_REGISTRY.keys()),
        },
        "queue": {
            "size": generation_queue.qsize(),
            "maxsize": generation_queue.maxsize,
            "active_job_id": active_job_id,
            "tracked_jobs": len(job_registry),
        },
        "pipeline": {
            "loaded": pipeline_manager.pipe is not None,
            "base_model": pipeline_manager.key,
            "second_pass_model": pipeline_manager.current_second_pass_model,
            "loras": list(pipeline_manager.current_loras.keys()),
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
    style: str = Form("detailed"),
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
        response = FileResponse(thumb_path)
        # Smaller chunks reduce peak per-request memory under high concurrency.
        response.chunk_size = 16 * 1024
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Thumbnail error: {e}"})
