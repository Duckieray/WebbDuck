"""FastAPI application and endpoints."""

import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, Form, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from webbduck.server.state import snapshot
from webbduck.server.events import broadcast_state, active_sockets
from webbduck.server.storage import save_images, BASE, to_web_path, resolve_web_path
from webbduck.core.worker import gpu_worker
from webbduck.models.registry import MODEL_REGISTRY, LORA_REGISTRY

app = FastAPI()

# Queue for GPU jobs
generation_queue = asyncio.Queue(maxsize=1)


async def enqueue(job):
    """Enqueue job and wait for result."""
    await generation_queue.put(job)
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket connection for live updates."""
    await ws.accept()
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
    prompt: str = Form(...),
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

    job = {
        "type": "test",
        "settings": settings,
        "future": future,
    }

    try:
        validate_second_pass(settings)
    except ValueError as e:
        return {
            "error": str(e),
            "type": "validation_error"
        }

    return await enqueue(job)


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
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
    }

    job = {
        "type": "batch",
        "settings": settings,
        "future": future,
    }

    try:
        validate_second_pass(settings)
    except ValueError as e:
        return {
            "error": str(e),
            "type": "validation_error"
        }

    return await enqueue(job)


@app.get("/gallery")
def gallery():
    """List all generated image runs."""
    runs = sorted(BASE.iterdir(), reverse=True)
    out = []
    for r in runs:
        if not r.is_dir():
            continue
        meta_file = r / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.load(open(meta_file))
        imgs = [to_web_path(p) for p in r.glob("*.png")]
        out.append({"run": r.name, "images": imgs, "meta": meta})
    return out


@app.get("/models")
def list_models():
    """List available base models."""
    return [
        {
            "name": name,
            "defaults": info.get("defaults", {}),
            "type": info.get("type", "unknown"),
        }
        for name, info in MODEL_REGISTRY.items()
    ]


@app.get("/second_pass_models")
def list_second_pass_models():
    """List available second pass models."""
    return list(MODEL_REGISTRY.keys())


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
):
    """Upscale an image."""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    job = {
        "type": "upscale",
        "image": str(resolve_web_path(image)),
        "scale": scale,
        "future": future,
    }

    await generation_queue.put(job)
    return await future


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
    text: str = Form(...),
    base_model: str = Form(...),
    which: str = Form("prompt"),
):
    """Count tokens in prompt."""
    from webbduck.core.pipeline import pipeline_manager

    pipe, _, _ = pipeline_manager.get(
        base_model=base_model,
        second_pass_model=None,
        loras=[],
    )

    tokenizer = (
        pipe.tokenizer_2 if which == "prompt_2"
        else pipe.tokenizer
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