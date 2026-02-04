"""GPU worker for processing generation and upscale jobs."""

import asyncio
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

from webbduck.core.generation import run_generation
from webbduck.server.storage import save_images, append_session_entry
from webbduck.server.events import broadcast_state
from webbduck.server.state import update_stage, update_progress, snapshot
from webbduck.models.upscaler import get_upsampler


async def run_upscale(job):
    """Execute upscale task."""
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
        upscaled_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)

    update_progress(0.9)
    await broadcast_state(snapshot())

    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(upscaled_rgb)

    out_path = image_path.with_name(f"{image_path.stem}_x{scale}.png")
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
        
        if job["type"] == "upscale":
            try:
                result = await run_upscale(job)
                job["future"].set_result({"image": result})
            except Exception as e:
                update_stage("Error")
                await broadcast_state(snapshot())
                job["future"].set_exception(e)
            finally:
                queue.task_done()
            continue

        try:
            update_stage("Generating")
            update_progress(0.4)
            await broadcast_state(snapshot())

            images, seed = await loop.run_in_executor(
                None, run_generation, job["settings"]
            )

            update_stage("Decoding")
            update_progress(0.85)
            await broadcast_state(snapshot())

            update_stage("Saving")
            update_progress(0.9)
            await broadcast_state(snapshot())

            paths = save_images(images, job["settings"])

            update_progress(1.0)
            update_stage("Idle")
            await broadcast_state(snapshot())

            job["future"].set_result({
                "seed": seed,
                "images": paths,
            })

            append_session_entry({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": job["type"],
                "seed": seed,
                "settings": job["settings"],
                "images": paths,
            })

        except Exception as e:
            update_stage("Error")
            await broadcast_state(snapshot())
            job["future"].set_exception(e)

        finally:
            queue.task_done()