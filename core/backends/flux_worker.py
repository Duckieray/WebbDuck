"""Standalone FLUX Diffusers worker.

This file intentionally imports torch/diffusers only inside the isolated runtime.
It can therefore be launched with WEBBDUCK_FLUX_PYTHON pointing at a backend-
specific environment without constraining WebbDuck's core dependency set.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import traceback
from pathlib import Path


def _snap(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _run(request: dict, output_dir: Path) -> dict:
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline

    model_path = str(request["model_path"])
    prompt = str(request["prompt"])
    width = _snap(int(request.get("width") or 1024))
    height = _snap(int(request.get("height") or 1024))
    steps = max(1, int(request.get("steps") or 4))
    guidance = float(request.get("guidance") if request.get("guidance") is not None else 1.0)
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") or 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)

    offload_mode = str(os.getenv("WEBBDUCK_FLUX_OFFLOAD", "auto")).strip().lower()
    total_vram_gb = 0.0
    if device == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total_vram_gb = 0.0

    if device == "cuda" and (offload_mode in {"1", "true", "yes", "cpu"} or (offload_mode == "auto" and total_vram_gb < 18.0)):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    input_image = None
    input_path = request.get("input_image")
    if input_path:
        input_image = Image.open(str(input_path)).convert("RGB")

    try:
        call_sig = inspect.signature(pipe.__call__)
        call_params = call_sig.parameters
    except Exception:
        call_params = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index in range(num_images):
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
        }
        if input_image is not None and (not call_params or "image" in call_params):
            kwargs["image"] = input_image
        generator_device = device if device == "cuda" else "cpu"
        kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed + index)
        result = pipe(**kwargs)
        image = result.images[0]
        path = output_dir / f"flux_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    return {"ok": True, "images": saved, "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = _run(request, Path(args.output_dir))
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
