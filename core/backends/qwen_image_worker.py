"""Standalone Qwen Image Diffusers worker.

The worker owns Qwen-specific pipeline classes and memory policy so WebbDuck's
core process remains architecture agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def _snap(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _load_image(path: str | None, mode: str):
    if not path:
        return None
    from PIL import Image

    with Image.open(path) as image:
        return image.convert(mode).copy()


def _configure_memory(pipe, device: str, total_vram_gb: float) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("WEBBDUCK_QWEN_IMAGE_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        # Qwen-Image-2512 contains a 20B image model plus a 7B-class vision-language
        # encoder. Whole-component model offload can still exceed a 16 GB card,
        # so use submodule-level sequential offload below 24 GB.
        mode = "sequential" if total_vram_gb < 24.0 else "model"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"
    if mode in {"model", "cpu", "1", "true", "yes"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    pipe.to("cuda")
    return "none"


def _pipeline_for_operation(model_path: str, operation: str, dtype):
    if operation == "img2img":
        from diffusers import QwenImageImg2ImgPipeline

        return QwenImageImg2ImgPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if operation == "inpaint":
        from diffusers import QwenImageInpaintPipeline

        return QwenImageInpaintPipeline.from_pretrained(model_path, torch_dtype=dtype)

    from diffusers import QwenImagePipeline

    return QwenImagePipeline.from_pretrained(model_path, torch_dtype=dtype)


def _run(request: dict, output_dir: Path) -> dict:
    import torch

    model_path = str(request["model_path"])
    operation = str(request.get("operation") or "text2img").strip().lower()
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")

    negative_prompt = str(request.get("negative_prompt") or "")
    width = _snap(int(request.get("width") or 1328))
    height = _snap(int(request.get("height") or 1328))
    steps = max(1, int(request.get("steps") or 50))
    cfg = float(request.get("true_cfg_scale") if request.get("true_cfg_scale") is not None else 4.0)
    strength = min(1.0, max(0.0, float(request.get("strength") or 0.85)))
    num_images = max(1, int(request.get("num_images") or 1))
    seed = int(request.get("seed") if request.get("seed") is not None else 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = _pipeline_for_operation(model_path, operation, dtype)

    total_vram_gb = 0.0
    if device == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            total_vram_gb = 0.0
    offload = _configure_memory(pipe, device, total_vram_gb)

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    input_image = _load_image(request.get("input_image"), "RGB")
    mask_image = _load_image(request.get("mask_image"), "L")
    if operation == "img2img" and input_image is None:
        raise ValueError("Qwen Image img2img requires an input image")
    if operation == "inpaint" and (input_image is None or mask_image is None):
        raise ValueError("Qwen Image inpaint requires both an input image and a mask")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index in range(num_images):
        generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed + index)
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg,
            "generator": generator,
        }
        if operation == "text2img":
            kwargs.update({"width": width, "height": height})
        elif operation == "img2img":
            kwargs.update({"image": input_image, "strength": strength})
        elif operation == "inpaint":
            kwargs.update(
                {
                    "image": input_image,
                    "mask_image": mask_image,
                    "strength": strength,
                }
            )
        else:
            raise ValueError(f"Unsupported Qwen Image operation: {operation}")

        result = pipe(**kwargs)
        image = result.images[0]
        path = output_dir / f"qwen_image_{index:03d}.png"
        image.save(path)
        saved.append(str(path))

    return {
        "ok": True,
        "images": saved,
        "seed": seed,
        "runtime": {
            "operation": operation,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "offload": offload,
            "total_vram_gb": round(total_vram_gb, 2),
        },
    }


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
