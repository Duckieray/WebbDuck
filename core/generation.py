"""Main generation orchestration."""

from pathlib import Path
import torch
import random
from PIL import Image, ImageOps

from webbduck.core.pipeline import pipeline_manager
from webbduck.core.captioner import unload_captioners
from webbduck.modes import select_mode


def run_generation(settings):
    """Execute image generation based on settings."""
    second_pass_model = settings.get("second_pass_model")
    if second_pass_model == "None":
        second_pass_model = None

    # Unload any captioner models to free VRAM before loading generation pipelines
    unload_captioners()

    pipe, img2img, base_img2img, base_inpaint, _ = pipeline_manager.get(
        base_model=settings["base_model"],
        second_pass_model=second_pass_model,
        loras=settings.get("loras", []),
        scheduler_name=settings.get("scheduler"),
    )
    
    # Load input image if present (for Img2Img)
    if settings.get("image"):
        img_path = Path(settings["image"])
        if img_path.exists():
            # Load and ensure RGB
            pil_image = Image.open(img_path).convert("RGB")
            
            # Resize to target dimensions
            target_w = int(settings.get("width", 1024))
            target_h = int(settings.get("height", 1024))
            
            if pil_image.size != (target_w, target_h):
                # Use ImageOps.fit to center crop and resize, preserving aspect ratio
                pil_image = ImageOps.fit(pil_image, (target_w, target_h), method=Image.LANCZOS)
                
            settings["input_image"] = pil_image
            
            # Load mask if present
            if settings.get("mask_image"):
                mask_path = Path(settings["mask_image"])
                if mask_path.exists():
                    try:
                        mask_pil = Image.open(mask_path).convert("L") # Mask should be grayscale
                        
                        if mask_pil.size != pil_image.size:
                             mask_pil = ImageOps.fit(mask_pil, pil_image.size, method=Image.NEAREST)
                             
                        settings["mask_image"] = mask_pil
                    except Exception as e:
                         print(f"Failed to load mask: {e}")
                         del settings["mask_image"]
                else:
                    # Path implies existence, but if not found, remove it
                    del settings["mask_image"]


    seed = settings.get("seed") or random.randint(0, 2**32 - 1)
    generator = torch.Generator(pipe.device).manual_seed(seed)

    mode = select_mode(settings, pipe, img2img, base_img2img, base_inpaint)

    images, out_seed = mode.run(
        settings=settings,
        pipe=pipe,
        img2img=img2img,
        base_img2img=base_img2img,
        base_inpaint=base_inpaint,
        generator=generator,
    )

    return images, out_seed