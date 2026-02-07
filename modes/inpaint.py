"""Inpainting generation mode."""

import logging
from .base import GenerationMode

log = logging.getLogger(__name__)

class InpaintMode(GenerationMode):
    def can_run(self, settings, pipe, img2img, base_img2img, base_inpaint=None):
        return (
            settings.get("mask_image") is not None
            and settings.get("input_image") is not None
        )

    def run(self, *, settings, pipe, img2img, base_img2img, base_inpaint, generator):
        log.info("Running Inpaint Mode")

        prompt = settings["prompt"]
        prompt_2 = settings.get("prompt_2", "")
        negative_prompt = settings.get("negative_prompt", "")
        # For inpainting, strength essentially acts like denoise strength on the masked area.
        # But standard InpaintPipeline usually takes 'strength' as how much to change the masked area? 
        # Actually SDXL Inpaint takes `strength` (0.0-1.0). 1.0 = destruct.
        strength = float(settings.get("strength", 0.99)) # Default to high replacement if not specified? 
        # Frontend slider defaults to 0.75 which is good.
        
        guidance_scale = float(settings.get("cfg", 7.5))
        num_inference_steps = int(settings.get("steps", 30))
        num_images_per_prompt = int(settings.get("num_images", 1))
        
        image = settings["input_image"]
        mask_image = settings["mask_image"]
        
        
        inpainting_fill = settings.get("inpainting_fill", "replace")
        
        # Handle "Keep Masked" (Protect Mode)
        if inpainting_fill == "keep":
             # Invert mask so we paint what we want to PROTECT
             from PIL import ImageOps
             mask_image = ImageOps.invert(mask_image)
             log.info("Inverted mask for 'Keep Masked' mode")

        # Apply Mask Blur / Feathering
        mask_blur = int(settings.get("mask_blur", 0))
        if mask_blur > 0:
            from PIL import ImageFilter
            log.info(f"Applying mask blur: {mask_blur}")
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        # Standard SDXL Inpainting uses the base model via StableDiffusionXLInpaintPipeline
        # We assume base_inpaint is available.
        if base_inpaint is None:
             raise RuntimeError("Inpainting pipeline not available")
             
        # Scheduler swap handled in pipeline manager

        out = base_inpaint(
            prompt=prompt,
            prompt_2=prompt_2 or None,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            width=int(settings.get("width", 1024)),
            height=int(settings.get("height", 1024)),
            generator=generator,
        ).images

        return out, settings.get("seed")
