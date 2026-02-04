"""Image-to-image generation mode."""

import torch
from webbduck.modes.base import GenerationMode
from webbduck.core.pipeline import pipeline_manager


class Img2ImgMode(GenerationMode):
    def can_run(self, settings, pipe, img2img, base_img2img):
        return settings.get("input_image") is not None

    def run(self, *, settings, pipe, img2img, base_img2img, generator):
        # Prefer second pass model (refiner) if available, per user request
        if img2img is not None:
            active_pipe = img2img
            pipeline_manager.set_active_unet("second_pass")
        else:
            active_pipe = base_img2img
            pipeline_manager.set_active_unet("base")

        image = settings["input_image"]
        prompt = settings["prompt"]
        prompt_2 = settings.get("prompt_2")
        negative = settings["negative_prompt"]

        is_latent = isinstance(image, torch.Tensor)

        kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "negative_prompt": negative,
            "num_inference_steps": settings["steps"],
            "guidance_scale": settings["cfg"],
            "generator": generator,
            "strength": settings.get("strength", 0.75),
            "num_images_per_prompt": settings["num_images"], # Default strength
        }

        if is_latent:
            kwargs["image"] = image
            kwargs["denoising_start"] = settings.get("denoising_start", 0.8)
            kwargs["output_type"] = "latent"
            # For pure latent input, we might need different handling if using base_img2img
            # But usually input_image from UI is a PIL Image
        else:
            kwargs["image"] = image

        result = active_pipe(**kwargs)
        images = result.images
        
        return images, generator.initial_seed()