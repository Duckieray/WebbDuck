"""Image-to-image generation mode."""

import torch
from webbduck.modes.base import GenerationMode
from webbduck.core.pipeline import pipeline_manager


class Img2ImgMode(GenerationMode):
    def can_run(self, settings, pipe, img2img):
        return (
            settings.get("input_image") is not None
            and img2img is not None
        )

    def run(self, *, settings, pipe, img2img, generator):
        assert img2img is not None, "Img2ImgMode requires img2img pipeline"

        image = settings["input_image"]
        prompt = settings["prompt"]
        negative = settings["negative_prompt"]

        is_latent = isinstance(image, torch.Tensor)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "num_inference_steps": settings["steps"],
            "guidance_scale": settings["cfg"],
            "generator": generator,
        }

        if is_latent:
            kwargs["image"] = image
            kwargs["denoising_start"] = settings.get("denoising_start", 0.8)
            kwargs["output_type"] = "latent"
        else:
            kwargs["image"] = image
            kwargs["strength"] = settings.get("strength", 0.5)

        pipeline_manager.set_active_unet("second_pass")
        result = img2img(**kwargs)
        images = result.images if hasattr(result, "images") else result
        
        return images, generator.initial_seed()