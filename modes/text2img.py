"""Text-to-image generation mode."""

import logging
from webbduck.core.pipeline import pipeline_manager
from webbduck.modes.base import GenerationMode
from webbduck.prompt.experimental import build_sdxl_conditioning_dispatch

log = logging.getLogger(__name__)


class Text2ImgMode(GenerationMode):
    def can_run(self, settings, pipe, img2img, base_img2img, base_inpaint=None):
        return settings.get("input_image") is None

    def run(self, *, settings, pipe, img2img, base_img2img, base_inpaint, generator, callback=None):
        num_images = settings["num_images"]
        prompt = settings["prompt"]
        prompt_2 = settings.get("prompt_2")
        negative = settings["negative_prompt"]

        cb = callback.get_callback() if callback else None

        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = build_sdxl_conditioning_dispatch(
            pipe=pipe,
            prompt=prompt,
            prompt_2=prompt_2,
            negative=negative,
            experimental=settings.get("experimental_compress", False),
        )

        pipeline_manager.set_active_unet("base")

        images = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            added_cond_kwargs={},
            cross_attention_kwargs=None,
            width=settings["width"],
            height=settings["height"],
            num_inference_steps=settings["steps"],
            guidance_scale=settings["cfg"],
            num_images_per_prompt=num_images,
            generator=generator,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=['latents'],
        ).images

        return images, generator.initial_seed()