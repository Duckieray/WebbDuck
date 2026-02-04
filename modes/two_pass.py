"""Two-pass generation mode with optional experimental prompt splitting."""

import torch
import logging
from webbduck.prompt.experimental import (
    split_prompt_for_two_pass,
    build_sdxl_conditioning_dispatch,
    build_sdxl_refiner_conditioning,
)
from webbduck.prompt.management import truncate_to_tokens
from webbduck.core.pipeline import pipeline_manager
from webbduck.modes.base import GenerationMode

log = logging.getLogger(__name__)


def resolve_second_pass_type(settings, img2img):
    mode = settings.get("second_pass_mode", "auto")

    if mode in ("refiner", "img2img"):
        return mode

    cross_dim = getattr(img2img.unet.config, "cross_attention_dim", 2048)

    if cross_dim <= 1280:
        return "refiner"

    return "img2img"

def build_refiner_added_cond(pipe, height, width, batch_size):
    device = pipe.device

    add_time_ids = torch.tensor(
        [
            height, width,  # original_size
            height, width,  # target_size
            0, 0,           # crop_coords
        ],
        device=device,
        dtype=torch.long,
    )

    add_time_ids = add_time_ids.repeat(batch_size, 1)

    return {
        "time_ids": add_time_ids,
    }

def decode_latents(pipe, latents):
    """Decode latent tensors to PIL images."""
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

    images = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor
    ).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.float().cpu().permute(0, 2, 3, 1)

    from PIL import Image
    return [
        Image.fromarray((img.numpy() * 255).astype("uint8"))
        for img in images
    ]


class TwoPassMode(GenerationMode):
    def can_run(self, settings, pipe, img2img, base_img2img, base_inpaint=None):
        second_pass = settings.get("second_pass_model")
        has_second_pass = second_pass not in (None, "", "None")
        return has_second_pass and img2img is not None

    def run(self, *, settings, pipe, img2img, base_img2img, base_inpaint, generator):
        log.info("Entering two-pass generation")
        
        pipeline_manager.set_active_unet("base")
        torch.cuda.synchronize()
    
        prompt = settings["prompt"]
        prompt_2 = settings.get("prompt_2")
        negative = settings.get("negative_prompt")
        height = settings["height"]
        width = settings["width"]
        experimental = settings.get("experimental_compress", False)

        # Pass 1: Base composition
        if experimental:
            base_prompt, late_prompt = split_prompt_for_two_pass(
                pipe.tokenizer, prompt
            )
            base_prompt_2, late_prompt_2 = split_prompt_for_two_pass(
                pipe.tokenizer_2, prompt_2 or prompt
            )
        else:
            base_prompt = prompt
            base_prompt_2 = prompt_2 or prompt
            late_prompt = None
            late_prompt_2 = None

        base_prompt = truncate_to_tokens(pipe.tokenizer, base_prompt, max_tokens=75)
        base_prompt_2 = truncate_to_tokens(pipe.tokenizer_2, base_prompt_2, max_tokens=75)

        log.debug(f"Pass 1 tokens: {len(pipe.tokenizer(base_prompt)['input_ids'])}")
        
        with torch.no_grad():
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = build_sdxl_conditioning_dispatch(
                pipe=pipe,
                prompt=base_prompt,
                prompt_2=base_prompt_2,
                negative=negative,
                experimental=experimental,
            )

            pipeline_manager.set_active_unet("base")

            base_latents = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=width,
                height=height,
                num_inference_steps=settings["steps"],
                guidance_scale=min(settings["cfg"], 6.0),
                num_images_per_prompt=settings["num_images"],
                generator=generator,
                output_type="latent",
            ).images

        # Pass 2: Refinement
        if experimental and late_prompt is None and late_prompt_2 is None:
            if settings.get("second_pass_mode") != "img2img":
                log.info("Skipping pass 2 (no late content)")
                pipeline_manager.set_active_unet("base")
                images = decode_latents(pipe, base_latents)
                return images, generator.initial_seed()

        second_pass_type = resolve_second_pass_type(settings, img2img)
        log.info(f"Pass 2 mode: {second_pass_type}")
      
        refine_prompt = late_prompt or "highly detailed, realistic skin texture"
        refine_prompt_2 = late_prompt_2 or None

        if img2img is None:
            log.warning("No second pass model loaded")
            return base_latents, generator.initial_seed()

        pipeline_manager.set_active_unet("second_pass")
        torch.cuda.synchronize()

        with torch.no_grad():
            if second_pass_type == "refiner":
                log.info("Using true SDXL refiner")
                log.debug(
                    f"Refiner config: "
                    f"addition_time_embed_dim={img2img.unet.config.addition_time_embed_dim}, "
                    f"requires_aesthetics_score={img2img.config.requires_aesthetics_score}"
                )

                # True Refiner detected
                img2img.register_to_config(requires_aesthetics_score=True)

                (
                    r_prompt_embeds,
                    r_pooled_prompt_embeds,
                    r_neg_prompt_embeds,
                    r_neg_pooled_prompt_embeds,
                ) = build_sdxl_refiner_conditioning(
                    pipe=img2img,
                    prompt=refine_prompt,
                    prompt_2=refine_prompt_2,
                    negative=negative,
                )

                extra_args = {
                    "original_size": (height, width),
                    "crops_coords_top_left": (0, 0),
                    "aesthetic_score": 6.0,
                }

                refined_latents = img2img(
                    image=base_latents,
                    prompt_embeds=r_prompt_embeds,
                    pooled_prompt_embeds=r_pooled_prompt_embeds,
                    negative_prompt_embeds=r_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=r_neg_pooled_prompt_embeds,
                    guidance_scale=settings["cfg"],
                    num_inference_steps=int(settings["steps"] * 0.7),
                    strength=settings.get("refinement_strength", 0.3),
                    output_type="latent",
                    generator=generator,
                    **extra_args,
                ).images

            else:
                log.info("Using generic SDXL img2img model")

                (
                    p_embeds,
                    p_pooled,
                    n_embeds,
                    n_pooled,
                ) = build_sdxl_conditioning_dispatch(
                    pipe=img2img,
                    prompt=refine_prompt,
                    prompt_2=refine_prompt_2,
                    negative=negative,
                    experimental=False,
                )

                bsz = base_latents.shape[0]
                p_embeds = p_embeds.repeat(bsz, 1, 1)
                n_embeds = n_embeds.repeat(bsz, 1, 1)
                p_pooled = p_pooled.repeat(bsz, 1)
                n_pooled = n_pooled.repeat(bsz, 1)

                assert (
                    p_embeds.shape[-1] == img2img.unet.config.cross_attention_dim
                ), "Prompt / UNet dimension mismatch"

                refined_latents = img2img(
                    image=base_latents,
                    prompt_embeds=p_embeds,
                    pooled_prompt_embeds=p_pooled,
                    negative_prompt_embeds=n_embeds,
                    negative_pooled_prompt_embeds=n_pooled,
                    guidance_scale=settings["cfg"],
                    num_inference_steps=int(settings["steps"] * 0.7),
                    strength=settings.get("refinement_strength", 0.3),
                    generator=generator,
                    output_type="latent",
                    original_size=(height, width),
                    target_size=(height, width),
                ).images

            pipeline_manager.set_active_unet("base")
            images = decode_latents(pipe, refined_latents)
            return images, generator.initial_seed()
