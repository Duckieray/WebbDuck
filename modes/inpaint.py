"""Inpainting generation mode."""

import logging
from PIL import Image, ImageOps, ImageFilter, ImageStat
import torch
from .base import GenerationMode
from webbduck.server.state import update_stage

log = logging.getLogger(__name__)


def _build_smart_extend_seam_mask(size, source_box, seam_width):
    width, height = size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    seam = max(2, int(seam_width))
    # Blend both inward and outward more aggressively to reduce visible box seams.
    seam_out = max(seam, int(round(seam * 1.25)))
    seam_in = max(6, int(round(seam * 0.85)))

    mask = Image.new("L", (width, height), 0)
    mx0, my0 = x, y
    mx1, my1 = x + w, y + h

    # Paint a centered seam band across boundaries that face extension.
    # This includes both sides of the boundary (inward + outward) so blending
    # can harmonize old and new pixels together.
    if mx0 > 0:
        mask.paste(255, (max(0, mx0 - seam_out), my0, min(width, mx0 + seam_in), my1))
    if mx1 < width:
        mask.paste(255, (max(0, mx1 - seam_in), my0, min(width, mx1 + seam_out), my1))
    if my0 > 0:
        mask.paste(255, (mx0, max(0, my0 - seam_out), mx1, min(height, my0 + seam_in)))
    if my1 < height:
        mask.paste(255, (mx0, max(0, my1 - seam_in), mx1, min(height, my1 + seam_out)))

    return mask


def _round8(value, minimum=8):
    return max(int(minimum), int(round(float(value) / 8.0) * 8))


def _build_step_fractions(src_w, src_h, dst_w, dst_h, growth):
    if dst_w <= src_w and dst_h <= src_h:
        return [1.0]

    # Allow larger growth jumps so users can reduce pass count aggressively.
    growth = max(1.05, min(3.00, float(growth)))
    fractions = []
    curr_w, curr_h = src_w, src_h
    progress = 0.0

    span_w = max(1, dst_w - src_w)
    span_h = max(1, dst_h - src_h)

    for _ in range(32):
        if curr_w >= dst_w and curr_h >= dst_h:
            break

        next_w = min(dst_w, _round8(curr_w * growth, minimum=src_w))
        next_h = min(dst_h, _round8(curr_h * growth, minimum=src_h))
        if next_w == curr_w and curr_w < dst_w:
            next_w = min(dst_w, curr_w + 8)
        if next_h == curr_h and curr_h < dst_h:
            next_h = min(dst_h, curr_h + 8)

        p_w = ((next_w - src_w) / span_w) if dst_w > src_w else 1.0
        p_h = ((next_h - src_h) / span_h) if dst_h > src_h else 1.0
        next_progress = min(1.0, max(progress + 0.01, p_w, p_h))

        fractions.append(next_progress)
        progress = next_progress

        curr_w = _round8(src_w + (dst_w - src_w) * progress, minimum=src_w)
        curr_h = _round8(src_h + (dst_h - src_h) * progress, minimum=src_h)

        if progress >= 0.999:
            break

    if not fractions or fractions[-1] < 1.0:
        fractions.append(1.0)
    return fractions


def _build_stage_canvas_and_mask(current_image, p_curr, p_next, src_w, src_h, dst_w, dst_h, left, top, feather):
    if p_next >= 0.999:
        stage_w, stage_h = dst_w, dst_h
    else:
        stage_w = _round8(src_w + (dst_w - src_w) * p_next, minimum=src_w)
        stage_h = _round8(src_h + (dst_h - src_h) * p_next, minimum=src_h)

    curr_w, curr_h = current_image.size
    nx = int(round(left * p_next))
    ny = int(round(top * p_next))
    cx = int(round(left * p_curr))
    cy = int(round(top * p_curr))
    ox = nx - cx
    oy = ny - cy

    # Context initializer for each stage.
    # Important: do NOT project a resized full previous image into newly added areas.
    # That can "echo" anatomy into empty space and create repeated-body artifacts.
    top_strip = current_image.crop((0, 0, curr_w, 1)).resize((curr_w, 1))
    bottom_strip = current_image.crop((0, curr_h - 1, curr_w, curr_h)).resize((curr_w, 1))
    left_strip = current_image.crop((0, 0, 1, curr_h)).resize((curr_h, 1))
    right_strip = current_image.crop((curr_w - 1, 0, curr_w, curr_h)).resize((curr_h, 1))
    border_strip = Image.new("RGB", (curr_w * 2 + curr_h * 2, 1))
    cursor = 0
    for seg in (top_strip, bottom_strip, left_strip, right_strip):
        border_strip.paste(seg, (cursor, 0))
        cursor += seg.width
    mean = ImageStat.Stat(border_strip).mean
    fill_rgb = tuple(int(max(0, min(255, round(v)))) for v in mean[:3])

    stage_bg = Image.new("RGB", (stage_w, stage_h), fill_rgb)
    # Gentle blur just smooths flat fill; no structure injection.
    blur_r = max(6, min(24, int(min(stage_w, stage_h) * 0.015)))
    stage_bg = stage_bg.filter(ImageFilter.GaussianBlur(radius=blur_r))
    stage_image = stage_bg
    stage_image.paste(current_image, (ox, oy))

    # White = repaint; Black = keep previous stage.
    stage_mask = Image.new("L", (stage_w, stage_h), 255)
    stage_mask.paste(0, (ox, oy, ox + curr_w, oy + curr_h))

    seam = max(1, min(10, 2 + int(feather) // 4))
    if ox > 0:
        stage_mask.paste(255, (ox, oy, min(stage_w, ox + seam), oy + curr_h))
    if (ox + curr_w) < stage_w:
        stage_mask.paste(255, (max(0, ox + curr_w - seam), oy, ox + curr_w, oy + curr_h))
    if oy > 0:
        stage_mask.paste(255, (ox, oy, ox + curr_w, min(stage_h, oy + seam)))
    if (oy + curr_h) < stage_h:
        stage_mask.paste(255, (ox, max(0, oy + curr_h - seam), ox + curr_w, oy + curr_h))

    placement = {
        "ox": ox,
        "oy": oy,
        "w": curr_w,
        "h": curr_h,
    }
    return stage_image, stage_mask, placement


def _build_stage_refine_mask(size, placement, seam_width):
    width, height = size
    ox = int(placement.get("ox", 0))
    oy = int(placement.get("oy", 0))
    w = int(placement.get("w", 0))
    h = int(placement.get("h", 0))
    seam = max(2, int(seam_width))
    seam_out = max(seam, int(round(seam * 1.25)))
    seam_in = max(6, int(round(seam * 0.85)))

    mask = Image.new("L", (width, height), 0)
    if ox > 0:
        mask.paste(255, (max(0, ox - seam_out), oy, min(width, ox + seam_in), oy + h))
    if (ox + w) < width:
        mask.paste(255, (max(0, ox + w - seam_in), oy, min(width, ox + w + seam_out), oy + h))
    if oy > 0:
        mask.paste(255, (ox, max(0, oy - seam_out), ox + w, min(height, oy + seam_in)))
    if (oy + h) < height:
        mask.paste(255, (ox, max(0, oy + h - seam_in), ox + w, min(height, oy + h + seam_out)))
    return mask


def _build_soft_source_keep_mask(size, source_box, blend_radius):
    """Build soft keep mask for source region with inward/outward feather."""
    width, height = size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    if w <= 0 or h <= 0:
        return None

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (x0, y0, x1, y1))
    r = max(0.0, float(blend_radius))
    if r > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
    return mask


def _smart_extend_preserve_blend(src, generated, source_box, feather_px=8, seam_width=24, preserve_weight=1.0):
    """Blend original source back in with a soft source-box keep mask."""
    blend_radius = max(
        10.0,
        min(96.0, max(float(feather_px) * 2.0, float(seam_width) * 2.25)),
    )
    keep_mask = _build_soft_source_keep_mask(generated.size, source_box, blend_radius)
    if keep_mask is None:
        return generated
    w = max(0.0, min(1.0, float(preserve_weight)))
    if w < 0.999:
        keep_mask = keep_mask.point(lambda v: int(v * w))
    return Image.composite(src, generated.convert("RGB"), keep_mask)

class InpaintMode(GenerationMode):
    def can_run(self, settings, pipe, img2img, base_img2img, base_inpaint=None):
        return (
            settings.get("mask_image") is not None
            and settings.get("input_image") is not None
        )

    def run(self, *, settings, pipe, img2img, base_img2img, base_inpaint, generator, callback=None):
        log.info("Running Inpaint Mode")

        prompt = settings["prompt"]
        prompt_2 = settings.get("prompt_2", "")
        negative_prompt = settings.get("negative_prompt", "")
        # For inpainting, strength essentially acts like denoise strength on the masked area.
        # But standard InpaintPipeline usually takes 'strength' as how much to change the masked area? 
        # Actually SDXL Inpaint takes `strength` (0.0-1.0). 1.0 = destruct.
        strength = float(settings.get("strength", 0.99)) # Default to high replacement if not specified? 
        # Frontend slider defaults to 0.75 which is good.
        smart_extend_enabled = bool(settings.get("smart_extend"))
        
        guidance_scale = float(settings.get("cfg", 7.5))
        num_inference_steps = int(settings.get("steps", 30))
        num_images_per_prompt = int(settings.get("num_images", 1))
        
        image = settings["input_image"]
        mask_image = settings["mask_image"]
        
        
        inpainting_fill = settings.get("inpainting_fill", "replace")
        
        # Handle "Keep Masked" (Protect Mode)
        if inpainting_fill == "keep":
             # Invert mask so we paint what we want to PROTECT
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

        cb = callback.get_callback() if callback else None
        out = None

        if settings.get("smart_extend") and bool(settings.get("smart_extend_auto_step", True)):
            source_box = settings.get("smart_extend_source_box") or {}
            sb_x = int(source_box.get("x", 0))
            sb_y = int(source_box.get("y", 0))
            sb_w = int(source_box.get("w", 0))
            sb_h = int(source_box.get("h", 0))
            final_w = int(settings.get("width", image.size[0]))
            final_h = int(settings.get("height", image.size[1]))

            if sb_w > 0 and sb_h > 0 and final_w >= sb_w and final_h >= sb_h:
                left = sb_x
                top = sb_y
                source_crop = image.crop((sb_x, sb_y, sb_x + sb_w, sb_y + sb_h)).convert("RGB")
                growth = float(settings.get("smart_extend_step_growth", 1.25))
                feather = int(settings.get("smart_extend_feather", 8))
                refine_enabled = bool(settings.get("smart_extend_refine", True))
                refine_each_step = bool(settings.get("smart_extend_refine_each_step", True))
                refine_width = int(settings.get("smart_extend_refine_width", 24))
                refine_strength_base = float(settings.get("smart_extend_refine_strength", 0.28))
                fractions = _build_step_fractions(sb_w, sb_h, final_w, final_h, growth)

                base_seed = generator.initial_seed()
                staged_out = []
                batch_count = max(1, num_images_per_prompt)
                total_passes = len(fractions)
                # Smart-extend batches are processed sequentially to avoid large-memory
                # final-stage spikes on high-resolution outpaint jobs.
                for img_idx in range(batch_count):
                    p_curr = 0.0
                    curr_image = source_crop
                    staged_image = None
                    image_seed_base = base_seed + (img_idx * 1000003)
                    img_label = f" (img {img_idx + 1}/{batch_count})" if batch_count > 1 else ""

                    for idx, p_next in enumerate(fractions):
                        is_final = idx == (len(fractions) - 1)
                        stage_steps = num_inference_steps if is_final else max(
                            12,
                            int(num_inference_steps * (0.55 + 0.45 * float(p_next))),
                        )
                        if callback:
                            update_stage(f"Outpaint pass {idx + 1}/{total_passes}{img_label}")
                        stage_img, stage_mask, stage_place = _build_stage_canvas_and_mask(
                            curr_image,
                            p_curr,
                            p_next,
                            sb_w,
                            sb_h,
                            final_w,
                            final_h,
                            left,
                            top,
                            feather,
                        )
                        local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + (idx * 1009))
                        stage_kwargs = {
                            "prompt": prompt,
                            "prompt_2": prompt_2 or None,
                            "negative_prompt": negative_prompt,
                            "image": stage_img,
                            "mask_image": stage_mask,
                            "strength": strength,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": stage_steps,
                            "num_images_per_prompt": 1,
                            "width": stage_img.size[0],
                            "height": stage_img.size[1],
                            "generator": local_gen,
                        }
                        if cb:
                            stage_kwargs["callback_on_step_end"] = cb
                            stage_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']

                        stage_out = base_inpaint(**stage_kwargs).images
                        if callback and hasattr(callback, "finish_pass"):
                            callback.finish_pass(stage_steps)

                        # Keep previous stage stable and blend edge softly.
                        keep_mask = ImageOps.invert(stage_mask)
                        # Preserve prior pass output strongly to prevent anatomy drift
                        # on second/third expansions.
                        blend_r = max(0.0, min(1.0, feather / 20.0))
                        if blend_r > 0:
                            keep_mask = keep_mask.filter(ImageFilter.GaussianBlur(radius=blend_r))
                        composed = []
                        for img in stage_out:
                            gen = img.convert("RGB")
                            preserve = gen.copy()
                            preserve.paste(curr_image, (stage_place["ox"], stage_place["oy"]))
                            composed.append(Image.composite(preserve, gen, keep_mask))
                        stage_out = composed

                        # Intermediate seam polish keeps earlier stage joins from accumulating artifacts.
                        if (
                            refine_enabled
                            and refine_each_step
                            and not is_final
                            and len(stage_out) > 0
                        ):
                            stage_refine_steps = max(6, int(stage_steps * 0.30))
                            stage_refine_strength = max(
                                0.10,
                                min(0.40, float(refine_strength_base) * 0.75),
                            )
                            seam_mask = _build_stage_refine_mask(
                                stage_img.size,
                                stage_place,
                                max(8, int(refine_width * 0.8)),
                            )
                            if callback:
                                update_stage(f"Outpaint seam {idx + 1}/{max(1, total_passes - 1)}{img_label}")
                            refine_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + 7000 + idx)
                            refine_kwargs = {
                                "prompt": prompt,
                                "prompt_2": prompt_2 or None,
                                "negative_prompt": negative_prompt,
                                "image": stage_out[0].convert("RGB"),
                                "mask_image": seam_mask,
                                "strength": stage_refine_strength,
                                "guidance_scale": guidance_scale,
                                "num_inference_steps": stage_refine_steps,
                                "num_images_per_prompt": 1,
                                "width": stage_img.size[0],
                                "height": stage_img.size[1],
                                "generator": refine_gen,
                            }
                            if cb:
                                refine_kwargs["callback_on_step_end"] = cb
                                refine_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                            stage_refined = base_inpaint(**refine_kwargs).images[0]
                            if callback and hasattr(callback, "finish_pass"):
                                callback.finish_pass(stage_refine_steps)
                            stage_out[0] = stage_refined

                        if is_final:
                            staged_image = stage_out[0]
                        else:
                            curr_image = stage_out[0]
                            p_curr = p_next

                    if staged_image is not None:
                        staged_out.append(staged_image)

                out = staged_out

        if out is None:
            if callback:
                update_stage("Inpaint pass 1/1")
            kwargs = {
                "prompt": prompt,
                "prompt_2": prompt_2 or None,
                "negative_prompt": negative_prompt,
                "image": image,
                "mask_image": mask_image,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "width": int(settings.get("width", 1024)),
                "height": int(settings.get("height", 1024)),
                "generator": generator,
            }

            if cb:
                kwargs["callback_on_step_end"] = cb
                kwargs["callback_on_step_end_tensor_inputs"] = ['latents']

            out = base_inpaint(
                **kwargs,
            ).images

        # Smart extend should preserve source identity while avoiding hard box seams.
        # We only do a light post-refine seam preserve blend to avoid wide halo artifacts.
        if settings.get("smart_extend") and settings.get("input_image") is not None:
            src = settings["input_image"]
            mask = settings.get("mask_image")
            source_box = settings.get("smart_extend_source_box") or {}
            feather = int(settings.get("smart_extend_feather", 8))
            seam_width = int(settings.get("smart_extend_refine_width", 24))

            # Optional second pass: seam-only harmonization for cleaner joins.
            if bool(settings.get("smart_extend_refine", True)):
                if callback:
                    update_stage("Seam refine pass 1/1")
                seam_width_base = int(settings.get("smart_extend_refine_width", 24))
                feather = int(settings.get("smart_extend_feather", 8))
                # Manual seam-fix works because it repaints a slightly wider seam region
                # with meaningful denoise. Mirror that behavior automatically.
                seam_width = max(seam_width_base, int(feather * 2 + 16))
                seam_mask = _build_smart_extend_seam_mask(src.size, source_box, seam_width)
                # Soft mask edges reduce visible hard transition lines.
                seam_blur = max(4, min(20, int(max(feather * 1.2, seam_width * 0.35))))
                seam_mask = seam_mask.filter(ImageFilter.GaussianBlur(radius=seam_blur))
                refine_strength = float(settings.get("smart_extend_refine_strength", 0.28))
                # Auto-elevate seam cleanup strength so it consistently removes boundary artifacts.
                # User slider remains a floor; smart-extend denoise influences upper bound.
                target_cleanup_strength = max(
                    refine_strength,
                    min(0.75, max(0.50, float(strength) * 0.70))
                )
                refine_strength = max(0.12, min(0.75, target_cleanup_strength))
                refine_steps = max(10, int(num_inference_steps * 0.45))

                refined = []
                base_seed = generator.initial_seed()
                for idx, img in enumerate(out):
                    local_gen = torch.Generator(base_inpaint.device).manual_seed(base_seed + 1000 + idx)
                    refine_kwargs = dict(
                        prompt=prompt,
                        prompt_2=prompt_2 or None,
                        negative_prompt=negative_prompt,
                        image=img.convert("RGB"),
                        mask_image=seam_mask,
                        strength=refine_strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=refine_steps,
                        num_images_per_prompt=1,
                        width=int(settings.get("width", 1024)),
                        height=int(settings.get("height", 1024)),
                        generator=local_gen,
                    )
                    if cb:
                        refine_kwargs["callback_on_step_end"] = cb
                        refine_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                    res = base_inpaint(**refine_kwargs).images[0]
                    if callback and hasattr(callback, "finish_pass"):
                        callback.finish_pass(refine_steps)
                    refined.append(res)
                out = refined

                # Reintroduce source softly so seams blend without hard box restamping.
                if mask is not None:
                    try:
                        out = [
                            _smart_extend_preserve_blend(
                                src,
                                img,
                                source_box,
                                feather_px=feather,
                                seam_width=seam_width,
                                preserve_weight=0.35,
                            )
                            for img in out
                        ]
                    except Exception as e:
                        log.warning("Smart-extend post-refine preserve composite skipped: %s", e)

        return out, generator.initial_seed()
