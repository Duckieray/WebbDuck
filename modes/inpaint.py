"""Inpainting generation mode."""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from .base import GenerationMode
from . import outpaint as pyramid_outpaint
from webbduck.server.state import update_stage

log = logging.getLogger(__name__)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _smart_extend_debug_enabled(settings: dict) -> bool:
    default_flag = os.getenv("WEBBDUCK_SMART_EXTEND_DEBUG", "1")
    return _truthy(settings.get("smart_extend_debug", default_flag))


def _init_smart_extend_debug_session(settings, base_seed, source_box, final_size, fractions):
    if not _smart_extend_debug_enabled(settings):
        return None
    root = Path(os.getenv("WEBBDUCK_SMART_EXTEND_DEBUG_DIR", "outputs/_debug"))
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_seed{base_seed}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "smart_extend_debug",
        "created_at": time.time(),
        "run_id": run_id,
        "seed": int(base_seed),
        "source_box": {
            "x": int(source_box.get("x", 0)),
            "y": int(source_box.get("y", 0)),
            "w": int(source_box.get("w", 0)),
            "h": int(source_box.get("h", 0)),
        },
        "target_size": [int(final_size[0]), int(final_size[1])],
        "fractions": [float(x) for x in fractions],
        "settings": {
            "steps": int(settings.get("steps", 30)),
            "strength": float(settings.get("strength", 0.99)),
            "cfg": float(settings.get("cfg", 7.5)),
            "smart_extend_feather": int(settings.get("smart_extend_feather", 8)),
            "smart_extend_step_growth": float(settings.get("smart_extend_step_growth", 1.25)),
            "smart_extend_refine": bool(settings.get("smart_extend_refine", True)),
            "smart_extend_refine_each_step": bool(settings.get("smart_extend_refine_each_step", True)),
            "smart_extend_refine_width": int(settings.get("smart_extend_refine_width", 24)),
            "smart_extend_refine_strength": float(settings.get("smart_extend_refine_strength", 0.28)),
        },
        "events": [],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dir": run_dir, "meta": meta}


def _debug_save_image(debug_session, relative_name, image):
    if not debug_session:
        return
    try:
        path = debug_session["dir"] / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    except Exception as e:
        log.warning("Smart-extend debug image save failed (%s): %s", relative_name, e)


def _debug_log_event(debug_session, event):
    if not debug_session:
        return
    try:
        meta = debug_session["meta"]
        events = meta.setdefault("events", [])
        events.append(event)
        (debug_session["dir"] / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Smart-extend debug meta write failed: %s", e)


def _debug_mask_leak_stats(base_image, updated_image, mask_image):
    base = np.array(base_image.convert("RGB"), dtype=np.int16)
    updated = np.array(updated_image.convert("RGB"), dtype=np.int16)
    mask = np.array(mask_image.convert("L"), dtype=np.uint8)
    diff = np.abs(updated - base).astype(np.float32)
    diff_map = np.max(diff, axis=2)
    outside = mask <= 1
    outside_count = int(np.count_nonzero(outside))
    if outside_count <= 0:
        return {"outside_px": 0, "outside_diff_mean": 0.0, "outside_diff_max": 0.0}
    outside_vals = diff_map[outside]
    return {
        "outside_px": outside_count,
        "outside_diff_mean": float(np.mean(outside_vals)),
        "outside_diff_max": float(np.max(outside_vals)),
    }


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

    # Very large growth jumps increase semantic drift/duplication on later passes.
    # Keep staged expansion conservative for continuity.
    growth = max(1.05, min(1.50, float(growth)))
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


def _auto_repeat_passes(src_w, src_h, dst_w, dst_h):
    """Auto-select non-auto-step repeat passes from expansion ratio."""
    if src_w <= 0 or src_h <= 0:
        return 1
    scale_w = float(dst_w) / float(src_w)
    scale_h = float(dst_h) / float(src_h)
    scale = max(scale_w, scale_h)
    if scale <= 1.20:
        return 1
    if scale <= 1.75:
        return 2
    if scale <= 2.20:
        return 2
    if scale <= 2.80:
        return 3
    return 4


def _intermediate_progress_for_large_scale(src_w, src_h, dst_w, dst_h, dim_ratio):
    """Compute two-stage first pass progress using a target fraction of final dimensions."""
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    dst_w = max(src_w, int(dst_w))
    dst_h = max(src_h, int(dst_h))
    target_ratio = max(0.45, min(0.85, float(dim_ratio)))
    if target_ratio >= 0.999:
        return 1.0
    area_src = float(src_w) * float(src_h)
    area_dst = float(dst_w) * float(dst_h)
    target_area = max(area_src + 1.0, min(area_dst - 1.0, area_dst * target_ratio * target_ratio))

    dw = float(dst_w - src_w)
    dh = float(dst_h - src_h)
    if abs(dw) < 1e-6 and abs(dh) < 1e-6:
        return 1.0

    # Solve (src_w + dw*p) * (src_h + dh*p) = target_area for p in [0,1].
    a = dw * dh
    b = (float(src_w) * dh) + (float(src_h) * dw)
    c = area_src - target_area
    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            p = 1.0
        else:
            p = -c / b
    else:
        disc = (b * b) - (4.0 * a * c)
        disc = max(0.0, disc)
        sqrt_disc = disc ** 0.5
        p1 = (-b + sqrt_disc) / (2.0 * a)
        p2 = (-b - sqrt_disc) / (2.0 * a)
        candidates = [x for x in (p1, p2) if 0.0 <= x <= 1.0]
        p = candidates[0] if candidates else p1
    return max(0.30, min(0.90, float(p)))


def _build_stage_canvas_and_mask(
    current_image,
    p_curr,
    p_next,
    src_w,
    src_h,
    dst_w,
    dst_h,
    left,
    top,
    feather,
    stage_mask_blur=20.0,
    initializer="edge_strips",
):
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

    if str(initializer).lower() == "soft_context":
        # Softer context init prevents hard strip artifacts on very large chunk jumps.
        context = ImageOps.fit(current_image, (stage_w, stage_h), method=Image.LANCZOS)
        blur_r = max(18, min(120, int(min(stage_w, stage_h) * 0.06)))
        stage_image = context.filter(ImageFilter.GaussianBlur(radius=blur_r))
        stage_image.paste(current_image, (ox, oy))
    else:
        # Edge-strip extension initializer (from outpaint_retry logic):
        # seed new canvas with current image border pixels, not global blur.
        stage_image = Image.new("RGB", (stage_w, stage_h), current_image.getpixel((curr_w // 2, curr_h // 2)))
        stage_image.paste(current_image, (ox, oy))

        left_added = ox
        right_added = max(0, stage_w - (ox + curr_w))
        top_added = oy
        bottom_added = max(0, stage_h - (oy + curr_h))

        if left_added > 0:
            left_strip = current_image.crop((0, 0, 1, curr_h)).resize((left_added, curr_h), Image.BILINEAR)
            stage_image.paste(left_strip, (0, oy))
        if right_added > 0:
            right_strip = current_image.crop((curr_w - 1, 0, curr_w, curr_h)).resize((right_added, curr_h), Image.BILINEAR)
            stage_image.paste(right_strip, (ox + curr_w, oy))
        if top_added > 0:
            top_strip = current_image.crop((0, 0, curr_w, 1)).resize((curr_w, top_added), Image.BILINEAR)
            stage_image.paste(top_strip, (ox, 0))
        if bottom_added > 0:
            bottom_strip = current_image.crop((0, curr_h - 1, curr_w, curr_h)).resize((curr_w, bottom_added), Image.BILINEAR)
            stage_image.paste(bottom_strip, (ox, oy + curr_h))

        # Corner fills for diagonal growth.
        if left_added > 0 and top_added > 0:
            stage_image.paste(Image.new("RGB", (left_added, top_added), current_image.getpixel((0, 0))), (0, 0))
        if right_added > 0 and top_added > 0:
            stage_image.paste(
                Image.new("RGB", (right_added, top_added), current_image.getpixel((curr_w - 1, 0))),
                (ox + curr_w, 0),
            )
        if left_added > 0 and bottom_added > 0:
            stage_image.paste(
                Image.new("RGB", (left_added, bottom_added), current_image.getpixel((0, curr_h - 1))),
                (0, oy + curr_h),
            )
        if right_added > 0 and bottom_added > 0:
            stage_image.paste(
                Image.new("RGB", (right_added, bottom_added), current_image.getpixel((curr_w - 1, curr_h - 1))),
                (ox + curr_w, oy + curr_h),
            )

    placement = {
        "ox": ox,
        "oy": oy,
        "w": curr_w,
        "h": curr_h,
    }
    stage_hard_mask = _build_stage_hard_mask(stage_image.size, placement)
    stage_mask = stage_hard_mask
    # Optional tiny blur to soften transition while preserving strict source lock.
    blur_r = max(0.0, float(stage_mask_blur))
    if blur_r > 0:
        stage_mask = stage_mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
        stage_mask.paste(0, (ox, oy, ox + curr_w, oy + curr_h))
    # Use the same mask for both inpaint and composite to avoid interpretation mismatch.
    return stage_image, stage_mask, stage_mask, placement


def _build_stage_soft_mask(size, placement, feather):
    width, height = size
    ox = int(placement.get("ox", 0))
    oy = int(placement.get("oy", 0))
    w = int(placement.get("w", 0))
    h = int(placement.get("h", 0))
    x0, y0 = ox, oy
    x1, y1 = ox + w, oy + h

    # Start with outside-region repaint, then strictly protect source region.
    mask_arr = np.full((height, width), 255.0, dtype=np.float32)
    if x1 > x0 and y1 > y0:
        mask_arr[y0:y1, x0:x1] = 0.0

    # Feather only in extension areas, never inside source region.
    feather_px = max(4, min(32, int(feather)))
    if feather_px > 0:
        if x0 > 0:
            band = min(x0, feather_px)
            ramp = np.linspace(255.0, 128.0, band, endpoint=True, dtype=np.float32)
            mask_arr[y0:y1, x0 - band:x0] = ramp[None, :]
        if x1 < width:
            band = min(width - x1, feather_px)
            ramp = np.linspace(128.0, 255.0, band, endpoint=True, dtype=np.float32)
            mask_arr[y0:y1, x1:x1 + band] = ramp[None, :]
        if y0 > 0:
            band = min(y0, feather_px)
            ramp = np.linspace(255.0, 128.0, band, endpoint=True, dtype=np.float32)
            mask_arr[y0 - band:y0, x0:x1] = ramp[:, None]
        if y1 < height:
            band = min(height - y1, feather_px)
            ramp = np.linspace(128.0, 255.0, band, endpoint=True, dtype=np.float32)
            mask_arr[y1:y1 + band, x0:x1] = ramp[:, None]

    return Image.fromarray(np.clip(mask_arr, 0, 255).astype(np.uint8), mode="L")


def _build_stage_hard_mask(size, placement):
    width, height = size
    ox = int(placement.get("ox", 0))
    oy = int(placement.get("oy", 0))
    w = int(placement.get("w", 0))
    h = int(placement.get("h", 0))
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, (ox, oy, ox + w, oy + h))
    return mask


def _composite_masked_update(base_image, updated_image, mask_image):
    """Apply diffusion update only where mask allows, preserving baseline elsewhere."""
    return Image.composite(
        updated_image.convert("RGB"),
        base_image.convert("RGB"),
        mask_image.convert("L"),
    )


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


def _build_soft_source_keep_mask(size, source_box, blend_radius, inset_px=0):
    """Build soft keep mask for source region with inward/outward feather."""
    width, height = size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    if w <= 0 or h <= 0:
        return None

    inset = max(0, int(inset_px))
    x0 = max(0, x + inset)
    y0 = max(0, y + inset)
    x1 = min(width, x + w - inset)
    y1 = min(height, y + h - inset)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (x0, y0, x1, y1))
    r = max(0.0, float(blend_radius))
    if r > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
    return mask


def _smart_extend_preserve_blend(
    src,
    generated,
    source_box,
    feather_px=8,
    seam_width=24,
    preserve_weight=1.0,
    preserve_inset_px=0,
):
    """Blend original source back in with a soft source-box keep mask."""
    blend_radius = max(
        10.0,
        min(96.0, max(float(feather_px) * 2.0, float(seam_width) * 2.25)),
    )
    keep_mask = _build_soft_source_keep_mask(
        generated.size,
        source_box,
        blend_radius,
        inset_px=preserve_inset_px,
    )
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
        debug_session = None

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
                refine_final_stage = bool(settings.get("smart_extend_refine_final_stage", True))
                refine_width = int(settings.get("smart_extend_refine_width", 24))
                refine_strength_base = float(settings.get("smart_extend_refine_strength", 0.28))
                stage_mask_blur = float(settings.get("smart_extend_stage_mask_blur", 20.0))
                # Outpaint typically needs stronger CFG for scene/subject continuity.
                stage_guidance_scale = max(guidance_scale, float(settings.get("smart_extend_min_cfg", 10.0)))
                stage_inpaint_strength = max(
                    0.60,
                    min(0.95, float(settings.get("smart_extend_stage_inpaint_strength", 0.80))),
                )
                stage_late_strength_drop = max(
                    0.0,
                    min(0.20, float(settings.get("smart_extend_stage_late_strength_drop", 0.12))),
                )
                stage_negative_prompt = negative_prompt
                if bool(settings.get("smart_extend_anatomy_guard", True)):
                    guard = "elongated limbs, long arms, extra limbs, duplicated body, malformed shoulders"
                    stage_negative_prompt = f"{negative_prompt}, {guard}" if negative_prompt else guard
                fractions = _build_step_fractions(sb_w, sb_h, final_w, final_h, growth)

                base_seed = generator.initial_seed()
                debug_session = _init_smart_extend_debug_session(
                    settings,
                    base_seed,
                    source_box,
                    (final_w, final_h),
                    fractions,
                )
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
                        stage_img, stage_mask, stage_comp_mask, stage_place = _build_stage_canvas_and_mask(
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
                            stage_mask_blur,
                        )
                        pass_dir = f"img_{img_idx + 1:02d}/pass_{idx + 1:02d}"
                        _debug_save_image(debug_session, f"{pass_dir}/00_stage_input.png", stage_img)
                        _debug_save_image(debug_session, f"{pass_dir}/01_stage_mask.png", stage_mask)
                        _debug_save_image(debug_session, f"{pass_dir}/01b_stage_comp_mask.png", stage_comp_mask)
                        # Verify stage mask does not leak into protected source region.
                        s_ox = int(stage_place.get("ox", 0))
                        s_oy = int(stage_place.get("oy", 0))
                        s_w = int(stage_place.get("w", 0))
                        s_h = int(stage_place.get("h", 0))
                        src_band = np.array(stage_mask.convert("L"), dtype=np.uint8)[s_oy:s_oy + s_h, s_ox:s_ox + s_w]
                        leak_max = int(src_band.max()) if src_band.size > 0 else 0
                        if leak_max > 1:
                            log.warning("MASK LEAK in stage pass %s: source max=%s", idx + 1, leak_max)
                        stage_strength_this_pass = max(
                            0.68,
                            min(
                                0.92,
                                stage_inpaint_strength - (stage_late_strength_drop if p_next > 0.85 else 0.0),
                            ),
                        )
                        _debug_log_event(debug_session, {
                            "type": "stage",
                            "img_index": img_idx + 1,
                            "pass_index": idx + 1,
                            "p_curr": float(p_curr),
                            "p_next": float(p_next),
                            "stage_size": [int(stage_img.size[0]), int(stage_img.size[1])],
                            "placement": {
                                "ox": int(stage_place.get("ox", 0)),
                                "oy": int(stage_place.get("oy", 0)),
                                "w": int(stage_place.get("w", 0)),
                                "h": int(stage_place.get("h", 0)),
                            },
                            "stage_strength": float(stage_strength_this_pass),
                            "stage_cfg": float(stage_guidance_scale),
                            "stage_mask_blur": float(stage_mask_blur),
                            "mask_source_max": leak_max,
                            "steps": int(stage_steps),
                        })
                        # Keep stage seed stable across passes for continuity.
                        # Per-pass seed jumps can cause large semantic drift in new regions.
                        local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base)
                        stage_kwargs = {
                            "prompt": prompt,
                            "prompt_2": prompt_2 or None,
                            "negative_prompt": stage_negative_prompt,
                            "image": stage_img,
                            "mask_image": stage_mask,
                            "strength": stage_strength_this_pass,
                            "guidance_scale": stage_guidance_scale,
                            "num_inference_steps": stage_steps,
                            "num_images_per_prompt": 1,
                            "width": stage_img.size[0],
                            "height": stage_img.size[1],
                            "generator": local_gen,
                        }
                        if cb:
                            stage_kwargs["callback_on_step_end"] = cb
                            stage_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']

                        stage_raw = base_inpaint(**stage_kwargs).images[0].convert("RGB")
                        stage_out = [stage_raw]
                        _debug_save_image(debug_session, f"{pass_dir}/02_stage_raw.png", stage_raw)
                        if callback and hasattr(callback, "finish_pass"):
                            callback.finish_pass(stage_steps)

                        # Strict stage chaining from outpaint_retry behavior:
                        # only update pixels selected by stage_mask.
                        stage_composited = Image.composite(stage_raw, stage_img, stage_comp_mask)
                        # Explicitly lock previous stage pixels.
                        stage_composited.paste(curr_image, (stage_place["ox"], stage_place["oy"]))
                        stage_out = [stage_composited]
                        _debug_save_image(debug_session, f"{pass_dir}/03_stage_composited.png", stage_composited)
                        _debug_log_event(debug_session, {
                            "type": "stage_composite",
                            "img_index": img_idx + 1,
                            "pass_index": idx + 1,
                            **_debug_mask_leak_stats(stage_img, stage_composited, stage_comp_mask),
                        })

                        # Intermediate seam polish keeps earlier stage joins from accumulating artifacts.
                        if (
                            refine_enabled
                            and refine_each_step
                            and len(stage_out) > 0
                            and (not is_final or refine_final_stage)
                        ):
                            stage_refine_steps = max(
                                8 if is_final else 6,
                                int(stage_steps * (0.45 if is_final else 0.30)),
                            )
                            stage_refine_strength = max(
                                0.16 if is_final else 0.08,
                                min(
                                    0.24 if is_final else 0.16,
                                    float(refine_strength_base) * (0.72 if is_final else 0.58),
                                ),
                            )
                            seam_mask = _build_stage_refine_mask(
                                stage_img.size,
                                stage_place,
                                max(8, int(refine_width * 0.8)),
                            )
                            if callback:
                                if is_final:
                                    update_stage(f"Outpaint final seam {idx + 1}/{total_passes}{img_label}")
                                else:
                                    update_stage(f"Outpaint seam {idx + 1}/{max(1, total_passes - 1)}{img_label}")
                            refine_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + 7000)
                            refine_kwargs = {
                                "prompt": prompt,
                                "prompt_2": prompt_2 or None,
                                "negative_prompt": stage_negative_prompt,
                                "image": stage_out[0].convert("RGB"),
                                "mask_image": seam_mask,
                                "strength": stage_refine_strength,
                                "guidance_scale": stage_guidance_scale,
                                "num_inference_steps": stage_refine_steps,
                                "num_images_per_prompt": 1,
                                "width": stage_img.size[0],
                                "height": stage_img.size[1],
                                "generator": refine_gen,
                            }
                            if cb:
                                refine_kwargs["callback_on_step_end"] = cb
                                refine_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                            baseline_stage = stage_out[0].convert("RGB")
                            _debug_save_image(debug_session, f"{pass_dir}/04_refine_mask.png", seam_mask)
                            stage_refined = base_inpaint(**refine_kwargs).images[0]
                            _debug_save_image(debug_session, f"{pass_dir}/05_refine_raw.png", stage_refined)
                            if callback and hasattr(callback, "finish_pass"):
                                callback.finish_pass(stage_refine_steps)
                            stage_out[0] = _composite_masked_update(
                                baseline_stage,
                                stage_refined,
                                seam_mask,
                            )
                            _debug_save_image(debug_session, f"{pass_dir}/06_refine_composited.png", stage_out[0])
                            _debug_log_event(debug_session, {
                                "type": "stage_refine_composite",
                                "img_index": img_idx + 1,
                                "pass_index": idx + 1,
                                "refine_steps": int(stage_refine_steps),
                                "refine_strength": float(stage_refine_strength),
                                **_debug_mask_leak_stats(baseline_stage, stage_out[0], seam_mask),
                            })

                        if is_final:
                            staged_image = stage_out[0]
                        else:
                            curr_image = stage_out[0]
                            p_curr = p_next

                    if staged_image is not None:
                        staged_out.append(staged_image)

                out = staged_out

        elif settings.get("smart_extend") and not bool(settings.get("smart_extend_auto_step", True)):
            source_box = settings.get("smart_extend_source_box") or {}
            sb_x = int(source_box.get("x", 0))
            sb_y = int(source_box.get("y", 0))
            sb_w = int(source_box.get("w", 0))
            sb_h = int(source_box.get("h", 0))
            final_w = int(settings.get("width", image.size[0]))
            final_h = int(settings.get("height", image.size[1]))
            src_base_w = sb_w if sb_w > 0 else int(image.size[0])
            src_base_h = sb_h if sb_h > 0 else int(image.size[1])

            repeat_raw = settings.get("smart_extend_repeat_passes", "auto")
            auto_repeat = repeat_raw is None or (isinstance(repeat_raw, str) and repeat_raw.strip().lower() == "auto")
            if auto_repeat:
                repeat_passes = _auto_repeat_passes(src_base_w, src_base_h, final_w, final_h)
            else:
                try:
                    repeat_passes = max(1, int(repeat_raw))
                except Exception:
                    repeat_passes = _auto_repeat_passes(src_base_w, src_base_h, final_w, final_h)

            debug_session = _init_smart_extend_debug_session(
                settings,
                generator.initial_seed(),
                {
                    "x": sb_x,
                    "y": sb_y,
                    "w": src_base_w,
                    "h": src_base_h,
                },
                (final_w, final_h),
                [1.0],
            )
            _debug_log_event(debug_session, {
                "type": "repeat_mode_selected",
                "auto_repeat": bool(auto_repeat),
                "repeat_raw": str(repeat_raw),
                "repeat_passes": int(repeat_passes),
                "source_size": [int(src_base_w), int(src_base_h)],
                "target_size": [int(final_w), int(final_h)],
                "scale_w": float(final_w / max(1, src_base_w)),
                "scale_h": float(final_h / max(1, src_base_h)),
            })

            # Optional pyramid path for very large non-auto outpaints.
            # Kept behind explicit feature flag to avoid surprising behavior changes.
            pyramid_enabled = bool(settings.get("smart_extend_pyramid_enable", False))
            pyramid_ratio_trigger = float(settings.get("smart_extend_pyramid_trigger_ratio", 2.4))
            scale_w = float(final_w) / float(max(1, src_base_w))
            scale_h = float(final_h) / float(max(1, src_base_h))
            pyramid_ratio = max(scale_w, scale_h)
            pyramid_eligible = (
                pyramid_enabled
                and sb_w > 0
                and sb_h > 0
                and pyramid_ratio >= pyramid_ratio_trigger
            )
            if pyramid_eligible:
                _debug_log_event(debug_session, {
                    "type": "pyramid_selected",
                    "enabled": bool(pyramid_enabled),
                    "trigger_ratio": float(pyramid_ratio_trigger),
                    "expansion_ratio": float(pyramid_ratio),
                    "source_box": {"x": int(sb_x), "y": int(sb_y), "w": int(sb_w), "h": int(sb_h)},
                    "target_size": [int(final_w), int(final_h)],
                })
                out_imgs = []
                batch_count = max(1, num_images_per_prompt)
                base_seed = generator.initial_seed()
                source_crop = image.crop((sb_x, sb_y, sb_x + sb_w, sb_y + sb_h)).convert("RGB")
                pyramid_failed = False
                for img_idx in range(batch_count):
                    img_label = f" (img {img_idx + 1}/{batch_count})" if batch_count > 1 else ""
                    if callback:
                        update_stage(f"Pyramid outpaint {img_idx + 1}/{batch_count}{img_label}")
                    local_gen = torch.Generator(base_inpaint.device).manual_seed(base_seed + (img_idx * 1000003))
                    pass_dir = f"img_{img_idx + 1:02d}/pyramid"
                    _debug_save_image(debug_session, f"{pass_dir}/00_source_crop.png", source_crop)
                    try:
                        result = pyramid_outpaint.run_pyramid_outpaint(
                            source_image=source_crop,
                            source_box={"x": sb_x, "y": sb_y, "w": sb_w, "h": sb_h},
                            target_size=(final_w, final_h),
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            base_inpaint=base_inpaint,
                            generator=local_gen,
                            settings=settings,
                            callback=callback,
                            update_stage_fn=update_stage if callback else None,
                            debug_dir=(debug_session["dir"] / f"img_{img_idx + 1:02d}") if debug_session else None,
                        )
                        out_imgs.append(result)
                        _debug_save_image(debug_session, f"{pass_dir}/99_result.png", result)
                        _debug_log_event(debug_session, {
                            "type": "pyramid_complete",
                            "img_index": img_idx + 1,
                            "seed": int(base_seed + (img_idx * 1000003)),
                            "result_size": [int(result.size[0]), int(result.size[1])],
                        })
                    except Exception as e:
                        pyramid_failed = True
                        log.exception("Pyramid outpaint failed for image %s", img_idx + 1)
                        _debug_log_event(debug_session, {
                            "type": "pyramid_failed",
                            "img_index": img_idx + 1,
                            "error": str(e),
                        })
                        break
                if not pyramid_failed and len(out_imgs) == batch_count:
                    out = out_imgs
                else:
                    _debug_log_event(debug_session, {
                        "type": "pyramid_fallback_to_repeat",
                        "reason": "exception_or_incomplete_batch",
                    })

            if out is None and repeat_passes > 1:
                repeat_strength = max(
                    0.60,
                    min(0.95, float(settings.get("smart_extend_repeat_strength", 0.80))),
                )
                repeat_strength_drop = max(
                    0.0,
                    min(0.20, float(settings.get("smart_extend_repeat_strength_drop", 0.08))),
                )
                repeat_cfg = max(8.0, guidance_scale, float(settings.get("smart_extend_repeat_min_cfg", 8.0)))
                repeat_seam_strength = max(
                    0.18,
                    min(0.60, float(settings.get("smart_extend_repeat_seam_strength", 0.42))),
                )
                repeat_seam_cfg = max(
                    7.0,
                    guidance_scale,
                    float(settings.get("smart_extend_repeat_seam_cfg", 7.0)),
                )
                repeat_negative_prompt = negative_prompt
                if bool(settings.get("smart_extend_repeat_anatomy_guard", True)):
                    guard = (
                        "duplicate person, extra person, second body, cloned body, duplicate torso, "
                        "extra breasts, malformed anatomy, stretched limbs, elongated arms"
                    )
                    repeat_negative_prompt = f"{negative_prompt}, {guard}" if negative_prompt else guard
                base_seed = generator.initial_seed()
                scale_w = float(final_w) / float(max(1, src_base_w))
                scale_h = float(final_h) / float(max(1, src_base_h))
                large_scale = max(scale_w, scale_h) >= 2.20
                use_chunked_two_stage = bool(settings.get("smart_extend_repeat_chunked", True)) and large_scale
                intermediate_ratio = float(settings.get("smart_extend_chunk_scale", 0.60))
                intermediate_progress = _intermediate_progress_for_large_scale(
                    src_base_w,
                    src_base_h,
                    final_w,
                    final_h,
                    intermediate_ratio,
                )
                intermediate_stage_w = _round8(
                    src_base_w + (final_w - src_base_w) * intermediate_progress,
                    minimum=src_base_w,
                )
                intermediate_stage_h = _round8(
                    src_base_h + (final_h - src_base_h) * intermediate_progress,
                    minimum=src_base_h,
                )
                # Large full-canvas jumps can become unstable when repeated too many times.
                # Cap auto repeats to a conservative range for huge expansions.
                if auto_repeat and large_scale:
                    repeat_passes = min(repeat_passes, 4)
                if debug_session:
                    debug_session["meta"]["fractions"] = (
                        [float(intermediate_progress), 1.0] + ([1.0] * max(0, repeat_passes - 2))
                        if use_chunked_two_stage
                        else [1.0 for _ in range(repeat_passes)]
                    )
                    _debug_log_event(debug_session, {
                        "type": "repeat_mode_strategy",
                        "strategy": "chunked_two_stage_then_seam" if use_chunked_two_stage else "full_canvas_then_seam",
                        "repeat_seed_initializer": str(settings.get("smart_extend_repeat_seed_initializer", "none")),
                        "repeat_passes": int(repeat_passes),
                        "repeat_strength": float(repeat_strength),
                        "repeat_cfg": float(repeat_cfg),
                        "repeat_seam_strength": float(repeat_seam_strength),
                        "repeat_seam_cfg": float(repeat_seam_cfg),
                        "intermediate_ratio": float(intermediate_ratio),
                        "intermediate_progress": float(intermediate_progress),
                        "intermediate_stage_size": [int(intermediate_stage_w), int(intermediate_stage_h)],
                        "scale_w": float(scale_w),
                        "scale_h": float(scale_h),
                    })

                out_imgs = []
                batch_count = max(1, num_images_per_prompt)
                for img_idx in range(batch_count):
                    image_seed_base = base_seed + (img_idx * 1000003)
                    img_label = f" (img {img_idx + 1}/{batch_count})" if batch_count > 1 else ""
                    curr_full = image.convert("RGB")
                    source_crop = None
                    repeat_seam_mask = None
                    if sb_w > 0 and sb_h > 0:
                        source_crop = image.crop((sb_x, sb_y, sb_x + sb_w, sb_y + sb_h)).convert("RGB")
                        seed_initializer = str(settings.get("smart_extend_repeat_seed_initializer", "none")).strip().lower()
                        if large_scale and seed_initializer in {"edge_strips", "soft_context"}:
                            seed_feather = int(settings.get("smart_extend_feather", 8))
                            seed_blur = float(settings.get("smart_extend_stage_mask_blur", 20.0))
                            # Optional large-scale preseed. Default is off to match small/medium full-canvas behavior.
                            seeded_full, _, _, _ = _build_stage_canvas_and_mask(
                                source_crop,
                                0.0,
                                1.0,
                                src_base_w,
                                src_base_h,
                                final_w,
                                final_h,
                                sb_x,
                                sb_y,
                                seed_feather,
                                seed_blur,
                                seed_initializer,
                            )
                            curr_full = seeded_full
                        seam_w = max(12, int(settings.get("smart_extend_refine_width", 64)))
                        seam_blur = max(4, min(20, int(max(8, seam_w * 0.28))))
                        repeat_seam_mask = _build_smart_extend_seam_mask(
                            curr_full.size,
                            {"x": sb_x, "y": sb_y, "w": sb_w, "h": sb_h},
                            seam_w,
                        ).filter(ImageFilter.GaussianBlur(radius=seam_blur))

                    if use_chunked_two_stage and source_crop is not None and repeat_passes >= 2:
                        # Pass 1: outpaint to intermediate size (~60% of target dims by default).
                        p_curr = 0.0
                        p_next = float(intermediate_progress)
                        stage_img, stage_mask, stage_comp_mask, _ = _build_stage_canvas_and_mask(
                            source_crop,
                            p_curr,
                            p_next,
                            src_base_w,
                            src_base_h,
                            final_w,
                            final_h,
                            sb_x,
                            sb_y,
                            int(settings.get("smart_extend_feather", 8)),
                            float(settings.get("smart_extend_stage_mask_blur", 20.0)),
                            "edge_strips",
                        )
                        if callback:
                            update_stage(f"Outpaint repeat 1/{repeat_passes}{img_label}")
                        pass_dir = f"img_{img_idx + 1:02d}/repeat_full_pass_01"
                        _debug_save_image(debug_session, f"{pass_dir}/00_input.png", stage_img)
                        _debug_save_image(debug_session, f"{pass_dir}/01_mask.png", stage_mask)
                        s1_strength = min(repeat_strength, float(settings.get("smart_extend_large_first_strength", 0.55)))
                        s1_cfg = max(6.5, float(settings.get("smart_extend_large_first_cfg", 7.5)))
                        local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + 1009)
                        pass_kwargs = {
                            "prompt": prompt,
                            "prompt_2": prompt_2 or None,
                            "negative_prompt": repeat_negative_prompt,
                            "image": stage_img,
                            "mask_image": stage_mask,
                            "strength": s1_strength,
                            "guidance_scale": s1_cfg,
                            "num_inference_steps": num_inference_steps,
                            "num_images_per_prompt": 1,
                            "width": stage_img.size[0],
                            "height": stage_img.size[1],
                            "generator": local_gen,
                        }
                        if cb:
                            pass_kwargs["callback_on_step_end"] = cb
                            pass_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                        raw = base_inpaint(**pass_kwargs).images[0].convert("RGB")
                        _debug_save_image(debug_session, f"{pass_dir}/02_raw.png", raw)
                        if callback and hasattr(callback, "finish_pass"):
                            callback.finish_pass(num_inference_steps)
                        curr_full = Image.composite(raw, stage_img, stage_comp_mask)
                        curr_full.paste(source_crop, (int(round(sb_x * p_next)), int(round(sb_y * p_next))))
                        _debug_save_image(debug_session, f"{pass_dir}/03_composited.png", curr_full)
                        _debug_log_event(debug_session, {
                            "type": "repeat_full_composite",
                            "img_index": img_idx + 1,
                            "pass_index": 1,
                            "repeat_passes": int(repeat_passes),
                            "repeat_strength": float(repeat_strength),
                            "pass_strength": float(s1_strength),
                            "pass_mask_mode": "intermediate_outpaint",
                            "repeat_cfg": float(repeat_cfg),
                            "pass_guidance_scale": float(s1_cfg),
                            "stage_size": [int(stage_img.size[0]), int(stage_img.size[1])],
                            **_debug_mask_leak_stats(stage_img, curr_full, stage_comp_mask),
                        })

                        # Pass 2: expand intermediate to full target.
                        p_curr = p_next
                        p_next = 1.0
                        stage_img, stage_mask, stage_comp_mask, _ = _build_stage_canvas_and_mask(
                            curr_full,
                            p_curr,
                            p_next,
                            src_base_w,
                            src_base_h,
                            final_w,
                            final_h,
                            sb_x,
                            sb_y,
                            int(settings.get("smart_extend_feather", 8)),
                            float(settings.get("smart_extend_stage_mask_blur", 20.0)),
                            "edge_strips",
                        )
                        if callback:
                            update_stage(f"Outpaint repeat 2/{repeat_passes}{img_label}")
                        pass_dir = f"img_{img_idx + 1:02d}/repeat_full_pass_02"
                        _debug_save_image(debug_session, f"{pass_dir}/00_input.png", stage_img)
                        _debug_save_image(debug_session, f"{pass_dir}/01_mask.png", stage_mask)
                        s2_strength = max(
                            0.52,
                            min(0.64, float(settings.get("smart_extend_large_second_strength", 0.58))),
                        )
                        s2_cfg = max(8.0, float(settings.get("smart_extend_large_second_cfg", 10.0)))
                        local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + 2018)
                        pass_kwargs = {
                            "prompt": prompt,
                            "prompt_2": prompt_2 or None,
                            "negative_prompt": repeat_negative_prompt,
                            "image": stage_img,
                            "mask_image": stage_mask,
                            "strength": s2_strength,
                            "guidance_scale": s2_cfg,
                            "num_inference_steps": num_inference_steps,
                            "num_images_per_prompt": 1,
                            "width": stage_img.size[0],
                            "height": stage_img.size[1],
                            "generator": local_gen,
                        }
                        if cb:
                            pass_kwargs["callback_on_step_end"] = cb
                            pass_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                        raw = base_inpaint(**pass_kwargs).images[0].convert("RGB")
                        _debug_save_image(debug_session, f"{pass_dir}/02_raw.png", raw)
                        if callback and hasattr(callback, "finish_pass"):
                            callback.finish_pass(num_inference_steps)
                        curr_full = Image.composite(raw, stage_img, stage_comp_mask)
                        if source_crop is not None:
                            curr_full.paste(source_crop, (sb_x, sb_y))
                        _debug_save_image(debug_session, f"{pass_dir}/03_composited.png", curr_full)
                        _debug_log_event(debug_session, {
                            "type": "repeat_full_composite",
                            "img_index": img_idx + 1,
                            "pass_index": 2,
                            "repeat_passes": int(repeat_passes),
                            "repeat_strength": float(repeat_strength),
                            "pass_strength": float(s2_strength),
                            "pass_mask_mode": "chunk_to_final",
                            "repeat_cfg": float(repeat_cfg),
                            "pass_guidance_scale": float(s2_cfg),
                            "stage_size": [int(stage_img.size[0]), int(stage_img.size[1])],
                            **_debug_mask_leak_stats(stage_img, curr_full, stage_comp_mask),
                        })

                        # Remaining passes become seam-only refinement on final canvas.
                        seam_start = 3
                        seam_total = int(repeat_passes)
                        for pass_index in range(seam_start, seam_total + 1):
                            if repeat_seam_mask is None:
                                break
                            if callback:
                                update_stage(f"Outpaint repeat {pass_index}/{repeat_passes}{img_label}")
                            pass_dir = f"img_{img_idx + 1:02d}/repeat_full_pass_{pass_index:02d}"
                            _debug_save_image(debug_session, f"{pass_dir}/00_input.png", curr_full)
                            _debug_save_image(debug_session, f"{pass_dir}/01_mask.png", repeat_seam_mask)
                            seam_strength = max(
                                0.20,
                                min(repeat_seam_strength, repeat_strength - ((pass_index - 1) * repeat_strength_drop)),
                            )
                            seam_cfg = repeat_seam_cfg
                            local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + (pass_index * 1009))
                            pass_kwargs = {
                                "prompt": prompt,
                                "prompt_2": prompt_2 or None,
                                "negative_prompt": repeat_negative_prompt,
                                "image": curr_full,
                                "mask_image": repeat_seam_mask,
                                "strength": seam_strength,
                                "guidance_scale": seam_cfg,
                                "num_inference_steps": num_inference_steps,
                                "num_images_per_prompt": 1,
                                "width": curr_full.size[0],
                                "height": curr_full.size[1],
                                "generator": local_gen,
                            }
                            if cb:
                                pass_kwargs["callback_on_step_end"] = cb
                                pass_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']
                            raw = base_inpaint(**pass_kwargs).images[0].convert("RGB")
                            _debug_save_image(debug_session, f"{pass_dir}/02_raw.png", raw)
                            if callback and hasattr(callback, "finish_pass"):
                                callback.finish_pass(num_inference_steps)
                            composited = Image.composite(raw, curr_full, repeat_seam_mask)
                            _debug_save_image(debug_session, f"{pass_dir}/03_composited.png", composited)
                            _debug_log_event(debug_session, {
                                "type": "repeat_full_composite",
                                "img_index": img_idx + 1,
                                "pass_index": int(pass_index),
                                "repeat_passes": int(repeat_passes),
                                "repeat_strength": float(repeat_strength),
                                "pass_strength": float(seam_strength),
                                "pass_mask_mode": "seam_refine",
                                "repeat_cfg": float(repeat_cfg),
                                "pass_guidance_scale": float(seam_cfg),
                                **_debug_mask_leak_stats(curr_full, composited, repeat_seam_mask),
                            })
                            mask_np = np.array(repeat_seam_mask.convert("L"), dtype=np.uint8)
                            before_np = np.array(curr_full.convert("RGB"), dtype=np.int16)
                            after_np = np.array(composited.convert("RGB"), dtype=np.int16)
                            seam_sel = mask_np > 1
                            if np.any(seam_sel):
                                seam_diff = np.abs(after_np - before_np).max(axis=2)[seam_sel]
                                seam_mean = float(np.mean(seam_diff))
                                _debug_log_event(debug_session, {
                                    "type": "repeat_seam_delta",
                                    "img_index": img_idx + 1,
                                    "pass_index": int(pass_index),
                                    "seam_mean_delta": seam_mean,
                                })
                                curr_full = composited
                                if seam_mean < float(settings.get("smart_extend_repeat_seam_stop_delta", 1.6)):
                                    break
                            else:
                                curr_full = composited
                    else:
                        for idx in range(repeat_passes):
                            if callback:
                                update_stage(f"Outpaint repeat {idx + 1}/{repeat_passes}{img_label}")
                            pass_dir = f"img_{img_idx + 1:02d}/repeat_full_pass_{idx + 1:02d}"
                            _debug_save_image(debug_session, f"{pass_dir}/00_input.png", curr_full)
                            pass_mask_mode = "full_outpaint" if idx == 0 else "seam_refine"
                            pass_mask = mask_image if (idx == 0 or repeat_seam_mask is None) else repeat_seam_mask
                            pass_strength = max(0.58, min(0.90, repeat_strength - (idx * repeat_strength_drop)))
                            pass_guidance_scale = repeat_cfg if idx == 0 else min(repeat_cfg, 7.0)
                            if large_scale and idx == 0:
                                pass_strength = min(pass_strength, float(settings.get("smart_extend_large_first_strength", 0.62)))
                                pass_guidance_scale = min(pass_guidance_scale, float(settings.get("smart_extend_large_first_cfg", 5.5)))
                            if idx > 0 and pass_mask_mode == "seam_refine":
                                pass_strength = max(0.20, min(repeat_seam_strength, pass_strength))
                                pass_guidance_scale = repeat_seam_cfg
                            _debug_save_image(debug_session, f"{pass_dir}/01_mask.png", pass_mask)

                            local_gen = torch.Generator(base_inpaint.device).manual_seed(image_seed_base + (idx * 1009))
                            pass_kwargs = {
                                "prompt": prompt,
                                "prompt_2": prompt_2 or None,
                                "negative_prompt": repeat_negative_prompt,
                                "image": curr_full,
                                "mask_image": pass_mask,
                                "strength": pass_strength,
                                "guidance_scale": pass_guidance_scale,
                                "num_inference_steps": num_inference_steps,
                                "num_images_per_prompt": 1,
                                "width": curr_full.size[0],
                                "height": curr_full.size[1],
                                "generator": local_gen,
                            }
                            if cb:
                                pass_kwargs["callback_on_step_end"] = cb
                                pass_kwargs["callback_on_step_end_tensor_inputs"] = ['latents']

                            raw = base_inpaint(**pass_kwargs).images[0].convert("RGB")
                            _debug_save_image(debug_session, f"{pass_dir}/02_raw.png", raw)
                            if callback and hasattr(callback, "finish_pass"):
                                callback.finish_pass(num_inference_steps)

                            composited = Image.composite(raw, curr_full, pass_mask)
                            if source_crop is not None and idx == 0:
                                composited.paste(source_crop, (sb_x, sb_y))
                            _debug_save_image(debug_session, f"{pass_dir}/03_composited.png", composited)
                            _debug_log_event(debug_session, {
                                "type": "repeat_full_composite",
                                "img_index": img_idx + 1,
                                "pass_index": idx + 1,
                                "repeat_passes": int(repeat_passes),
                                "repeat_strength": float(repeat_strength),
                                "pass_strength": float(pass_strength),
                                "pass_mask_mode": pass_mask_mode,
                                "repeat_cfg": float(repeat_cfg),
                                "pass_guidance_scale": float(pass_guidance_scale),
                                **_debug_mask_leak_stats(curr_full, composited, pass_mask),
                            })

                            if idx > 0 and pass_mask_mode == "seam_refine":
                                mask_np = np.array(pass_mask.convert("L"), dtype=np.uint8)
                                before_np = np.array(curr_full.convert("RGB"), dtype=np.int16)
                                after_np = np.array(composited.convert("RGB"), dtype=np.int16)
                                seam_sel = mask_np > 1
                                if np.any(seam_sel):
                                    seam_diff = np.abs(after_np - before_np).max(axis=2)[seam_sel]
                                    seam_mean = float(np.mean(seam_diff))
                                    _debug_log_event(debug_session, {
                                        "type": "repeat_seam_delta",
                                        "img_index": img_idx + 1,
                                        "pass_index": idx + 1,
                                        "seam_mean_delta": seam_mean,
                                    })
                                    if seam_mean < float(settings.get("smart_extend_repeat_seam_stop_delta", 1.6)):
                                        curr_full = composited
                                        break
                            curr_full = composited

                    out_imgs.append(curr_full)
                out = out_imgs
            else:
                _debug_log_event(debug_session, {
                    "type": "repeat_mode_fallback_single",
                    "reason": "repeat_passes<=1",
                })

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

            # Optional post-pass source-box seam harmonization.
            # For auto-step staged outpaint, this tends to reintroduce old box seams,
            # so keep it off by default unless explicitly enabled.
            run_post_refine = bool(
                settings.get(
                    "smart_extend_post_refine",
                    not bool(settings.get("smart_extend_auto_step", True))
                    and (
                        _auto_repeat_passes(
                            int((settings.get("smart_extend_source_box") or {}).get("w", image.size[0])),
                            int((settings.get("smart_extend_source_box") or {}).get("h", image.size[1])),
                            int(settings.get("width", image.size[0])),
                            int(settings.get("height", image.size[1])),
                        ) <= 1
                    ),
                )
            )
            if bool(settings.get("smart_extend_refine", True)) and run_post_refine:
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
                _debug_save_image(debug_session, "final_refine/00_seam_mask.png", seam_mask)
                refine_strength = float(settings.get("smart_extend_refine_strength", 0.28))
                # Auto-elevate seam cleanup strength so it consistently removes boundary artifacts.
                # User slider remains a floor; smart-extend denoise influences upper bound.
                target_cleanup_strength = max(
                    refine_strength,
                    min(0.75, max(0.50, float(strength) * 0.70))
                )
                if not bool(settings.get("smart_extend_auto_step", True)):
                    target_cleanup_strength = min(target_cleanup_strength, 0.45)
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
                    baseline_img = img.convert("RGB")
                    res = base_inpaint(**refine_kwargs).images[0]
                    _debug_save_image(debug_session, f"final_refine/img_{idx + 1:02d}_01_raw.png", res)
                    if callback and hasattr(callback, "finish_pass"):
                        callback.finish_pass(refine_steps)
                    refined_img = _composite_masked_update(baseline_img, res, seam_mask)
                    _debug_save_image(debug_session, f"final_refine/img_{idx + 1:02d}_02_composited.png", refined_img)
                    _debug_log_event(debug_session, {
                        "type": "final_refine_composite",
                        "img_index": idx + 1,
                        "refine_steps": int(refine_steps),
                        "refine_strength": float(refine_strength),
                        **_debug_mask_leak_stats(baseline_img, refined_img, seam_mask),
                    })
                    refined.append(refined_img)
                out = refined

                # Reintroduce source softly so seams blend without hard box restamping.
                if mask is not None:
                    try:
                        preserve_weight = float(settings.get("smart_extend_preserve_weight", 0.20))
                        preserve_inset = int(
                            settings.get(
                                "smart_extend_preserve_inset",
                                max(12, int(max(seam_width * 0.60, feather * 1.8))),
                            )
                        )
                        out = [
                            _smart_extend_preserve_blend(
                                src,
                                img,
                                source_box,
                                feather_px=feather,
                                seam_width=seam_width,
                                preserve_weight=preserve_weight,
                                preserve_inset_px=preserve_inset,
                            )
                            for img in out
                        ]
                        for idx, img in enumerate(out):
                            _debug_save_image(debug_session, f"final_refine/img_{idx + 1:02d}_03_after_preserve.png", img)
                    except Exception as e:
                        log.warning("Smart-extend post-refine preserve composite skipped: %s", e)

        return out, generator.initial_seed()
