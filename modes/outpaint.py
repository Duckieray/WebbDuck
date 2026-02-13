"""Pyramid downscale-first outpainting helpers.

This module is used by the non-auto smart-extend path in modes/inpaint.py.
"""

import logging
import time
import json
import math
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat
import torch
from webbduck.prompt.experimental import build_sdxl_conditioning_dispatch
from webbduck.prompt.conditioning import inject_prompt_conditioning_kwargs

log = logging.getLogger(__name__)


def _round8(value, minimum=8):
    return max(int(minimum), int(round(float(value) / 8.0) * 8))


def _build_step_fractions(src_w, src_h, dst_w, dst_h, growth):
    if dst_w <= src_w and dst_h <= src_h:
        return [1.0]
    # Large expansions are more stable with smaller growth jumps per pass.
    expansion = max(
        float(dst_w) / max(1.0, float(src_w)),
        float(dst_h) / max(1.0, float(src_h)),
    )
    growth_cap = 1.30 if expansion >= 2.4 else 1.45
    growth = max(1.05, min(growth_cap, float(growth)))
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


def _calculate_pyramid_stages(
    src_w,
    src_h,
    dst_w,
    dst_h,
    max_gen_size=2048,
    min_target_long_ratio=0.68,
):
    """Calculate generation/downscale/upscale stages for large outpaint."""
    src_fits = max(src_w, src_h) <= max_gen_size
    target_huge = max(dst_w, dst_h) > int(max_gen_size * 1.8)

    if src_fits and not target_huge:
        return {
            "needs_pyramid": False,
            "downscale_factor": 1.0,
            "gen_src_size": (src_w, src_h),
            "gen_target_size": (dst_w, dst_h),
            "upscale_factor": 1.0,
            "stages": ["outpaint"],
        }

    target_gen_size = int(max_gen_size * 0.85)
    downscale_factor = min(1.0, float(target_gen_size) / max(1.0, float(max(src_w, src_h))))

    gen_src_w = _round8(src_w * downscale_factor, minimum=64)
    gen_src_h = _round8(src_h * downscale_factor, minimum=64)

    expansion_w = float(dst_w) / max(1.0, float(src_w))
    expansion_h = float(dst_h) / max(1.0, float(src_h))
    gen_dst_w = _round8(gen_src_w * expansion_w, minimum=gen_src_w)
    gen_dst_h = _round8(gen_src_h * expansion_h, minimum=gen_src_h)

    if max(gen_dst_w, gen_dst_h) > max_gen_size:
        gen_scale = float(max_gen_size) / max(1.0, float(max(gen_dst_w, gen_dst_h)))
        # Keep source and target in the same resized coordinate system.
        gen_src_w = _round8(gen_src_w * gen_scale, minimum=64)
        gen_src_h = _round8(gen_src_h * gen_scale, minimum=64)
        gen_dst_w = _round8(gen_dst_w * gen_scale, minimum=gen_src_w)
        gen_dst_h = _round8(gen_dst_h * gen_scale, minimum=gen_src_h)
        downscale_factor *= gen_scale

    # Prevent overly tiny working targets for very large jobs.
    min_target_long_ratio = max(0.45, min(0.95, float(min_target_long_ratio)))
    dst_long = max(int(dst_w), int(dst_h))
    gen_long = max(int(gen_dst_w), int(gen_dst_h))
    min_gen_long = _round8(dst_long * min_target_long_ratio, minimum=512)
    if gen_long < min_gen_long:
        up = float(min_gen_long) / max(1.0, float(gen_long))
        gen_src_w = _round8(gen_src_w * up, minimum=64)
        gen_src_h = _round8(gen_src_h * up, minimum=64)
        gen_dst_w = _round8(gen_dst_w * up, minimum=gen_src_w)
        gen_dst_h = _round8(gen_dst_h * up, minimum=gen_src_h)
        # Never exceed the true target size.
        gen_dst_w = min(int(dst_w), gen_dst_w)
        gen_dst_h = min(int(dst_h), gen_dst_h)

    upscale_factor = max(
        float(dst_w) / max(1.0, float(gen_dst_w)),
        float(dst_h) / max(1.0, float(gen_dst_h)),
    )

    stages = []
    if downscale_factor < 0.999:
        stages.append("downscale")
    stages.append("outpaint")
    if upscale_factor > 1.001:
        stages.append("upscale")
    stages.append("refine")

    return {
        "needs_pyramid": True,
        "downscale_factor": downscale_factor,
        "gen_src_size": (gen_src_w, gen_src_h),
        "gen_target_size": (gen_dst_w, gen_dst_h),
        "upscale_factor": upscale_factor,
        "stages": stages,
    }


def _upscale_image(image, target_size, method="lanczos"):
    if method == "lanczos":
        return image.resize(target_size, Image.LANCZOS)
    if method == "bicubic":
        return image.resize(target_size, Image.BICUBIC)
    return image.resize(target_size, Image.NEAREST)


def _ensure_min_steps_for_strength(num_steps, strength):
    """Diffusers inpaint can collapse to 0 effective steps when strength is too low."""
    steps = max(1, int(num_steps))
    s = float(strength)
    if s <= 0.0:
        return steps
    # Keep at least one effective denoise step after strength scaling.
    needed = int(math.ceil(1.0 / s))
    return max(steps, needed)


def _build_stage_seam_mask(size, source_box, seam_width):
    width, height = size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    seam = max(8, int(seam_width))
    outer = max(seam, int(round(seam * 1.2)))
    inner = max(8, int(round(seam * 0.85)))
    mask = Image.new("L", (width, height), 0)
    if x > 0:
        mask.paste(255, (max(0, x - outer), y, min(width, x + inner), y + h))
    if (x + w) < width:
        mask.paste(255, (max(0, x + w - inner), y, min(width, x + w + outer), y + h))
    if y > 0:
        mask.paste(255, (x, max(0, y - outer), x + w, min(height, y + inner)))
    if (y + h) < height:
        mask.paste(255, (x, max(0, y + h - inner), x + w, min(height, y + h + outer)))
    blur_r = max(4, int(seam * 0.30))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_r))


def _save_debug_image(debug_dir, relative_name, image):
    if not debug_dir:
        return
    try:
        p = Path(debug_dir) / relative_name
        p.parent.mkdir(parents=True, exist_ok=True)
        image.save(p)
    except Exception:
        pass


def _append_debug_event(debug_dir, event):
    if not debug_dir:
        return
    try:
        p = Path(debug_dir) / "pyramid" / "events.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(event)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _edge_mean_color(image):
    src_w, src_h = image.size
    top = image.crop((0, 0, src_w, 1))
    bottom = image.crop((0, src_h - 1, src_w, src_h))
    left = image.crop((0, 0, 1, src_h)).resize((src_h, 1), Image.BILINEAR)
    right = image.crop((src_w - 1, 0, src_w, src_h)).resize((src_h, 1), Image.BILINEAR)
    border_strip = Image.new("RGB", (src_w * 2 + src_h * 2, 1))
    cursor = 0
    for seg in (top, bottom, left, right):
        w = seg.width
        border_strip.paste(seg, (cursor, 0))
        cursor += w
    mean = ImageStat.Stat(border_strip).mean
    return tuple(int(max(0, min(255, round(v)))) for v in mean[:3])


def _resolve_canvas_initializer(initializer, left_added, right_added, top_added, bottom_added):
    mode = str(initializer or "auto").strip().lower()
    if mode not in {"auto", "edge_strips", "soft_context"}:
        mode = "auto"
    if mode == "auto":
        max_added = max(int(left_added), int(right_added), int(top_added), int(bottom_added))
        mode = "soft_context" if max_added >= 96 else "edge_strips"
    return mode


def _build_pyramid_canvas(source_image, source_box, target_size, initializer="auto"):
    """Initialize stage canvas with adaptive seeding for large staged outpaint.

    - edge_strips: preserves edge continuity for small growth steps
    - soft_context: avoids long stripe artifacts on large growth steps
    """
    width, height = target_size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    src_w, src_h = source_image.size
    x = max(0, min(width - src_w, x))
    y = max(0, min(height - src_h, y))
    left_added = x
    right_added = max(0, width - (x + src_w))
    top_added = y
    bottom_added = max(0, height - (y + src_h))

    mode = _resolve_canvas_initializer(initializer, left_added, right_added, top_added, bottom_added)

    if mode == "soft_context":
        fill_rgb = _edge_mean_color(source_image)
        flat = Image.new("RGB", target_size, fill_rgb)
        context = source_image.resize((width, height), Image.LANCZOS)
        blur_r = max(16, min(140, int(min(width, height) * 0.08)))
        context = context.filter(ImageFilter.GaussianBlur(radius=blur_r))
        canvas = Image.blend(flat, context, alpha=0.58)
        canvas.paste(source_image, (x, y))
        return canvas

    canvas = Image.new("RGB", target_size, source_image.getpixel((src_w // 2, src_h // 2)))
    canvas.paste(source_image, (x, y))
    if left_added > 0:
        left_strip = source_image.crop((0, 0, 1, src_h)).resize((left_added, src_h), Image.BILINEAR)
        canvas.paste(left_strip, (0, y))
    if right_added > 0:
        right_strip = source_image.crop((src_w - 1, 0, src_w, src_h)).resize((right_added, src_h), Image.BILINEAR)
        canvas.paste(right_strip, (x + src_w, y))
    if top_added > 0:
        top_strip = source_image.crop((0, 0, src_w, 1)).resize((src_w, top_added), Image.BILINEAR)
        canvas.paste(top_strip, (x, 0))
    if bottom_added > 0:
        bottom_strip = source_image.crop((0, src_h - 1, src_w, src_h)).resize((src_w, bottom_added), Image.BILINEAR)
        canvas.paste(bottom_strip, (x, y + src_h))
    if left_added > 0 and top_added > 0:
        canvas.paste(Image.new("RGB", (left_added, top_added), source_image.getpixel((0, 0))), (0, 0))
    if right_added > 0 and top_added > 0:
        canvas.paste(
            Image.new("RGB", (right_added, top_added), source_image.getpixel((src_w - 1, 0))),
            (x + src_w, 0),
        )
    if left_added > 0 and bottom_added > 0:
        canvas.paste(
            Image.new("RGB", (left_added, bottom_added), source_image.getpixel((0, src_h - 1))),
            (0, y + src_h),
        )
    if right_added > 0 and bottom_added > 0:
        canvas.paste(
            Image.new("RGB", (right_added, bottom_added), source_image.getpixel((src_w - 1, src_h - 1))),
            (x + src_w, y + src_h),
        )
    return canvas


def _soft_source_reintroduce(base_image, source_image, source_box, blur_radius=28, weight=0.92):
    width, height = base_image.size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    if w <= 0 or h <= 0:
        return base_image
    composed = base_image.copy()
    composed.paste(source_image, (x, y))
    keep = Image.new("L", (width, height), 0)
    keep.paste(255, (x, y, x + w, y + h))
    keep = keep.filter(ImageFilter.GaussianBlur(radius=max(0, int(blur_radius))))
    wgt = max(0.0, min(1.0, float(weight)))
    if wgt < 0.999:
        keep = keep.point(lambda v: int(v * wgt))
    return Image.composite(composed, base_image, keep)


def _build_refine_seam_mask(size, source_box, seam_width):
    width, height = size
    x = int(source_box.get("x", 0))
    y = int(source_box.get("y", 0))
    w = int(source_box.get("w", 0))
    h = int(source_box.get("h", 0))
    seam = max(8, int(seam_width))
    seam_out = max(seam, int(round(seam * 1.2)))
    seam_in = max(8, int(round(seam * 0.8)))
    mask = Image.new("L", (width, height), 0)
    if x > 0:
        mask.paste(255, (max(0, x - seam_out), y, min(width, x + seam_in), y + h))
    if (x + w) < width:
        mask.paste(255, (max(0, x + w - seam_in), y, min(width, x + w + seam_out), y + h))
    if y > 0:
        mask.paste(255, (x, max(0, y - seam_out), x + w, min(height, y + seam_in)))
    if (y + h) < height:
        mask.paste(255, (x, max(0, y + h - seam_in), x + w, min(height, y + h + seam_out)))
    return mask.filter(ImageFilter.GaussianBlur(radius=max(4, int(seam * 0.25))))


def _build_stage_hard_mask(size, placement):
    width, height = size
    ox = int(placement.get("x", 0))
    oy = int(placement.get("y", 0))
    w = int(placement.get("w", 0))
    h = int(placement.get("h", 0))
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, (ox, oy, ox + w, oy + h))
    return mask


def _build_progressive_stage(
    current_image,
    p_curr,
    p_next,
    src_w,
    src_h,
    dst_w,
    dst_h,
    final_x,
    final_y,
    stage_mask_blur=20.0,
    initializer="edge_strips",
):
    if p_next >= 0.999:
        stage_w, stage_h = dst_w, dst_h
    else:
        stage_w = _round8(src_w + (dst_w - src_w) * p_next, minimum=src_w)
        stage_h = _round8(src_h + (dst_h - src_h) * p_next, minimum=src_h)

    curr_w, curr_h = current_image.size
    nx = int(round(final_x * p_next))
    ny = int(round(final_y * p_next))
    cx = int(round(final_x * p_curr))
    cy = int(round(final_y * p_curr))
    ox = max(0, min(stage_w - curr_w, nx - cx))
    oy = max(0, min(stage_h - curr_h, ny - cy))
    placement = {"x": int(ox), "y": int(oy), "w": int(curr_w), "h": int(curr_h)}

    resolved_init = _resolve_canvas_initializer(
        initializer,
        placement["x"],
        max(0, stage_w - (placement["x"] + placement["w"])),
        placement["y"],
        max(0, stage_h - (placement["y"] + placement["h"])),
    )
    stage_image = _build_pyramid_canvas(
        current_image,
        placement,
        (stage_w, stage_h),
        initializer=resolved_init,
    )
    stage_mask = _build_stage_hard_mask((stage_w, stage_h), placement)
    blur_r = max(0.0, float(stage_mask_blur))
    if blur_r > 0:
        stage_mask = stage_mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
        stage_mask.paste(0, (ox, oy, ox + curr_w, oy + curr_h))
    return stage_image, stage_mask, placement, resolved_init


def run_pyramid_outpaint(
    source_image,
    source_box,
    target_size,
    prompt,
    negative_prompt,
    base_inpaint,
    generator,
    settings,
    callback=None,
    update_stage_fn=None,
    debug_dir=None,
):
    """Execute pyramid outpainting with strict source lock and debug dumps."""
    src_w, src_h = source_image.size
    dst_w, dst_h = target_size
    max_gen_size = int(settings.get("smart_extend_pyramid_max_gen_size", 2048))
    min_target_ratio = float(settings.get("smart_extend_pyramid_min_target_ratio", 0.68))
    pyramid = _calculate_pyramid_stages(
        src_w,
        src_h,
        dst_w,
        dst_h,
        max_gen_size=max_gen_size,
        min_target_long_ratio=min_target_ratio,
    )

    steps = int(settings.get("steps", 30))
    prompt_2 = settings.get("prompt_2", "")
    cfg = max(7.0, min(10.5, float(settings.get("smart_extend_pyramid_cfg_stage1", 8.5))))
    strength = max(0.64, min(0.92, float(settings.get("smart_extend_pyramid_strength_stage1", 0.82))))
    feather = int(settings.get("smart_extend_feather", 12))
    stage_mask_blur = float(settings.get("smart_extend_stage_mask_blur", 20.0))

    guard = (
        "duplicate person, multiple people, second body, cloned body, duplicate torso, "
        "extra breasts, malformed anatomy, stretched limbs, elongated arms, extra limbs, "
        "framed picture, picture frame, poster, painting, picture-in-picture, nested image"
    )
    full_negative = f"{negative_prompt}, {guard}" if negative_prompt else guard

    cb = callback.get_callback() if callback else None
    current_image = source_image.convert("RGB")
    seed = generator.initial_seed()
    conditioning_cache = {}

    if "downscale" in pyramid["stages"]:
        if update_stage_fn:
            update_stage_fn("Pyramid: downscaling source")
        gen_src_w, gen_src_h = pyramid["gen_src_size"]
        current_image = _upscale_image(source_image, (gen_src_w, gen_src_h), method="lanczos")

    _save_debug_image(debug_dir, "pyramid/00_source.png", source_image)
    _save_debug_image(debug_dir, "pyramid/01_working_source.png", current_image)
    _append_debug_event(
        debug_dir,
        {
            "type": "pyramid_setup",
            "source_size": [int(src_w), int(src_h)],
            "target_size": [int(dst_w), int(dst_h)],
            "max_gen_size": int(max_gen_size),
            "min_target_ratio": float(min_target_ratio),
            "plan": pyramid,
        },
    )

    gen_dst_w, gen_dst_h = pyramid["gen_target_size"]
    scale_x = float(gen_dst_w) / max(1.0, float(dst_w))
    scale_y = float(gen_dst_h) / max(1.0, float(dst_h))
    src_gen_w = max(8, _round8(float(source_box.get("w", src_w)) * scale_x, minimum=8))
    src_gen_h = max(8, _round8(float(source_box.get("h", src_h)) * scale_y, minimum=8))
    src_gen_w = min(gen_dst_w, src_gen_w)
    src_gen_h = min(gen_dst_h, src_gen_h)
    if current_image.size != (src_gen_w, src_gen_h):
        current_image = _upscale_image(current_image, (src_gen_w, src_gen_h), method="lanczos")

    final_x = max(0, min(gen_dst_w - src_gen_w, int(round(float(source_box.get("x", 0)) * scale_x))))
    final_y = max(0, min(gen_dst_h - src_gen_h, int(round(float(source_box.get("y", 0)) * scale_y))))
    growth = float(settings.get("smart_extend_pyramid_step_growth", settings.get("smart_extend_step_growth", 1.25)))
    fractions = _build_step_fractions(src_gen_w, src_gen_h, gen_dst_w, gen_dst_h, growth)
    total_passes = len(fractions)
    seam_each_step = bool(settings.get("smart_extend_pyramid_seam_each_step", False))

    p_curr = 0.0
    for pass_idx, p_next in enumerate(fractions, start=1):
        pass_start = time.time()
        if update_stage_fn:
            update_stage_fn(f"Pyramid: outpaint pass {pass_idx}/{total_passes}")
        init_mode = settings.get("smart_extend_pyramid_initializer", "edge_strips")
        canvas, mask, lock_box, resolved_init_mode = _build_progressive_stage(
            current_image=current_image,
            p_curr=p_curr,
            p_next=p_next,
            src_w=src_gen_w,
            src_h=src_gen_h,
            dst_w=gen_dst_w,
            dst_h=gen_dst_h,
            final_x=final_x,
            final_y=final_y,
            stage_mask_blur=stage_mask_blur,
            initializer=init_mode,
        )
        stage_w, stage_h = canvas.size
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_10_stage_canvas.png", canvas)
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_11_stage_mask.png", mask)
        # Keep legacy file names for quick inspection in latest pass.
        _save_debug_image(debug_dir, "pyramid/10_stage_canvas.png", canvas)
        _save_debug_image(debug_dir, "pyramid/11_stage_mask.png", mask)

        stage_strength = max(0.64, min(0.92, strength - (0.04 * max(0, pass_idx - 1))))
        stage_cfg = max(7.0, min(10.2, cfg + (0.25 * min(2, max(0, pass_idx - 1)))))
        stage_steps = steps if p_next >= 0.999 else max(14, int(steps * (0.62 + 0.38 * p_next)))

        # Keep the pass seed stable for continuity across staged expansions.
        local_gen = torch.Generator(base_inpaint.device).manual_seed(seed)
        outpaint_kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2 or None,
            "negative_prompt": full_negative,
            "image": canvas,
            "mask_image": mask,
            "strength": stage_strength,
            "guidance_scale": stage_cfg,
            "num_inference_steps": stage_steps,
            "num_images_per_prompt": 1,
            "width": stage_w,
            "height": stage_h,
            "generator": local_gen,
        }
        if cb:
            outpaint_kwargs["callback_on_step_end"] = cb
            outpaint_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        outpainted = base_inpaint(
            **inject_prompt_conditioning_kwargs(
                outpaint_kwargs,
                active_pipe=base_inpaint,
                use_experimental=settings.get("experimental_compress", False),
                builder=build_sdxl_conditioning_dispatch,
                cache=conditioning_cache,
            )
        ).images[0].convert("RGB")
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_12_stage_raw.png", outpainted)
        _save_debug_image(debug_dir, "pyramid/12_stage_raw.png", outpainted)
        if callback and hasattr(callback, "finish_pass"):
            callback.finish_pass(stage_steps)

        # Progressive chaining: only update outside the locked previous stage.
        composed = Image.composite(outpainted, canvas, mask)
        # Seam polish defaults to final stage only to avoid nested interior frame artifacts.
        run_seam = bool(seam_each_step or (pass_idx == total_passes))
        seam_width = int(settings.get("smart_extend_pyramid_stage_seam_width", max(20, int(feather * 2.6))))
        seam_mask = Image.new("L", (stage_w, stage_h), 0)
        seam_steps = 0
        seam_strength = 0.0
        if run_seam:
            seam_mask = _build_stage_seam_mask(
                (stage_w, stage_h),
                lock_box,
                seam_width=seam_width,
            )
            seam_steps = max(8, int(stage_steps * 0.30))
            seam_raw = float(
                settings.get(
                    "smart_extend_pyramid_seam_strength",
                    settings.get("smart_extend_refine_strength", 0.28),
                )
            )
            if seam_raw > 0.0:
                seam_strength = max(0.14, min(0.42, seam_raw))
                seam_steps = _ensure_min_steps_for_strength(seam_steps, seam_strength)
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_14_seam_mask.png", seam_mask)
        if seam_strength > 0.0:
            seam_gen = torch.Generator(base_inpaint.device).manual_seed(seed + 777)
            seam_kwargs = {
                "prompt": prompt,
                "prompt_2": prompt_2 or None,
                "negative_prompt": full_negative,
                "image": composed,
                "mask_image": seam_mask,
                "strength": seam_strength,
                "guidance_scale": max(6.2, min(8.0, stage_cfg - 0.2)),
                "num_inference_steps": seam_steps,
                "num_images_per_prompt": 1,
                "width": stage_w,
                "height": stage_h,
                "generator": seam_gen,
            }
            if cb:
                seam_kwargs["callback_on_step_end"] = cb
                seam_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            seam_raw = base_inpaint(
                **inject_prompt_conditioning_kwargs(
                    seam_kwargs,
                    active_pipe=base_inpaint,
                    use_experimental=settings.get("experimental_compress", False),
                    builder=build_sdxl_conditioning_dispatch,
                    cache=conditioning_cache,
                )
            ).images[0].convert("RGB")
            _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_15_seam_raw.png", seam_raw)
            composed = Image.composite(seam_raw, composed, seam_mask)
            if callback and hasattr(callback, "finish_pass"):
                callback.finish_pass(seam_steps)
        else:
            _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_15_seam_raw.png", composed)
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_16_seam_composited.png", composed)
        _save_debug_image(debug_dir, f"pyramid/pass_{pass_idx:02d}_13_stage_composited.png", composed)
        _save_debug_image(debug_dir, "pyramid/13_stage_composited.png", composed)
        current_image = composed
        p_curr = p_next
        _append_debug_event(
            debug_dir,
            {
                "type": "pyramid_pass",
                "pass_index": int(pass_idx),
                "total_passes": int(total_passes),
                "stage_size": [int(stage_w), int(stage_h)],
                "place_box": {
                    "x": int(lock_box["x"]),
                    "y": int(lock_box["y"]),
                    "w": int(lock_box["w"]),
                    "h": int(lock_box["h"]),
                },
                "lock_box": {
                    "x": int(lock_box["x"]),
                    "y": int(lock_box["y"]),
                    "w": int(lock_box["w"]),
                    "h": int(lock_box["h"]),
                },
                "progress_fraction": float(p_next),
                "stage_steps": int(stage_steps),
                "seam_steps": int(seam_steps),
                "stage_strength": float(stage_strength),
                "stage_cfg": float(stage_cfg),
                "canvas_initializer": str(resolved_init_mode),
                "seam_width": int(seam_width),
                "seam_strength": float(seam_strength),
                "seam_box_count": 1,
                "seam_applied": bool(seam_strength > 0.0),
                "seam_mode": "each_step" if seam_each_step else "final_only",
                "elapsed_sec": float(time.time() - pass_start),
            },
        )

    if "upscale" in pyramid["stages"]:
        up_start = time.time()
        if update_stage_fn:
            update_stage_fn("Pyramid: upscaling to target resolution")
        current_image = _upscale_image(current_image, target_size, method="lanczos")
        preserve_weight = float(settings.get("smart_extend_pyramid_preserve_weight", 0.08))
        if preserve_weight > 0.001:
            current_image = _soft_source_reintroduce(
                current_image,
                source_image,
                source_box,
                blur_radius=max(20, int(feather * 2.8)),
                weight=preserve_weight,
            )
        _save_debug_image(debug_dir, "pyramid/20_upscaled.png", current_image)
        _append_debug_event(
            debug_dir,
            {
                "type": "pyramid_upscale",
                "size": [int(target_size[0]), int(target_size[1])],
                "elapsed_sec": float(time.time() - up_start),
            },
        )
    elif current_image.size == target_size:
        preserve_weight = float(settings.get("smart_extend_pyramid_preserve_weight", 0.08))
        if preserve_weight > 0.001:
            current_image = _soft_source_reintroduce(
                current_image,
                source_image,
                source_box,
                blur_radius=max(20, int(feather * 2.8)),
                weight=preserve_weight,
            )

    if "refine" in pyramid["stages"] and settings.get("smart_extend_refine", True):
        refine_start = time.time()
        if update_stage_fn:
            update_stage_fn("Pyramid: final refinement")
        seam_width = int(settings.get("smart_extend_refine_width", 48))
        refine_mask = _build_refine_seam_mask(target_size, source_box, seam_width)
        _save_debug_image(debug_dir, "pyramid/30_refine_mask.png", refine_mask)
        refine_strength = float(settings.get("smart_extend_pyramid_refine_strength", settings.get("smart_extend_refine_strength", 0.35)))
        refine_steps = max(10, int(steps * 0.5))
        refine_cfg = max(6.5, min(9.0, float(settings.get("smart_extend_pyramid_cfg_stage2", 8.5))))
        refine_steps = _ensure_min_steps_for_strength(refine_steps, max(0.0, min(0.60, refine_strength)))
        refine_gen = torch.Generator(base_inpaint.device).manual_seed(seed + 9999)
        refine_kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2 or None,
            "negative_prompt": full_negative,
            "image": current_image,
            "mask_image": refine_mask,
            "strength": max(0.12, min(0.60, refine_strength)),
            "guidance_scale": refine_cfg,
            "num_inference_steps": refine_steps,
            "num_images_per_prompt": 1,
            "width": dst_w,
            "height": dst_h,
            "generator": refine_gen,
        }
        if cb:
            refine_kwargs["callback_on_step_end"] = cb
            refine_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        refined = base_inpaint(
            **inject_prompt_conditioning_kwargs(
                refine_kwargs,
                active_pipe=base_inpaint,
                use_experimental=settings.get("experimental_compress", False),
                builder=build_sdxl_conditioning_dispatch,
                cache=conditioning_cache,
            )
        ).images[0].convert("RGB")
        _save_debug_image(debug_dir, "pyramid/31_refine_raw.png", refined)
        if callback and hasattr(callback, "finish_pass"):
            callback.finish_pass(refine_steps)
        current_image = Image.composite(refined, current_image, refine_mask)
        preserve_weight = float(settings.get("smart_extend_pyramid_preserve_weight", 0.08))
        if preserve_weight > 0.001:
            current_image = _soft_source_reintroduce(
                current_image,
                source_image,
                source_box,
                blur_radius=max(20, int(feather * 2.8)),
                weight=preserve_weight,
            )
        _save_debug_image(debug_dir, "pyramid/32_refine_composited.png", current_image)
        _append_debug_event(
            debug_dir,
            {
                "type": "pyramid_refine",
                "refine_steps": int(refine_steps),
                "elapsed_sec": float(time.time() - refine_start),
            },
        )

    return current_image
