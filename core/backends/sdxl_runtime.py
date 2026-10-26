"""Main generation orchestration."""

from pathlib import Path
import torch
import os
import random
import logging
from PIL import Image, ImageOps, ImageStat, ImageFilter

from core.backends.sdxl_pipeline import pipeline_manager
from core.captioner import unload_captioners
from core.exceptions import GenerationCancelledError
from core.image_operations import build_smart_extend_inputs
from core.perf import reset_metrics, snapshot_metrics, stage_timer
from modes import select_mode
from modes.inpaint import _build_step_fractions, _auto_repeat_passes, _effective_refine_steps
from server.state import update_progress

log = logging.getLogger(__name__)


class GlobalProgress:
    """Tracks progress across multiple generation passes."""
    
    def __init__(self, total_estimated_steps, cancel_event=None):
        self.total_estimated_steps = total_estimated_steps
        self.current_global_step = 0
        self.pass_start_step = 0
        self.cancel_event = cancel_event
        
    def get_callback(self):
        """Returns a diffusers-compatible callback."""
        def callback(pipe, step_index, timestep, callback_kwargs):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise GenerationCancelledError("Generation cancelled by user")
            # Calculate actual global step
            # step_index is 0-based index of Current Pass
            
            # Update global counter
            current_pass_step = step_index + 1
            global_step = self.pass_start_step + current_pass_step
            
            # Clamping
            global_step = min(global_step, self.total_estimated_steps)
            
            # Generation occupies the bulk of runtime.
            # Keep loading mostly in 0-40% and reserve only a tiny tail for decode/save.
            # This avoids the UI appearing "90% done" while most work is still running.
            
            gen_progress = global_step / max(1, self.total_estimated_steps)
            
            # Map generation progress (0-1) to UI range (0.40 - 0.97)
            ui_progress = 0.40 + (gen_progress * 0.57)
            
            update_progress(ui_progress, step=global_step, total_steps=self.total_estimated_steps)
            
            return callback_kwargs

        return callback

    def finish_pass(self, steps_completed):
        """Mark a pass as complete and advance the baseline."""
        self.pass_start_step += steps_completed


def estimate_total_steps(settings):
    """Estimate total steps based on settings."""
    steps = int(settings.get("steps", 30))
    batch_count = max(1, int(settings.get("num_images", 1)))
    smart_extend = bool(settings.get("smart_extend"))
    smart_extend_auto_step = bool(settings.get("smart_extend_auto_step", True))
    refine_enabled = bool(settings.get("smart_extend_refine", True))

    def _get_src_dims():
        source_box = settings.get("smart_extend_source_box") or {}
        if source_box.get("w") and source_box.get("h"):
            return int(source_box.get("w")), int(source_box.get("h"))
        if settings.get("input_image") is not None:
            try:
                return settings["input_image"].size
            except Exception:
                return None, None
        return None, None

    def _post_refine_enabled_for_non_auto(src_w, src_h, dst_w, dst_h):
        explicit = settings.get("smart_extend_post_refine")
        if explicit is not None:
            return bool(explicit)
        return _auto_repeat_passes(src_w, src_h, dst_w, dst_h) <= 1

    # Smart-extend multi-pass inpaint can run several denoise passes plus optional seam refine.
    # Estimate these so the progress bar represents staged generation more accurately.
    if smart_extend and smart_extend_auto_step:
        src_w, src_h = _get_src_dims()

        dst_w = int(settings.get("width", 1024))
        dst_h = int(settings.get("height", 1024))
        growth = float(settings.get("smart_extend_step_growth", 1.25))

        if src_w and src_h:
            fractions = _build_step_fractions(src_w, src_h, dst_w, dst_h, growth)
            passes = len(fractions)
            per_image_total = 0
            refine_each = bool(settings.get("smart_extend_refine_each_step", True))
            refine_final_stage = bool(settings.get("smart_extend_refine_final_stage", True))
            for idx, p_next in enumerate(fractions):
                is_final = idx == (passes - 1)
                stage_steps = steps if is_final else max(
                    12,
                    int(steps * (0.55 + 0.45 * float(p_next))),
                )
                per_image_total += stage_steps
                if refine_enabled and refine_each and (not is_final or refine_final_stage):
                    refine_strength = max(
                        0.16 if is_final else 0.08,
                        min(
                            0.24 if is_final else 0.16,
                            float(settings.get("smart_extend_refine_strength", 0.28))
                            * (0.72 if is_final else 0.58),
                        ),
                    )
                    per_image_total += _effective_refine_steps(
                        max(
                            8 if is_final else 6,
                            int(stage_steps * (0.45 if is_final else 0.30)),
                        ),
                        refine_strength,
                    )
            total = per_image_total * batch_count
            if refine_enabled and bool(settings.get("smart_extend_post_refine", False)):
                total += max(10, int(steps * 0.45)) * batch_count
            return total
    elif smart_extend and not smart_extend_auto_step:
        src_w, src_h = _get_src_dims()
        if src_w and src_h:
            dst_w = int(settings.get("width", src_w))
            dst_h = int(settings.get("height", src_h))
            pyramid_enabled = bool(settings.get("smart_extend_pyramid_enable", False))
            pyramid_ratio_trigger = float(settings.get("smart_extend_pyramid_trigger_ratio", 2.4))
            expansion_ratio = max(
                float(dst_w) / max(1.0, float(src_w)),
                float(dst_h) / max(1.0, float(src_h)),
            )
            if pyramid_enabled and expansion_ratio >= pyramid_ratio_trigger:
                # Pyramid path does one outpaint pass plus optional final refine pass.
                per_image_total = steps
                if refine_enabled and bool(settings.get("smart_extend_refine", True)):
                    per_image_total += max(12, int(steps * 0.5))
                return per_image_total * batch_count

            repeat_raw = settings.get("smart_extend_repeat_passes", "auto")
            auto_repeat = repeat_raw is None or (
                isinstance(repeat_raw, str) and repeat_raw.strip().lower() == "auto"
            )
            if auto_repeat:
                repeat_passes = _auto_repeat_passes(src_w, src_h, dst_w, dst_h)
            else:
                try:
                    repeat_passes = max(1, int(repeat_raw))
                except Exception:
                    repeat_passes = _auto_repeat_passes(src_w, src_h, dst_w, dst_h)

            per_image_total = steps * (repeat_passes if repeat_passes > 1 else 1)
            if refine_enabled and _post_refine_enabled_for_non_auto(src_w, src_h, dst_w, dst_h):
                per_image_total += max(10, int(steps * 0.45))
            return per_image_total * batch_count

    second_pass = settings.get("second_pass_model")
    
    if not second_pass or second_pass == "None":
        return steps
        
    # If two pass, we usually have base steps + refiner steps
    # Refiner steps are usuallly (steps * strength) or explicitly set
    # In webbduck/modes/two_pass.py logic:
    # Refiner uses int(steps * 0.7) usually, wait verify...
    
    # Actually, let's look at TwoPassMode.run:
    # Pass 1: steps
    # Pass 2: int(steps * 0.7) usually? 
    # Let's check logic in TwoPassMode again.
    # It uses settings["steps"] for base.
    # And settings["steps"] * 0.7 for refiner.
    
    # We should probably get exact logic from the execution, but estimation is fine.
    
    refine_steps = int(steps * 0.7) # Approximation based on default logic
    
    return steps + refine_steps


def inject_lora_trigger(prompt_text, trigger_phrase):
    """Append LoRA trigger phrase to prompt text if needed."""
    prompt = (prompt_text or "").strip()
    triggers = (trigger_phrase or "").strip()

    if not triggers:
        return prompt

    # Avoid duplicate appends when regenerating from metadata that already includes triggers.
    if triggers in prompt:
        return prompt

    if not prompt:
        return triggers

    return f"{prompt}, {triggers}"


def _normalize_dim8(value, fallback=1024) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = int(fallback)
    return max(8, int(round(float(n) / 8.0) * 8))


def _build_smart_extend_inputs(
    input_image: Image.Image,
    existing_mask: Image.Image | None,
    target_w: int,
    target_h: int,
    anchor: str,
    feather_px: int = 8,
    offset_x: int | None = None,
    offset_y: int | None = None,
    initializer: str = "noise",
) -> tuple[Image.Image, Image.Image, dict]:
    # Kept as a compatibility seam for tests and existing imports. New backends
    # use the architecture-neutral helper directly.
    if input_image.size == (int(target_w), int(target_h)) and existing_mask is not None:
        return input_image, existing_mask, {
            "x": 0,
            "y": 0,
            "w": input_image.width,
            "h": input_image.height,
        }
    return build_smart_extend_inputs(
        input_image,
        target_w,
        target_h,
        anchor,
        feather_px,
        offset_x,
        offset_y,
        initializer,
    )



def run_generation(settings, cancel_event=None):
    """Execute image generation based on settings."""
    reset_metrics()
    settings["width"] = _normalize_dim8(settings.get("width", 1024), fallback=1024)
    settings["height"] = _normalize_dim8(settings.get("height", 1024), fallback=1024)

    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("Generation cancelled before start")

    second_pass_model = settings.get("second_pass_model")
    if second_pass_model == "None":
        second_pass_model = None

    # Unload any captioner models to free VRAM before loading generation pipelines
    unload_captioners()

    with stage_timer("pipeline_get_seconds"):
        pipe, img2img, base_img2img, base_inpaint, trigger_phrase, faceid_kwargs = pipeline_manager.get(
            base_model=settings["base_model"],
            second_pass_model=second_pass_model,
            loras=settings.get("loras", []),
            embeddings=settings.get("embeddings", []),
            scheduler_name=settings.get("scheduler"),
            cancel_event=cancel_event,
            identity_adapter=settings.get("identity_adapter"),
        )

    # Capture identity adapter debug metadata into settings for persistence.
    id_debug = getattr(pipeline_manager, "_identity_debug", None)
    if id_debug:
        settings["identity_adapter_debug"] = dict(id_debug)

    # Always inject LoRA trigger phrases into active prompts.
    settings["prompt"] = inject_lora_trigger(settings.get("prompt"), trigger_phrase)
    if settings.get("prompt_2"):
        settings["prompt_2"] = inject_lora_trigger(settings.get("prompt_2"), trigger_phrase)
    settings["lora_trigger_phrase"] = trigger_phrase or ""
    
    smart_extend_enabled = bool(settings.get("smart_extend"))
    image_path_raw = settings.get("image")
    if smart_extend_enabled and not image_path_raw and settings.get("input_image") is None:
        raise ValueError("Smart Extend requires an input image, but none was provided.")

    # Load input image if present (for Img2Img / Inpaint / Smart Extend).
    if image_path_raw:
        img_path = Path(image_path_raw)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found at generation time: {img_path}")
        # Load and ensure RGB
        pil_image = Image.open(img_path).convert("RGB")

        # Target dimensions from UI
        target_w = int(settings.get("width", 1024))
        target_h = int(settings.get("height", 1024))

        mask_pil = None
        if settings.get("mask_image"):
            mask_value = settings["mask_image"]
            if isinstance(mask_value, Image.Image):
                mask_pil = mask_value.convert("L")
            else:
                mask_path = Path(mask_value)
                if mask_path.exists():
                    try:
                        mask_pil = Image.open(mask_path).convert("L")
                    except Exception as e:
                        log.warning("Failed to load mask image: %s", e)
                        mask_pil = None
                else:
                    log.warning("Mask path does not exist: %s", mask_path)
                    mask_pil = None
            if mask_pil is not None and mask_pil.size != pil_image.size:
                mask_pil = ImageOps.fit(mask_pil, pil_image.size, method=Image.NEAREST)

        if settings.get("smart_extend"):
            anchor = settings.get("smart_extend_anchor", "center")
            smart_extend_initializer = str(
                os.environ.get("WEBBDUCK_SMART_EXTEND_INITIALIZER")
                or settings.get("smart_extend_initializer")
                or "noise"
            )
            pil_image, smart_mask, source_box = _build_smart_extend_inputs(
                pil_image,
                mask_pil,
                target_w,
                target_h,
                anchor,
                settings.get("smart_extend_feather", 8),
                settings.get("smart_extend_offset_x"),
                settings.get("smart_extend_offset_y"),
                smart_extend_initializer,
            )
            settings["smart_extend_initializer"] = smart_extend_initializer
            settings["width"], settings["height"] = pil_image.size
            settings["inpainting_fill"] = "replace"
            # Smart extend should preserve the source crop edge; avoid over-blurring the auto-mask.
            settings["mask_blur"] = min(int(settings.get("mask_blur", 0)), 2)
            settings["input_image"] = pil_image
            settings["mask_image"] = smart_mask
            settings["smart_extend_source_box"] = source_box
            log.info(
                "Smart extend prepared: src=%sx%s target=%sx%s offset=(%s,%s)",
                pil_image.size[0],
                pil_image.size[1],
                settings["width"],
                settings["height"],
                settings.get("smart_extend_offset_x"),
                settings.get("smart_extend_offset_y"),
            )
        else:
            if pil_image.size != (target_w, target_h):
                # Standard behavior for classic img2img/inpaint: fit to target frame.
                pil_image = ImageOps.fit(pil_image, (target_w, target_h), method=Image.LANCZOS)
                if mask_pil is not None and mask_pil.size != pil_image.size:
                    mask_pil = ImageOps.fit(mask_pil, pil_image.size, method=Image.NEAREST)

            settings["input_image"] = pil_image
            if mask_pil is not None:
                settings["mask_image"] = mask_pil
            elif settings.get("mask_image") is not None:
                del settings["mask_image"]


    seed_raw = settings.get("seed")
    seed = int(seed_raw) if seed_raw is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(pipe.device).manual_seed(seed)

    mode = select_mode(settings, pipe, img2img, base_img2img, base_inpaint)
    if smart_extend_enabled and mode.__class__.__name__ != "InpaintMode":
        raise RuntimeError(
            f"Smart Extend must run via InpaintMode, but selected {mode.__class__.__name__}. "
            "Input/mask preparation likely failed."
        )

    # Setup progress tracking
    total_steps = estimate_total_steps(settings)
    progress_tracker = GlobalProgress(total_steps, cancel_event=cancel_event)
    
    with stage_timer("mode_run_seconds"):
        images, out_seed = mode.run(
            settings=settings,
            pipe=pipe,
            img2img=img2img,
            base_img2img=base_img2img,
            base_inpaint=base_inpaint,
            generator=generator,
            callback=progress_tracker,
            faceid_kwargs=faceid_kwargs,
        )

    settings["performance_timing"] = snapshot_metrics()

    return images, out_seed
