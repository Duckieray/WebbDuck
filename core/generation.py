"""Main generation orchestration."""

from pathlib import Path
import torch
import random
from PIL import Image, ImageOps

from webbduck.core.pipeline import pipeline_manager
from webbduck.core.captioner import unload_captioners
from webbduck.modes import select_mode
from webbduck.server.state import update_progress


class GlobalProgress:
    """Tracks progress across multiple generation passes."""
    
    def __init__(self, total_estimated_steps):
        self.total_estimated_steps = total_estimated_steps
        self.current_global_step = 0
        self.pass_start_step = 0
        
    def get_callback(self):
        """Returns a diffusers-compatible callback."""
        def callback(pipe, step_index, timestep, callback_kwargs):
            # Calculate actual global step
            # step_index is 0-based index of Current Pass
            
            # Update global counter
            current_pass_step = step_index + 1
            global_step = self.pass_start_step + current_pass_step
            
            # Clamping
            global_step = min(global_step, self.total_estimated_steps)
            
            # Calculate percentage (35% to 90% is the generation phase)
            # We reserve 0-35% for loading, and 90-100% for saving/decoding
            # So generation maps 0-100% of steps to 35-90% of global progress
            
            gen_progress = global_step / max(1, self.total_estimated_steps)
            
            # Map generation progress (0-1) to UI progress range (0.35 - 0.90)
            ui_progress = 0.35 + (gen_progress * 0.55)
            
            update_progress(ui_progress, step=global_step, total_steps=self.total_estimated_steps)
            
            return callback_kwargs

        return callback

    def finish_pass(self, steps_completed):
        """Mark a pass as complete and advance the baseline."""
        self.pass_start_step += steps_completed


def estimate_total_steps(settings):
    """Estimate total steps based on settings."""
    steps = int(settings.get("steps", 30))
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


def run_generation(settings):
    """Execute image generation based on settings."""
    second_pass_model = settings.get("second_pass_model")
    if second_pass_model == "None":
        second_pass_model = None

    # Unload any captioner models to free VRAM before loading generation pipelines
    unload_captioners()

    pipe, img2img, base_img2img, base_inpaint, trigger_phrase = pipeline_manager.get(
        base_model=settings["base_model"],
        second_pass_model=second_pass_model,
        loras=settings.get("loras", []),
        scheduler_name=settings.get("scheduler"),
    )

    # Always inject LoRA trigger phrases into active prompts.
    settings["prompt"] = inject_lora_trigger(settings.get("prompt"), trigger_phrase)
    if settings.get("prompt_2"):
        settings["prompt_2"] = inject_lora_trigger(settings.get("prompt_2"), trigger_phrase)
    settings["lora_trigger_phrase"] = trigger_phrase or ""
    
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

    # Setup progress tracking
    total_steps = estimate_total_steps(settings)
    progress_tracker = GlobalProgress(total_steps)
    
    images, out_seed = mode.run(
        settings=settings,
        pipe=pipe,
        img2img=img2img,
        base_img2img=base_img2img,
        base_inpaint=base_inpaint,
        generator=generator,
        callback=progress_tracker,
    )

    return images, out_seed
