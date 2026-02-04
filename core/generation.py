"""Main generation orchestration."""

import torch
import random

from webbduck.core.pipeline import pipeline_manager
from webbduck.modes import select_mode


def run_generation(settings):
    """Execute image generation based on settings."""
    second_pass_model = settings.get("second_pass_model")
    if second_pass_model == "None":
        second_pass_model = None

    pipe, img2img, _ = pipeline_manager.get(
        base_model=settings["base_model"],
        second_pass_model=second_pass_model,
        loras=settings.get("loras", []),
    )

    seed = settings.get("seed") or random.randint(0, 2**32 - 1)
    generator = torch.Generator(pipe.device).manual_seed(seed)

    mode = select_mode(settings, pipe, img2img)

    images, out_seed = mode.run(
        settings=settings,
        pipe=pipe,
        img2img=img2img,
        generator=generator,
    )

    return images, out_seed