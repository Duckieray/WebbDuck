"""Generation mode selection and execution."""

from .text2img import Text2ImgMode
from .img2img import Img2ImgMode
from .two_pass import TwoPassMode
import logging

log = logging.getLogger(__name__)

from .inpaint import InpaintMode

MODES = [
    InpaintMode(),
    TwoPassMode(),
    Img2ImgMode(),
    Text2ImgMode(),  # fallback
]


def select_mode(settings, pipe, img2img, base_img2img, base_inpaint=None):
    """Select appropriate generation mode based on settings."""
    log.debug("=== MODE SELECTION START ===")
    log.debug(f"experimental_compress = {settings.get('experimental_compress')}")
    log.debug(f"img2img loaded = {img2img is not None}")
    log.debug(f"input_image = {settings.get('input_image') is not None}")

    for mode in MODES:
        name = mode.__class__.__name__
        try:
            can_run = mode.can_run(settings, pipe, img2img, base_img2img, base_inpaint)
        except Exception as e:
            log.error(f"[MODE CHECK ERROR] {name}: {e}")
            continue

        log.debug(f"[MODE CHECK] {name}.can_run() -> {can_run}")

        if can_run:
            log.info(f"[MODE SELECTED] {name}")
            return mode

    log.error("❌ NO MODE SELECTED")
    raise RuntimeError("No valid generation mode found")


__all__ = ["select_mode", "Text2ImgMode", "Img2ImgMode", "TwoPassMode", "InpaintMode"]