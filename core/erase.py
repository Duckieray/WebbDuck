"""Defaults for diffusion-based object erasing.

Erase reuses the inpaint pipeline: the masked region is repainted from a prompt that
describes the surrounding scene instead of the removed object. Keeping the defaults in
one module lets the HTTP layer and the inpaint mode agree without duplicating strings.
"""

ERASE_PROMPT = (
    "empty background, seamless continuation of the surrounding scene, "
    "natural lighting, matching perspective and texture"
)
ERASE_NEGATIVE_PROMPT = (
    "object, subject, person, animal, text, watermark, logo, seam, blurry"
)
ERASE_DEFAULT_STRENGTH = 0.99
ERASE_DEFAULT_MASK_BLUR = 12
ERASE_DEFAULT_MASK_GROW = 6

__all__ = [
    "ERASE_PROMPT",
    "ERASE_NEGATIVE_PROMPT",
    "ERASE_DEFAULT_STRENGTH",
    "ERASE_DEFAULT_MASK_BLUR",
    "ERASE_DEFAULT_MASK_GROW",
]
