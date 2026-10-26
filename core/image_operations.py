"""CPU-side preparation shared by image-generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat


@dataclass
class PreparedImageOperation:
    image: Image.Image
    mask: Image.Image | None
    source_box: dict[str, int] | None = None


def _anchor_offset(anchor: str, src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    dx = max(0, dst_w - src_w)
    dy = max(0, dst_h - src_h)
    anchor = (anchor or "center").lower()
    positions = {
        "top": (dx // 2, 0), "bottom": (dx // 2, dy),
        "left": (0, dy // 2), "right": (dx, dy // 2),
        "top_left": (0, 0), "top_right": (dx, 0),
        "bottom_left": (0, dy), "bottom_right": (dx, dy),
    }
    return positions.get(anchor, (dx // 2, dy // 2))


def _tile_to_length(band: Image.Image, length: int) -> Image.Image:
    """Repeat a mirrored edge band until it covers ``length`` without rescaling it."""
    width, band_h = band.size
    if band_h <= 0:
        return band
    if band_h == length:
        return band
    tiled = Image.new(band.mode, (width, length))
    cursor = 0
    while cursor < length:
        tiled.paste(band, (0, cursor))
        cursor += band_h
    return tiled.crop((0, 0, width, length))


def _mirror_seeded_canvas(input_image: Image.Image, target_size, box) -> Image.Image:
    """Seed new canvas by mirroring the source outward at its own 1:1 scale.

    Two failure modes to avoid, both observed in outpaint output:

    * Seeding with a scale-to-cover ``ImageOps.fit`` magnifies the source to cover
      the target and centre crops it, so the canvas holds a blurred,
      multi-magnified slice of the picture right beside the real full-size source.
      The model cannot reconcile two scales and answers by generating an unrelated
      scene into the new area.
    * Seeding by stretching a one-pixel border strip outward (a plain
      ``crop``/``resize``) produces a constant repeat of that single row or column,
      which reads as the image being smeared or stretched downwards.

    Mirroring the edge band keeps a single consistent scale while giving the model
    plausible large-scale structure to continue from, instead of either a phantom
    or a streak.
    """
    width, height = target_size
    src_w, src_h = input_image.size
    x, y = box
    left = x
    right = max(0, width - (x + src_w))
    top = y
    bottom = max(0, height - (y + src_h))

    canvas = Image.new("RGB", (width, height), _edge_mean_color(input_image))
    canvas.paste(input_image, (x, y))

    if left > 0:
        take = min(left, src_w)
        band = input_image.crop((0, 0, take, src_h)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(_tile_to_length(band, left), (0, y))
    if right > 0:
        take = min(right, src_w)
        band = input_image.crop((src_w - take, 0, src_w, src_h)).transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(_tile_to_length(band, right), (x + src_w, y))
    if top > 0:
        take = min(top, src_h)
        band = input_image.crop((0, 0, src_w, take)).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(_tile_to_length(band, top), (x, 0))
    if bottom > 0:
        take = min(bottom, src_h)
        band = input_image.crop((0, src_h - take, src_w, src_h)).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(_tile_to_length(band, bottom), (x, y + src_h))

    # Corners sit outside every mirrored band, so seed them from the nearest
    # source corner colour rather than leaving flat filler.
    for cx, cy, colour in (
        (0, 0, input_image.getpixel((0, 0))),
        (x + src_w, 0, input_image.getpixel((src_w - 1, 0))),
        (0, y + src_h, input_image.getpixel((0, src_h - 1))),
        (x + src_w, y + src_h, input_image.getpixel((src_w - 1, src_h - 1))),
    ):
        if left > 0 and top > 0 and cx == 0 and cy == 0:
            canvas.paste(Image.new("RGB", (left, top), colour), (0, 0))
        if right > 0 and top > 0 and cx == x + src_w and cy == 0:
            canvas.paste(Image.new("RGB", (right, top), colour), (x + src_w, 0))
        if left > 0 and bottom > 0 and cx == 0 and cy == y + src_h:
            canvas.paste(Image.new("RGB", (left, bottom), colour), (0, y + src_h))
        if right > 0 and bottom > 0 and cx == x + src_w and cy == y + src_h:
            canvas.paste(Image.new("RGB", (right, bottom), colour), (x + src_w, y + src_h))
    return canvas


def _edge_strip_seeded_canvas(input_image: Image.Image, target_size, box) -> Image.Image:
    """Seed by stretching a one-pixel border strip outward (legacy ``edge_strips``)."""
    width, height = target_size
    src_w, src_h = input_image.size
    x, y = box
    canvas = Image.new("RGB", (width, height), _edge_mean_color(input_image))
    canvas.paste(input_image, (x, y))
    if x > 0:
        canvas.paste(input_image.crop((0, 0, 1, src_h)).resize((x, src_h), Image.BILINEAR), (0, y))
    if width - (x + src_w) > 0:
        right = width - (x + src_w)
        canvas.paste(
            input_image.crop((src_w - 1, 0, src_w, src_h)).resize((right, src_h), Image.BILINEAR),
            (x + src_w, y),
        )
    if y > 0:
        canvas.paste(input_image.crop((0, 0, src_w, 1)).resize((src_w, y), Image.BILINEAR), (x, 0))
    if height - (y + src_h) > 0:
        bottom = height - (y + src_h)
        canvas.paste(
            input_image.crop((0, src_h - 1, src_w, src_h)).resize((src_w, bottom), Image.BILINEAR),
            (x, y + src_h),
        )
    return canvas


def _noise_seeded_canvas(input_image: Image.Image, target_size, box, noise: int = 18) -> Image.Image:
    """Seed the new area with edge-tinted noise and no structure.

    Inpainting treats everything outside the mask as ground-truth context, so any
    structure placed there is a promise the model will faithfully keep. That is why
    a scale-to-cover fit yields a duplicated scene and a stretched border strip
    yields vertical smearing: the model continues the fabricated context rather
    than inventing new scene. Filling with featureless noise leaves nothing to copy,
    so the real source plus the model's own prior decide the continuation.
    """
    width, height = target_size
    src_w, src_h = input_image.size
    x, y = box
    rng = np.random.default_rng(0)
    base = _edge_mean_color(input_image)
    canvas = Image.new("RGB", (width, height), base)
    canvas.paste(input_image, (x, y))

    # Add noise only outside the source box so the pasted source stays bit-exact.
    pixels = np.asarray(canvas).astype(np.int16)
    region = np.zeros((height, width), dtype=bool)
    region[:, :] = True
    region[y : y + src_h, x : x + src_w] = False
    if region.any():
        jitter = rng.normal(0.0, noise, size=(height, width, 3))
        pixels[region] = np.clip(pixels[region] + jitter[region], 0, 255).astype(np.int16)
    return Image.fromarray(pixels.astype(np.uint8), "RGB")


def _seed_canvas(input_image: Image.Image, target_size, box, initializer: str = "noise") -> Image.Image:
    """Dispatch to a canvas initializer, defaulting to structureless noise."""
    choice = (initializer or "noise").strip().lower()
    if choice == "mirror":
        return _mirror_seeded_canvas(input_image, target_size, box)
    if choice in {"edge_strips", "edge_strip", "stretch"}:
        return _edge_strip_seeded_canvas(input_image, target_size, box)
    if choice == "soft_context":
        width, height = target_size
        src_w, src_h = input_image.size
        x, y = box
        canvas = Image.new("RGB", (width, height), _edge_mean_color(input_image))
        context = ImageOps.fit(input_image, (width, height), method=Image.LANCZOS)
        canvas.paste(context.filter(ImageFilter.GaussianBlur(radius=max(18, width // 20))), (0, 0))
        canvas.paste(input_image, (x, y))
        return canvas
    return _noise_seeded_canvas(input_image, target_size, box)


def _edge_mean_color(image: Image.Image) -> tuple[int, int, int]:
    src_w, src_h = image.size
    top = image.crop((0, 0, src_w, 1))
    bottom = image.crop((0, src_h - 1, src_w, src_h))
    left = image.crop((0, 0, 1, src_h)).resize((src_h, 1), Image.BILINEAR)
    right = image.crop((src_w - 1, 0, src_w, src_h)).resize((src_h, 1), Image.BILINEAR)
    strip = Image.new("RGB", (src_w * 2 + src_h * 2, 1))
    cursor = 0
    for segment in (top, bottom, left, right):
        strip.paste(segment, (cursor, 0))
        cursor += segment.width
    mean = ImageStat.Stat(strip).mean
    return tuple(int(max(0, min(255, round(value)))) for value in mean[:3])


def build_smart_extend_inputs(
    input_image: Image.Image,
    target_w: int,
    target_h: int,
    anchor: str,
    feather_px: int = 8,
    offset_x: int | None = None,
    offset_y: int | None = None,
    initializer: str = "noise",
) -> tuple[Image.Image, Image.Image, dict[str, int]]:
    """Place an image on an expanded canvas and mask only the new area."""
    input_image = input_image.convert("RGB")
    src_w, src_h = input_image.size
    dst_w, dst_h = max(src_w, int(target_w)), max(src_h, int(target_h))
    if offset_x is None or offset_y is None:
        x, y = _anchor_offset(anchor, src_w, src_h, dst_w, dst_h)
    else:
        x = max(0, min(dst_w - src_w, int(offset_x)))
        y = max(0, min(dst_h - src_h, int(offset_y)))

    expanded = _seed_canvas(input_image, (dst_w, dst_h), (x, y), initializer=initializer)

    mask = Image.new("L", (dst_w, dst_h), 255)
    mask.paste(0, (x, y, x + src_w, y + src_h))
    blend_px = max(2, min(12, 2 + max(0, min(64, int(feather_px))) // 4))
    if x > 0:
        mask.paste(255, (x, y, min(dst_w, x + blend_px), y + src_h))
    if x + src_w < dst_w:
        mask.paste(255, (max(x, x + src_w - blend_px), y, x + src_w, y + src_h))
    if y > 0:
        mask.paste(255, (x, y, x + src_w, min(dst_h, y + blend_px)))
    if y + src_h < dst_h:
        mask.paste(255, (x, max(y, y + src_h - blend_px), x + src_w, y + src_h))
    return expanded, mask, {"x": x, "y": y, "w": src_w, "h": src_h}


def _load_image(value: Any, mode: str) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert(mode)
    if not value:
        return None
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found at generation time: {path}")
    with Image.open(path) as image:
        return image.convert(mode)


def prepare_image_operation(
    settings: dict[str, Any],
    *,
    apply_mask_blur: bool = True,
) -> PreparedImageOperation | None:
    """Normalize inpaint/outpaint inputs without depending on a model runtime."""
    image = _load_image(settings.get("input_image") or settings.get("image"), "RGB")
    if image is None:
        return None
    mask = _load_image(settings.get("mask_image"), "L")
    target = (int(settings.get("width") or image.width), int(settings.get("height") or image.height))

    if settings.get("smart_extend"):
        prepared, mask, source_box = build_smart_extend_inputs(
            image, target[0], target[1], str(settings.get("smart_extend_anchor") or "center"),
            int(settings.get("smart_extend_feather") or 8), settings.get("smart_extend_offset_x"),
            settings.get("smart_extend_offset_y"),
        )
        return PreparedImageOperation(prepared, mask, source_box)

    if image.size != target:
        image = ImageOps.fit(image, target, method=Image.LANCZOS)
    if mask is not None and mask.size != image.size:
        mask = ImageOps.fit(mask, image.size, method=Image.NEAREST)
    if mask is not None:
        if settings.get("inpainting_fill", "replace") == "keep":
            mask = ImageOps.invert(mask)
        blur = max(0, int(settings.get("mask_blur") or 0))
        if blur and apply_mask_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return PreparedImageOperation(image, mask)
