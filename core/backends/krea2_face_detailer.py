"""ADetailer-style face/skin refinement pass for Krea 2.

A face that is small in the final image occupies only a handful of 16 px latent
tokens, so no decoder — including the 2x realism decoder — can recover pore or
hair detail that was never generated. This pass fixes the generation instead of
the decode:

1. Detect faces on the decoded image (OpenCV YuNet, which still resolves faces at
   roughly 32 px; Haar misses them).
2. Crop each face with padding so hair and jaw context are included.
3. Upscale the crop to about one megapixel, which turns it into an ordinary
   normal-sized Krea job that fits a 16 GB card.
4. Re-denoise the crop with the *same* prompt at a low denoise strength, reusing
   the already-loaded transformer.
5. Downscale, color-match against the original crop, and paste back through a
   feathered mask so the boundary is invisible.

Everything here is best-effort by design. A face that cannot be detected, a
missing detector model, or any failure inside a single crop must never fail the
generation: the unrefined image is returned with a note describing what happened.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

DEFAULT_DETECTOR_REPO = "opencv/opencv_zoo"
DEFAULT_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
# The file lives under the model directory, not a directory named after the file.
DEFAULT_DETECTOR_SUBFOLDER = "models/face_detection_yunet"
DEFAULT_TARGET_MEGAPIXELS = 1.0

# The face pass runs its own schedule, deliberately decoupled from the main
# generation's step count.
#
# ``img2img_schedule`` derives its noise floor from the *time-shifted* schedule,
# so "strength" is not a percentage of noise. For Krea's linspace(1, 1/steps,
# steps) base sigmas pushed through the Turbo shift (mu=1.15, shift~3.158), a
# 10-step run at strength 0.30 starts at shifted sigma 0.575 with only 3 steps to
# denoise - i.e. the crop is re-noised past halfway and re-rendered from
# scratch. Measured against a no-detailer control that rewrote ~77% of every
# facial pixel (mean 23/255) and shifted skin blue, which the elliptical
# composite then exposed as coloured discs.
#
# ``refine_steps=24`` with ``refine_strength=0.16`` lands at shifted sigma
# ~0.311 for 3 executed steps: same compute as before, but a genuine refinement.
DEFAULT_REFINE_STEPS = 24
DEFAULT_REFINE_STRENGTH = 0.16
DEFAULT_PAD_RATIO = 0.6
DEFAULT_FEATHER_PX = 12.0
DEFAULT_MAX_FACES = 4
DEFAULT_STRENGTH = 0.45
DEFAULT_MIN_FACE_PX = 32.0
# Above this fraction of the frame a face is already resolved by the main pass,
# so re-denoising it only risks changing the likeness.
DEFAULT_MAX_FACE_FRACTION = 0.40
# Cap the refined crop so a tiny budget face cannot explode into a huge job.
DEFAULT_MAX_CROP_EDGE = 1536

_OFF_VALUES = {"0", "false", "off", "no", "none", "disabled"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in _OFF_VALUES:
        return False
    return True if raw else default


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return _clamp(number, low, high)


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return int(default)
    return max(low, min(high, number))


def _weights_root() -> Path:
    """Resolve the upscaler/auxiliary weights root using the project convention."""
    explicit = str(os.getenv("WEBBDUCK_WEIGHTS_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    models_root = str(os.getenv("WEBBDUCK_MODELS_DIR") or "").strip()
    if models_root:
        return Path(models_root).expanduser() / "weights"

    return Path(__file__).resolve().parents[2] / "weights"


def resolve_detector_path(explicit: Any = None) -> Path:
    """Locate the YuNet ONNX model, downloading it into the weights cache once.

    Order: explicit request value, then ``WEBBDUCK_KREA2_FACE_DETECTOR``, then a
    cached copy under ``weights/faces/``. Raises when nothing is available so the
    caller can record a skipped pass instead of guessing at a detector.
    """
    raw = str(explicit or "").strip() or str(
        os.getenv("WEBBDUCK_KREA2_FACE_DETECTOR") or ""
    ).strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Krea face detector model not found: {path}")

    cached = _weights_root() / "faces" / DEFAULT_DETECTOR_FILE
    if cached.is_file():
        return cached
    return _download_detector(cached)


def _download_detector(destination: Path) -> Path:
    """Fetch the YuNet detector from opencv_zoo into the local weights cache."""
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = hf_hub_download(
        repo_id=DEFAULT_DETECTOR_REPO,
        filename=f"{DEFAULT_DETECTOR_SUBFOLDER}/{DEFAULT_DETECTOR_FILE}",
    )
    source = Path(resolved)
    if not source.is_file():
        raise FileNotFoundError(f"Downloaded Krea face detector is missing: {source}")
    if source != destination:
        destination.write_bytes(source.read_bytes())
    return destination


def normalize_detailer_request(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Build the worker ``face_detailer`` payload, or ``None`` when disabled.

    Opt-in only: without an explicit truthy toggle the pass never runs, so adding
    the capability cannot change existing generations.
    """
    cfg = settings.get("face_detailer")
    cfg = cfg if isinstance(cfg, dict) else {}
    if not _as_bool(cfg.get("enabled")):
        return None

    return {
        "enabled": True,
        "strength": round(_as_float(cfg.get("strength"), DEFAULT_STRENGTH, 0.05, 1.0), 4),
        # Face refine schedule. An explicitly supplied ``strength``/``steps`` is
        # still honoured as the refine budget, so existing saved presets keep
        # working; the defaults are the gentle ones.
        "refine_strength": round(
            _as_float(
                cfg.get("refine_strength"),
                _as_float(cfg.get("strength"), DEFAULT_REFINE_STRENGTH, 0.01, 1.0),
                0.01,
                1.0,
            ),
            4,
        ),
        "refine_steps": _as_int(
            cfg.get("refine_steps"),
            _as_int(cfg.get("steps"), DEFAULT_REFINE_STEPS, 1, 512),
            1,
            512,
        ),
        "min_face_px": round(_as_float(cfg.get("min_face_px"), DEFAULT_MIN_FACE_PX, 8.0, 2048.0), 2),
        "max_face_fraction": round(
            _as_float(cfg.get("max_face_fraction"), DEFAULT_MAX_FACE_FRACTION, 0.05, 1.0), 4
        ),
        "max_faces": _as_int(cfg.get("max_faces"), DEFAULT_MAX_FACES, 1, 16),
        "pad_ratio": round(_as_float(cfg.get("pad_ratio"), DEFAULT_PAD_RATIO, 0.0, 3.0), 4),
        "target_megapixels": round(
            _as_float(cfg.get("target_megapixels"), DEFAULT_TARGET_MEGAPIXELS, 0.05, 4.0), 4
        ),
        "feather_px": round(_as_float(cfg.get("feather_px"), DEFAULT_FEATHER_PX, 0.0, 256.0), 2),
        "max_crop_edge": _as_int(cfg.get("max_crop_edge"), DEFAULT_MAX_CROP_EDGE, 128, 4096),
        "detector_path": str(cfg.get("detector_path") or "").strip() or None,
    }


# ---------------------------------------------------------------------------
# Pure geometry / compositing (importable and testable without cv2 or torch)
# ---------------------------------------------------------------------------


def expand_face_box(
    face: tuple[int, int, int, int],
    image_size: tuple[int, int],
    pad_ratio: float = DEFAULT_PAD_RATIO,
) -> tuple[int, int, int, int]:
    """Grow a ``(x, y, w, h)`` detection by a padding ratio and clamp to the frame.

    Padding matters: a box tight to the eyes and mouth gives the refine pass no
    hair, jaw, or neck context, and the paste seam becomes visible where the
    refined crop ends.
    """
    img_w, img_h = int(image_size[0]), int(image_size[1])
    x, y, w, h = (int(round(float(v))) for v in face)
    pad_x = w * max(0.0, float(pad_ratio))
    pad_y = h * max(0.0, float(pad_ratio))
    x0 = max(0, int(math.floor(x - pad_x)))
    y0 = max(0, int(math.floor(y - pad_y)))
    x1 = min(img_w, int(math.ceil(x + w + pad_x)))
    y1 = min(img_h, int(math.ceil(y + h + pad_y)))
    # A face detected against the frame edge can collapse to a zero/negative
    # extent after clamping; keep at least one pixel so the crop is valid.
    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1


def refine_target_size(
    crop_size: tuple[int, int],
    *,
    target_megapixels: float = DEFAULT_TARGET_MEGAPIXELS,
    multiple: int = 16,
    max_edge: int = DEFAULT_MAX_CROP_EDGE,
) -> tuple[int, int]:
    """Scale a crop toward ``target_megapixels``, snapped to the latent grid.

    ``multiple`` must match Krea's constraint: the VAE downsamples by 8 and the
    transformer packs 2x2 patches, so a crop that is not a multiple of 16 would be
    silently resampled by the image processor and the paste would not line up.
    """
    crop_w, crop_h = max(1, int(crop_size[0])), max(1, int(crop_size[1]))
    multiple = max(16, int(multiple))
    max_edge = max(multiple, int(max_edge))

    area = float(crop_w) * float(crop_h)
    target_area = max(1.0, float(target_megapixels)) * 1_000_000.0
    scale = 1.0 if area <= 0 else math.sqrt(target_area / area)
    # Never upscale past the edge cap; a large close-up stays large.
    longest = max(crop_w, crop_h) * scale
    if longest > max_edge:
        scale *= max_edge / longest

    width = max(multiple, int(round(crop_w * scale / multiple)) * multiple)
    height = max(multiple, int(round(crop_h * scale / multiple)) * multiple)
    return width, height


def plan_face_passes(
    faces: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    *,
    min_face_px: float = DEFAULT_MIN_FACE_PX,
    max_face_fraction: float = DEFAULT_MAX_FACE_FRACTION,
    max_faces: int = DEFAULT_MAX_FACES,
) -> list[tuple[int, int, int, int]]:
    """Select which faces to refine, largest first.

    Faces below ``min_face_px`` are dropped because refining them cannot add real
    detail; faces wider than ``max_face_fraction`` of the frame are dropped
    because the main pass already resolved them and re-denoising would only
    drift the likeness. Largest-first ordering means a truncated ``max_faces``
    budget spends itself on the faces that benefit most.
    """
    img_w, img_h = int(image_size[0]), int(image_size[1])
    if img_w <= 0 or img_h <= 0:
        return []

    kept: list[tuple[int, int, int, int]] = []
    for face in faces:
        x, y, w, h = (int(round(float(v))) for v in face)
        if w <= 0 or h <= 0:
            continue
        shortest = float(min(w, h))
        if shortest < float(min_face_px):
            continue
        if float(w) / float(img_w) > float(max_face_fraction):
            continue
        if float(h) / float(img_h) > float(max_face_fraction):
            continue
        kept.append((x, y, w, h))

    kept.sort(key=lambda f: int(f[2]) * int(f[3]), reverse=True)
    return kept[: max(1, int(max_faces))]


def feather_alpha(
    size: tuple[int, int],
    feather_px: float = DEFAULT_FEATHER_PX,
) -> Any:
    """Build a float32 alpha mask that ramps 0 at the border to 1 in the interior.

    A hard-edged paste leaves a visible rectangle because the refined crop's
    texture statistics differ slightly from the surrounding pixels. The ramp lets
    the two regions cross-fade instead of meeting at a step.
    """
    import numpy as np

    width, height = max(1, int(size[0])), max(1, int(size[1]))
    feather = _clamp(float(feather_px), 0.0, float(min(width, height)) / 2.0)
    if feather <= 0.5:
        return np.ones((height, width), dtype=np.float32)

    ramp_x = np.ones(width, dtype=np.float32)
    ramp_y = np.ones(height, dtype=np.float32)
    ramp_x[: int(round(feather))] = np.linspace(0.0, 1.0, int(round(feather)), dtype=np.float32)
    ramp_x[width - int(round(feather)) :] = np.linspace(
        1.0, 0.0, int(round(feather)), dtype=np.float32
    )
    ramp_y[: int(round(feather))] = np.linspace(0.0, 1.0, int(round(feather)), dtype=np.float32)
    ramp_y[height - int(round(feather)) :] = np.linspace(
        1.0, 0.0, int(round(feather)), dtype=np.float32
    )
    return (ramp_y[:, None] * ramp_x[None, :]).astype(np.float32)


def face_feature_mask(
    size: tuple[int, int],
    box_xywh: tuple[int, int, int, int],
    feather_px: float = DEFAULT_FEATHER_PX,
) -> Any:
    """Build a feathered mask covering only the facial interior inside a crop.

    This is what keeps the pass from wrecking the likeness. A padded face crop also
    contains hair, ears, jaw, neck and background, and re-denoising all of it hands
    the model an entirely new composition - which is exactly how a refinement pass
    turns a good face uncanny. An ellipse inscribed in the YuNet box leaves the
    corners (hair and background) untouched, so only eyes, nose, mouth and cheeks
    are replaced.

    Returns an ``HxW`` float32 mask in ``[0, 1]``.
    """
    import numpy as np

    width, height = max(1, int(size[0])), max(1, int(size[1]))
    x, y, w, h = (int(v) for v in box_xywh)

    # Inset so the arc stays inside the detection and does not eat the hairline.
    cx = x + w / 2.0
    # Nudged down: a YuNet box includes the hair, the features sit lower.
    cy = y + h * 0.56
    rx = max(1.0, w / 2.0 - w * 0.06)
    ry = max(1.0, h / 2.0 - h * 0.10)

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    dist = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)

    feather = max(0.0, float(feather_px))
    if feather <= 0.0:
        return np.clip(1.5 - dist, 0.0, 1.0).astype(np.float32)

    band = max(1e-3, feather / max(rx, ry))
    return np.clip((1.0 - dist) / band, 0.0, 1.0).astype(np.float32)


def color_match(refined: Any, original: Any) -> Any:
    """Match the refined crop's per-channel mean and std to the original crop.

    Re-denoising at low strength slightly shifts exposure and contrast. Without
    this correction the crop reads as a brighter or flatter patch even though the
    geometry is correct. Per-channel affine matching keeps the refine pass from
    becoming a grading change.
    """
    import numpy as np

    ref = np.asarray(refined, dtype=np.float32)
    orig = np.asarray(original, dtype=np.float32)
    if ref.shape != orig.shape or ref.size == 0:
        return ref

    axes = (0, 1)
    ref_mean = ref.mean(axis=axes, keepdims=True)
    ref_std = ref.std(axis=axes, keepdims=True)
    orig_mean = orig.mean(axis=axes, keepdims=True)
    orig_std = orig.std(axis=axes, keepdims=True)
    # A perfectly flat refined crop has zero std; leave it alone rather than
    # dividing by ~0 and amplifying noise into a hard contrast shift.
    safe_std = np.where(ref_std < 1e-6, 1.0, ref_std)
    matched = (ref - ref_mean) * (orig_std / safe_std) + orig_mean
    return np.where(ref_std < 1e-6, ref, matched)


def paste_crop(
    base: Any,
    refined_crop: Any,
    box: tuple[int, int, int, int],
    feather_px: float = DEFAULT_FEATHER_PX,
) -> Any:
    """Alpha-composite a refined crop back into the base image at ``box``."""
    import numpy as np

    x0, y0, x1, y1 = (int(v) for v in box)
    region = base[y0:y1, x0:x1]
    if region.size == 0:
        return base
    crop = np.asarray(refined_crop, dtype=np.float32)[: region.shape[0], : region.shape[1]]
    if crop.shape[:2] != region.shape[:2]:
        return base

    alpha = feather_alpha((region.shape[1], region.shape[0]), feather_px)
    blended = crop * alpha[:, :, None] + region.astype(np.float32) * (1.0 - alpha[:, :, None])
    out = np.array(base, copy=True)
    out[y0:y1, x0:x1] = blended.astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Worker side: detection and the per-face refine loop
# ---------------------------------------------------------------------------


def create_detector(model_path: Path, input_size: tuple[int, int]) -> Any:
    """Build a YuNet detector sized for one image.

    YuNet requires the input size to match the image it is handed, so the
    detector is rebuilt per image rather than cached across differently sized
    frames.
    """
    import cv2

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (max(1, int(input_size[0])), max(1, int(input_size[1]))),
        float(_as_float(os.getenv("WEBBDUCK_KREA2_FACE_SCORE"), 0.6, 0.05, 1.0)),
        float(_as_float(os.getenv("WEBBDUCK_KREA2_FACE_NMS"), 0.3, 0.0, 1.0)),
        max(1, int(_as_int(os.getenv("WEBBDUCK_KREA2_FACE_TOP_K"), 5000, 1, 100000))),
    )
    if detector is None:
        raise RuntimeError(f"OpenCV could not load the Krea face detector: {model_path}")
    return detector


def detect_faces(detector: Any, image_array: Any) -> list[tuple[int, int, int, int]]:
    """Run YuNet and return ``(x, y, w, h)`` boxes in image pixel coordinates.

    ``image_array`` must be an RGB uint8 array. Returns an empty list when nothing
    is found, which is the normal case for a landscape without visible faces.
    """
    height, width = int(image_array.shape[0]), int(image_array.shape[1])
    detector.setInputSize((width, height))
    # OpenCV returns a 2-tuple ``(retval, faces)``, where ``faces`` is None when
    # nothing was found. Iterating the tuple itself would yield ints, not rows.
    result = detector.detect(image_array)
    if isinstance(result, tuple):
        _faces = result[1] if len(result) > 1 else None
    else:
        _faces = result
    if _faces is None or len(_faces) == 0:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for row in _faces:
        boxes.append(
            (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                max(1, int(round(float(row[2])))),
                max(1, int(round(float(row[3])))),
            )
        )
    return boxes


def refine_face(
    pipe: Any,
    vae: Any,
    *,
    image_array: Any,
    box: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
    config: dict[str, Any],
    prompt_embeds: Any,
    prompt_mask: Any,
    negative_embeds: Any,
    negative_mask: Any,
    guidance: float,
    seed: int,
    device: str,
    steps: int,
) -> Any:
    """Re-denoise one face crop and return it at the crop's original size.

    The crop is upscaled to roughly a megapixel so it becomes a normal-sized Krea
    img2img job, re-denoised through the shared latent-init path at the configured
    strength, then composited back through :func:`face_feature_mask`.

    Only the facial interior is taken from the refinement. The surrounding hair,
    jaw, neck and background come from the original crop, because regenerating
    them is what makes a refined face drift into something uncanny.
    """
    import numpy as np
    import torch
    from PIL import Image

    from core.backends.krea2_img2img import (
        encode_source_latent,
        img2img_schedule,
        noise_init_latent,
        release_vae,
    )

    from core.backends.krea2_worker_impl import _denoise_one, _decode_latents

    x0, y0, x1, y1 = (int(v) for v in box)
    crop_w, crop_h = x1 - x0, y1 - y0
    original = image_array[y0:y1, x0:x1].astype(np.float32)

    refine_w, refine_h = refine_target_size(
        (crop_w, crop_h),
        target_megapixels=float(config.get("target_megapixels") or DEFAULT_TARGET_MEGAPIXELS),
        max_edge=int(config.get("max_crop_edge") or DEFAULT_MAX_CROP_EDGE),
    )

    source = Image.fromarray(image_array[y0:y1, x0:x1].astype(np.uint8))
    source = source.resize((refine_w, refine_h), Image.Resampling.LANCZOS)

    # The refine schedule is intentionally its own budget rather than the main
    # pass's step count. See DEFAULT_REFINE_STEPS: reusing the main steps made the
    # crop start near sigma 0.58, which re-renders the face instead of refining it.
    refine_steps = max(1, int(config.get("refine_steps") or DEFAULT_REFINE_STEPS))
    _raw_strength = config.get("refine_strength")
    if _raw_strength is None:
        # Backward compatible: a config carrying only ``strength`` (older saved
        # presets, direct callers) still drives the refine budget.
        _raw_strength = config.get("strength")
    refine_strength = float(
        _raw_strength if _raw_strength is not None else DEFAULT_REFINE_STRENGTH
    )

    clean, seq_len = encode_source_latent(
        pipe, vae, source, width=refine_w, height=refine_h, device=device
    )
    sigmas, init_sigma, executed = img2img_schedule(
        pipe,
        steps=refine_steps,
        strength=refine_strength,
        image_seq_len=seq_len,
        device=device,
    )
    release_vae(vae)
    noised = noise_init_latent(clean, init_sigma, seed=seed, device=device)
    del clean

    latents = _denoise_one(
        pipe,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        negative_embeds=negative_embeds,
        negative_mask=negative_mask,
        width=refine_w,
        height=refine_h,
        steps=refine_steps,
        guidance=guidance,
        seed=seed,
        device=device,
        index=0,
        total_denoise_steps=max(1, executed),
        report=lambda *a, **k: None,
        init_latents=noised,
        init_sigmas=sigmas,
        executed_steps=executed,
    )
    del noised

    images = _decode_latents(pipe, vae, latents, width=refine_w, height=refine_h, device=device)
    release_vae(vae)
    refined = images[0].convert("RGB").resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    refined_array = np.asarray(refined, dtype=np.float32)
    del images, latents

    matched = color_match(refined_array, original)

    # Face box expressed in crop-local coordinates, so the mask lines up with the
    # crop we actually re-denoised.
    fx, fy, fw, fh = (int(v) for v in face_box)
    local_face = (fx - x0, fy - y0, fw, fh)
    mask = face_feature_mask(
        (crop_w, crop_h),
        local_face,
        float(
            config.get("feather_px")
            if config.get("feather_px") is not None
            else DEFAULT_FEATHER_PX
        ),
    )[:, :, None]

    composed = original * (1.0 - mask) + matched * mask
    info = {
        "init_sigma": round(float(init_sigma), 4),
        "executed_steps": int(executed),
        "refine_strength": round(float(refine_strength), 4),
        "refine_steps": int(refine_steps),
    }
    return composed.clip(0.0, 255.0).astype(np.uint8), info


def run_face_detailer(
    pipe: Any,
    vae: Any,
    image: Any,
    config: dict[str, Any],
    *,
    prompt_embeds: Any = None,
    prompt_mask: Any = None,
    negative_embeds: Any = None,
    negative_mask: Any = None,
    guidance: float = 0.0,
    seed: int = 0,
    device: str = "cuda",
    steps: int = 1,
    report: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Refine every selected face in ``image`` and return ``(image, note)``.

    Never raises. Any failure - no detector, no faces, a single crop blowing up -
    returns the original image with a note, because a partially refined portrait
    is a strictly better outcome than a failed generation.
    """
    import numpy as np

    note: dict[str, Any] = {
        "enabled": True,
        "detector": "yunet",
        "faces_detected": 0,
        "faces_refined": 0,
        "boxes": [],
        "skipped": None,
    }

    try:
        model_path = resolve_detector_path(config.get("detector_path"))
    except Exception as exc:
        note["skipped"] = "detector-unavailable"
        note["error"] = f"{type(exc).__name__}: {exc}"
        return image, note

    width, height = image.size
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    try:
        detector = create_detector(model_path, (width, height))
        # YuNet is an OpenCV DNN detector and expects BGR. The rest of this pass
        # works in RGB (PIL order), so hand the detector a BGR view of the same
        # pixels rather than swapping the channel order everywhere else.
        faces = detect_faces(detector, base[:, :, ::-1])
    except Exception as exc:
        note["skipped"] = "detector-failed"
        note["error"] = f"{type(exc).__name__}: {exc}"
        return image, note

    note["faces_detected"] = len(faces)
    selected = plan_face_passes(
        faces,
        (width, height),
        min_face_px=float(config.get("min_face_px") or DEFAULT_MIN_FACE_PX),
        max_face_fraction=float(config.get("max_face_fraction") or DEFAULT_MAX_FACE_FRACTION),
        max_faces=int(config.get("max_faces") or DEFAULT_MAX_FACES),
    )
    if not selected:
        # Distinguish "nothing was detected" from "everything was filtered out by
        # the size window", otherwise a size-gated pass reports as a detector miss.
        note["skipped"] = "no-faces" if not faces else "faces-below-threshold"
        note["boxes"] = [{"face": list(face)} for face in faces]
        return image, note

    pad_ratio = float(
        config.get("pad_ratio") if config.get("pad_ratio") is not None else DEFAULT_PAD_RATIO
    )
    feather_px = float(
        config.get("feather_px") if config.get("feather_px") is not None else DEFAULT_FEATHER_PX
    )
    working = base.astype(np.float32)
    for order, face in enumerate(selected):
        box = expand_face_box(face, (width, height), pad_ratio)
        try:
            if callable(report):
                report(f"Refining face {order + 1} of {len(selected)}")
            refined, refine_info = refine_face(
                pipe,
                vae,
                image_array=base.astype(np.uint8),
                box=box,
                face_box=face,
                config=config,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                negative_embeds=negative_embeds,
                negative_mask=negative_mask,
                guidance=guidance,
                seed=seed + order,
                device=device,
                steps=steps,
            )
        except Exception as exc:
            note["skipped"] = note["skipped"] or "crop-failed"
            note.setdefault("errors", []).append(
                f"face {order + 1}: {type(exc).__name__}: {exc}"
            )
            continue
        working = paste_crop(working, refined, box, feather_px)
        note["faces_refined"] += 1
        note["boxes"].append(
            {
                "face": list(face),
                "crop": list(box),
                "refine": refine_info,
            }
        )

    if note["faces_refined"] == 0:
        return image, note

    from PIL import Image

    refined_image = Image.fromarray(working.clip(0.0, 255.0).astype(np.uint8), mode="RGB")
    note["size"] = [refined_image.size[0], refined_image.size[1]]
    return refined_image, note
