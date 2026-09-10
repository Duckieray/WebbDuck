"""Reference preparation and final-size policy for Krea 2 identity runs.

Krea2Edit behaves more consistently when very large persona references are
normalized before either conditioning path sees them.  This module also detects
small faces and, when useful, makes a generous head/shoulders crop before the
canonical 1024px downscale.  The same prepared reference is then consumed by
both Qwen3-VL grounding and the VAE appearance path.

The denoiser may still render below the requested output size when the adaptive
planner protects a constrained GPU.  That is native inference resolution, not
artifact resolution; the final-size wrapper restores the exact requested size.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_REFERENCE_MAX_EDGE = 1024
DEFAULT_FACE_WIDTH_TRIGGER = 0.26
DEFAULT_FACE_AREA_TRIGGER = 0.08
DEFAULT_FACE_FILL_AFTER_CROP = 0.42
_OFF_VALUES = {"0", "false", "off", "no", "none", "disabled"}
_REAL_ESRGAN_VALUES = {"1", "true", "yes", "on", "auto", "realesrgan", "real-esrgan"}
_LEGACY_REAL_ESRGAN_VALUES = {"1", "true", "yes", "on"}


def _lanczos() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def reference_max_edge(identity_cfg: dict[str, Any] | None = None) -> int:
    """Return the canonical reference longest-edge cap."""
    cfg = identity_cfg if isinstance(identity_cfg, dict) else {}
    raw: Any = cfg.get("reference_max_edge")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_MAX_REFERENCE_EDGE", "")
    text = str(raw or "").strip().lower()
    if text in _OFF_VALUES:
        return 0
    if not text:
        return DEFAULT_REFERENCE_MAX_EDGE
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return DEFAULT_REFERENCE_MAX_EDGE
    return max(256, value)


def auto_face_crop_enabled(identity_cfg: dict[str, Any] | None = None) -> bool:
    """Whether small-face references should be reframed automatically.

    The old ``face_crop`` field was intentionally forced off in the provider
    contract while Krea identity was experimental, so this uses a new explicit
    ``auto_face_crop`` switch and defaults on.  Disable with
    ``WEBBDUCK_KREA2_IDENTITY_AUTO_FACE_CROP=0``.
    """
    cfg = identity_cfg if isinstance(identity_cfg, dict) else {}
    raw: Any = cfg.get("auto_face_crop")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_AUTO_FACE_CROP", "1")
    return str(raw).strip().lower() not in _OFF_VALUES


def detect_face(image: Image.Image) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """Best-effort CPU face detection without making Krea depend on OpenCV.

    OpenCV is already present in normal WebbDuck installs for FLUX identity
    helpers, but the isolated Krea runtime is allowed to omit it.  In that case
    this returns ``(None, None)`` and identity generation proceeds normally.
    The detector works on a <=1280px copy and maps the strongest face box back
    to the original image coordinates.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None, None

    rgb = image.convert("RGB")
    original_w, original_h = rgb.size
    if min(original_w, original_h) < 32:
        return None, "opencv-haar"

    detect = rgb
    scale = 1.0
    longest = max(original_w, original_h)
    if longest > 1280:
        scale = 1280.0 / float(longest)
        detect = rgb.resize(
            (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
            _lanczos(),
        )

    array = np.asarray(detect)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None, "opencv-haar"

    dh, dw = gray.shape[:2]
    min_side = max(24, min(dh, dw) // 14)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    if len(faces) == 0:
        return None, "opencv-haar"

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    inv = 1.0 / scale
    box = (
        max(0, int(round(float(x) * inv))),
        max(0, int(round(float(y) * inv))),
        max(1, int(round(float(w) * inv))),
        max(1, int(round(float(h) * inv))),
    )
    return box, "opencv-haar"


def face_occupancy(
    face_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    x, y, face_w, face_h = face_box
    width, height = image_size
    del x, y
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    return (
        float(face_w) / float(width),
        float(face_w * face_h) / float(width * height),
    )


def should_auto_face_crop(
    face_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    width_ratio, area_ratio = face_occupancy(face_box, image_size)
    return width_ratio < DEFAULT_FACE_WIDTH_TRIGGER or area_ratio < DEFAULT_FACE_AREA_TRIGGER


def _face_crop_box(
    image_size: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    face_fill: float = DEFAULT_FACE_FILL_AFTER_CROP,
) -> tuple[int, int, int, int]:
    """Generous square head/shoulders crop, shifted slightly below face center."""
    width, height = image_size
    x, y, face_w, face_h = face_box
    face_fill = max(0.25, min(0.65, float(face_fill)))
    side = int(round(max(face_w, face_h) / face_fill))
    side = max(max(face_w, face_h), min(side, width, height))

    cx = float(x) + float(face_w) / 2.0
    # Slight downward shift keeps neck/shoulders while retaining hairline.
    cy = float(y) + float(face_h) / 2.0 + 0.18 * float(face_h)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return left, top, left + side, top + side


def _transform_face_box_after_crop(
    face_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    left, top, _right, _bottom = crop_box
    return max(0, x - left), max(0, y - top), w, h


def _scale_face_box(
    face_box: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    return (
        max(0, int(round(x * scale))),
        max(0, int(round(y * scale))),
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )


def prepare_identity_reference(
    image: Image.Image,
    *,
    max_long_edge: int = DEFAULT_REFERENCE_MAX_EDGE,
    auto_face_crop: bool = True,
    face_box: tuple[int, int, int, int] | None = None,
    detector: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return one canonical reference for both Krea conditioning channels.

    Processing order is intentional: detect/crop first so a tiny face is not
    lost during downsampling, then cap the prepared reference to 1024px.  Small
    references are never upscaled.
    """
    prepared = image.convert("RGB")
    original_w, original_h = prepared.size
    max_long_edge = int(max_long_edge or 0)

    if face_box is None:
        face_box, detector = detect_face(prepared)

    original_face_box = tuple(face_box) if face_box is not None else None
    width_ratio = area_ratio = 0.0
    if face_box is not None:
        width_ratio, area_ratio = face_occupancy(face_box, prepared.size)

    meta: dict[str, Any] = {
        "original_size": [original_w, original_h],
        "prepared_size": [original_w, original_h],
        "max_reference_edge": max_long_edge,
        "downscaled": False,
        "scale": 1.0,
        "face_detected": face_box is not None,
        "face_detector": detector,
        "face_box_original": list(original_face_box) if original_face_box is not None else None,
        "face_box_prepared": list(original_face_box) if original_face_box is not None else None,
        "face_width_ratio": round(width_ratio, 6),
        "face_area_ratio": round(area_ratio, 6),
        "auto_face_crop_enabled": bool(auto_face_crop),
        "auto_face_cropped": False,
        "crop_box": None,
    }

    if auto_face_crop and face_box is not None and should_auto_face_crop(face_box, prepared.size):
        crop_box = _face_crop_box(prepared.size, face_box)
        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]
        # Avoid a no-op crop when the face is already close enough to full frame.
        if crop_w * crop_h < int(original_w * original_h * 0.90):
            prepared = prepared.crop(crop_box)
            face_box = _transform_face_box_after_crop(face_box, crop_box)
            meta["auto_face_cropped"] = True
            meta["crop_box"] = list(crop_box)

    longest = max(prepared.size)
    if max_long_edge > 0 and longest > max_long_edge:
        scale = float(max_long_edge) / float(longest)
        new_w = max(1, int(round(prepared.size[0] * scale)))
        new_h = max(1, int(round(prepared.size[1] * scale)))
        prepared = prepared.resize((new_w, new_h), _lanczos())
        if face_box is not None:
            face_box = _scale_face_box(face_box, scale)
        meta["downscaled"] = True
        meta["scale"] = round(scale, 6)

    meta["prepared_size"] = list(prepared.size)
    meta["face_box_prepared"] = list(face_box) if face_box is not None else None
    if face_box is not None:
        prepared_width_ratio, prepared_area_ratio = face_occupancy(face_box, prepared.size)
        meta["prepared_face_width_ratio"] = round(prepared_width_ratio, 6)
        meta["prepared_face_area_ratio"] = round(prepared_area_ratio, 6)
    else:
        meta["prepared_face_width_ratio"] = 0.0
        meta["prepared_face_area_ratio"] = 0.0

    return prepared, meta


def _prepare_request_reference(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, Path | None]:
    """Prepare the reference and pass face geometry to the face-attention layer."""
    tuned = dict(request)
    identity_cfg = tuned.get("identity")
    if not isinstance(identity_cfg, dict):
        return tuned, None, None

    identity = dict(identity_cfg)
    raw_path = str(identity.get("reference_image") or "").strip()
    if not raw_path:
        return tuned, None, None

    max_edge = reference_max_edge(identity)
    auto_crop = auto_face_crop_enabled(identity)
    try:
        with Image.open(raw_path) as source:
            source.load()
            prepared, meta = prepare_identity_reference(
                source,
                max_long_edge=max_edge,
                auto_face_crop=auto_crop,
            )
    except Exception:
        # Let the baseline worker raise its existing actionable reference error.
        return tuned, None, None

    identity["_prepared_face_box"] = meta.get("face_box_prepared")
    identity["_face_detector"] = meta.get("face_detector")
    identity["_reference_auto_face_cropped"] = bool(meta.get("auto_face_cropped"))

    changed = bool(meta.get("downscaled") or meta.get("auto_face_cropped"))
    if not changed:
        tuned["identity"] = identity
        return tuned, meta, None

    handle = tempfile.NamedTemporaryFile(
        prefix="webbduck_krea2_identity_ref_",
        suffix=".png",
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        prepared.save(temp_path, format="PNG")
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        tuned["identity"] = identity
        return tuned, meta, None

    identity["reference_image"] = str(temp_path)
    tuned["identity"] = identity
    return tuned, meta, temp_path


def _attach_reference_meta(result: Any, meta: dict[str, Any] | None) -> None:
    if not isinstance(result, dict) or not isinstance(meta, dict):
        return
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        return
    identity = runtime.setdefault("identity", {})
    if isinstance(identity, dict):
        identity["reference_preprocess"] = dict(meta)


def _final_size_upscaler(original: Any):
    """Restore exact requested artifact dimensions after safe native denoising."""

    def _call_realesrgan(image: Any, *, requested: Any, effective: Any, raw: str):
        key = "WEBBDUCK_KREA2_IDENTITY_UPSCALE"
        previous = os.environ.get(key)
        if raw not in _LEGACY_REAL_ESRGAN_VALUES:
            os.environ[key] = "1"
        try:
            return original(image, requested=requested, effective=effective)
        finally:
            if raw not in _LEGACY_REAL_ESRGAN_VALUES:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def _maybe_upscale(image: Any, *, requested: Any, effective: Any):
        raw = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE") or "").strip().lower()
        if raw in _OFF_VALUES:
            return image, None
        if raw in _REAL_ESRGAN_VALUES:
            return _call_realesrgan(
                image,
                requested=requested,
                effective=effective,
                raw=raw,
            )

        if requested is None:
            return image, None
        try:
            req_w, req_h = int(requested[0]), int(requested[1])
        except Exception:
            return image, None
        if req_w <= 0 or req_h <= 0:
            return image, None

        current_w, current_h = image.size
        if (current_w, current_h) == (req_w, req_h):
            return image, None

        image = image.resize((req_w, req_h), _lanczos())
        note = {
            "from": [current_w, current_h],
            "to": [req_w, req_h],
            "upscaler": "lanczos",
            "policy": "exact-requested-size",
            "native_effective_size": list(effective) if effective is not None else [current_w, current_h],
        }
        return image, note

    return _maybe_upscale


def install_identity_reference_patch(base: Any) -> None:
    """Install canonical reference prep and exact final-size restoration."""
    if bool(getattr(base, "_krea_identity_reference_patch", False)):
        return

    original_run = base._run_identity
    original_upscale = base._maybe_upscale_identity_artifact

    def _run_with_prepared_reference(request: dict[str, Any], *args: Any, **kwargs: Any):
        tuned, meta, temp_path = _prepare_request_reference(request)
        try:
            result = original_run(tuned, *args, **kwargs)
            _attach_reference_meta(result, meta)
            return result
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    base._run_identity = _run_with_prepared_reference
    base._maybe_upscale_identity_artifact = _final_size_upscaler(original_upscale)
    base._prepare_identity_reference = prepare_identity_reference
    base._detect_identity_face = detect_face
    base._krea_identity_reference_patch = True
