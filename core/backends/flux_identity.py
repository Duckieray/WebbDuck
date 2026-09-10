"""FLUX.2 native-reference preprocessing for Persona identity.

FLUX.2 native references are image-edit conditioning, not FaceID embeddings.
For Persona work we therefore keep the reference signal conservative:

* very large references are normalized to a 1024px longest edge;
* small references are never upscaled;
* automatic face reframing only happens when the face is genuinely small;
* the automatic crop is a generous head/shoulders crop (~42% face fill), not
  the old 80%-face passport crop;
* aspect ratio is never squashed.

Multi-reference helpers remain available for genuine FLUX.2 reference-edit
workflows. PhoenixDraw Persona requests now use one canonical anchor by default.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_CROP_SIZE = 1024  # backward-compatible public name: now a max edge cap
DEFAULT_MAX_EDGE = 1024
DEFAULT_FACE_FILL = 0.42
DEFAULT_FACE_WIDTH_TRIGGER = 0.26
DEFAULT_FACE_AREA_TRIGGER = 0.08

_OFF_VALUES = {"off", "false", "0", "none", "disabled", ""}
_FORCE_CROP_VALUES = {"on", "true", "1", "always", "tight"}


def _default_cache_dir() -> Path:
    explicit = os.getenv("WEBBDUCK_FLUX_IDENTITY_CACHE_DIR")
    if explicit:
        root = Path(explicit)
    else:
        base = os.getenv("WEBBDUCK_OUTPUT_DIR")
        root = (
            Path(base) / ".flux_identity_cache"
            if base
            else Path(tempfile.gettempdir()) / "webbduck_flux_identity"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


_FACE_APP = None
_FACE_APP_TRIED = False
_HAAR_CASCADE = None


def _detect_face(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the ``(x, y, w, h)`` box of the strongest detected face."""
    global _FACE_APP, _FACE_APP_TRIED, _HAAR_CASCADE
    height, width = img_bgr.shape[:2]
    if height < 24 or width < 24:
        return None

    if not _FACE_APP_TRIED:
        _FACE_APP_TRIED = True
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _FACE_APP = app
        except Exception:
            _FACE_APP = None

    if _FACE_APP is not None:
        try:
            faces = _FACE_APP.get(img_bgr)
        except Exception:
            faces = []
        if faces:
            best = max(
                faces,
                key=lambda f: (float(f.det_score), float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1])),
            )
            x1, y1, x2, y2 = (float(v) for v in best.bbox)
            return int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))

    if _HAAR_CASCADE is None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    min_side = max(20, min(height, width) // 10)
    faces = _HAAR_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    if len(faces):
        best = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        return tuple(int(v) for v in best)
    return None


def _face_occupancy(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Return (face-width ratio, face-area ratio) for crop decisions."""
    height, width = int(image_shape[0]), int(image_shape[1])
    _x, _y, face_w, face_h = box
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    return (
        float(face_w) / float(width),
        float(face_w * face_h) / float(width * height),
    )


def _should_auto_face_crop(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> bool:
    width_ratio, area_ratio = _face_occupancy(box, image_shape)
    return width_ratio < DEFAULT_FACE_WIDTH_TRIGGER or area_ratio < DEFAULT_FACE_AREA_TRIGGER


def _square_face_crop(
    img_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    face_fill: float = DEFAULT_FACE_FILL,
    size: int = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """Return a generous square face/head/shoulders crop without upscaling.

    ``size`` is a maximum output edge for compatibility with the old helper.
    The previous implementation always resized every crop to 1024x1024,
    magnifying tiny faces and sometimes producing smooth/featureless Persona
    conditioning. This version only downsamples when the crop itself is larger.
    """
    height, width = img_bgr.shape[:2]
    x, y, box_w, box_h = box
    fill = max(0.25, min(0.70, float(face_fill)))
    side = int(round(max(box_w, box_h) / fill))
    side = max(max(box_w, box_h), min(side, width, height))

    cx = float(x) + float(box_w) / 2.0
    # A slight downward shift retains jaw, neck and shoulders rather than
    # turning the reference into an eye/nose-only crop.
    cy = float(y) + float(box_h) / 2.0 + 0.15 * float(box_h)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    crop = img_bgr[top : top + side, left : left + side]
    if crop.size == 0:
        return img_bgr

    max_edge = max(1, int(size or DEFAULT_MAX_EDGE))
    if max(crop.shape[:2]) > max_edge:
        scale = float(max_edge) / float(max(crop.shape[:2]))
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return crop


def _normalize_long_edge(
    img_bgr: np.ndarray,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> tuple[np.ndarray, bool]:
    """Downscale to ``max_edge`` while preserving AR; never upscale."""
    height, width = img_bgr.shape[:2]
    longest = max(height, width)
    max_edge = max(1, int(max_edge or DEFAULT_MAX_EDGE))
    if longest <= max_edge:
        return img_bgr, False
    scale = float(max_edge) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), True


def _cache_key(
    src: Path,
    box: tuple[int, int, int, int] | None,
    max_edge: int,
    fill: float,
    mode: str,
) -> str:
    payload = (
        f"v2|{src.resolve()}|{src.stat().st_mtime_ns}|{src.stat().st_size}|"
        f"{max_edge}|{fill:.5f}|{box}|{mode}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _write_cached_image(
    src: Path,
    image: np.ndarray,
    *,
    box: tuple[int, int, int, int] | None,
    max_edge: int,
    fill: float,
    mode: str,
    cache_dir: Path | None,
) -> str:
    cache = cache_dir or _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"{_cache_key(src, box, max_edge, fill, mode)}.png"
    if not out_path.exists():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return str(src.resolve())
        out_path.write_bytes(encoded.tobytes())
    return str(out_path.resolve())


def prepare_flux_reference(
    path: str,
    *,
    face_crop: str = "auto",
    crop_size: int = DEFAULT_CROP_SIZE,
    face_fill: float = DEFAULT_FACE_FILL,
    cache_dir: Path | None = None,
) -> tuple[str, bool]:
    """Prepare one FLUX.2 identity reference conservatively.

    ``auto`` only crops when the detected face is small relative to the source.
    ``off`` retains the whole composition. Both modes still normalize oversized
    input to the configured max edge. Small sources/crops are never upscaled.
    ``always``/``tight`` remain available as explicit diagnostic modes.

    Returns ``(prepared_path, face_cropped)`` for backward compatibility.
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"FLUX identity reference image does not exist: {path}")

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        raise ValueError(f"FLUX identity reference image cannot be decoded: {path}")

    mode = str(face_crop or "auto").strip().lower()
    max_edge = max(256, int(crop_size or DEFAULT_MAX_EDGE))
    box = None if mode in _OFF_VALUES else _detect_face(img_bgr)
    crop_applied = False

    if box is not None:
        force_crop = mode in _FORCE_CROP_VALUES
        if force_crop or (mode == "auto" and _should_auto_face_crop(box, img_bgr.shape)):
            img_bgr = _square_face_crop(
                img_bgr,
                box,
                face_fill=face_fill,
                size=max_edge,
            )
            crop_applied = True

    img_bgr, downscaled = _normalize_long_edge(img_bgr, max_edge=max_edge)
    if not crop_applied and not downscaled:
        return str(src.resolve()), False

    prepared = _write_cached_image(
        src,
        img_bgr,
        box=box,
        max_edge=max_edge,
        fill=face_fill,
        mode=mode,
        cache_dir=cache_dir,
    )
    return prepared, crop_applied


def duplicate_anchor(refs: list[str], max_refs: int) -> list[str]:
    """Explicitly duplicate reference slot 0 for non-Persona experiments."""
    if not refs:
        return []
    out = list(refs)
    out.insert(1, refs[0])
    return out[: max(1, int(max_refs))]


def normalize_flux_prompt(prompt: str) -> str:
    """Convert an SDXL ``BREAK`` boundary into ordinary FLUX.2 prose."""
    text = str(prompt or "").strip()
    text = re.sub(r"\s+BREAK\s+", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def augment_identity_prompt(
    prompt: str,
    persona_name: str | None = None,
    *,
    face_focus: bool = False,
) -> str:
    """Ask FLUX.2 to preserve the referenced identity in the requested scene."""
    instruction = (
        "Create a new image of the same person shown in the reference image. "
        "Preserve their identity, facial structure, apparent age, skin tone, hair, "
        "and overall appearance while following the requested scene, pose, clothing, "
        "lighting, and composition. Do not create additional copies of the referenced person."
    )
    if persona_name:
        instruction += f" The saved persona is {persona_name}."
    if face_focus:
        instruction += (
            " Keep the face clearly visible and in sharp focus while respecting the requested framing."
        )
    return f"{instruction}\n\n{normalize_flux_prompt(prompt)}".strip()


def prepare_flux_references(
    paths: list[str],
    *,
    face_crop: str = "auto",
    anchor_dup: bool = False,
    max_refs: int = 5,
    crop_size: int = DEFAULT_CROP_SIZE,
    face_fill: float = DEFAULT_FACE_FILL,
    cache_dir: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Prepare FLUX.2 reference images and return lightweight telemetry."""
    prepared: list[str] = []
    crops = 0
    normalized = 0
    for path in paths:
        original = str(Path(path).expanduser().resolve())
        value, cropped = prepare_flux_reference(
            path,
            face_crop=face_crop,
            crop_size=crop_size,
            face_fill=face_fill,
            cache_dir=cache_dir,
        )
        prepared.append(value)
        crops += int(cropped)
        normalized += int(value != original)

    if anchor_dup:
        prepared = duplicate_anchor(prepared, max_refs)

    mode = str(face_crop or "auto").strip().lower()
    stats = {
        "face_crop_mode": "off" if mode in _OFF_VALUES else mode,
        "reference_face_crops": crops,
        "reference_normalized": normalized,
        "reference_anchor_dup": bool(anchor_dup),
        "reference_source_count": len(paths),
        "reference_prepared": len(prepared),
        "reference_max_edge": int(crop_size),
        "reference_face_fill": float(face_fill),
    }
    return prepared, stats
