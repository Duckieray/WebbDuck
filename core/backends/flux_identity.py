"""Generic FLUX.2 identity-reference preprocessing.

Three behind-the-scenes upgrades are baked into WebbDuck's native FLUX.2 persona
adapter, each validated by A/B sweeps on the FLUX.2 klein runtime:

1. ``face_crop`` ("auto" | "off"): every reference is re-framed around the
   strongest detected face as a tight square portrait (~80% face fill), padded
   with ``BORDER_REPLICATE`` (never squished), resized to a fixed square, and
   cached on disk. Whole-body/wide refs that used to dilute identity conditioning
   now land the face reliably. When no face is detected (non-personal assets),
   the original is passed through untouched.

2. ``flux2_anchor_dup`` (bool): FLUX.2 conditions strongest on the earliest
   reference slot (the first reference receives the smallest time offset), so
   the persona's strongest photo should always be first. Duplicating that anchor
   into slot 1 weights it roughly 2x. The total is capped at
   ``_MAX_FLUX_IDENTITY_REFERENCES`` by dropping trailing references.

3. ``face_focus`` (bool): appends close-up portrait framing guidance to the
   identity-augmented prompt so the scene composer keeps the face large,
   centered, and in sharp focus (larger in-frame faces score materially higher
   against identity gates).

Face detection prefers InsightFace ("buffalo_l") when present and falls back to
OpenCV's bundled Haar cascade. Both are CPU-only and deterministic, so cached
crops stay stable across runs.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_CROP_SIZE = 1024
DEFAULT_FACE_FILL = 0.80


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
    """Return the ``(x, y, w, h)`` box of the strongest face, or None."""
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
        faces = _FACE_APP.get(img_bgr)
        if faces:
            best = max(faces, key=lambda f: (float(f.det_score), f.bbox[2] * f.bbox[3]))
            x1, y1, x2, y2 = (float(v) for v in best.bbox)
            return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    if _HAAR_CASCADE is None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    min_side = max(20, min(height, width) // 8)
    faces = _HAAR_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )
    if len(faces):
        best = max(faces, key=lambda f: f[2] * f[3])
        return tuple(int(v) for v in best)
    return None


def _square_face_crop(
    img_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    face_fill: float = DEFAULT_FACE_FILL,
    size: int = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """Tight square crop around the face centre, padded (never squished)."""
    height, width = img_bgr.shape[:2]
    x, y, box_w, box_h = box
    cx = x + box_w / 2.0
    cy = y + box_h / 2.0
    side = int(round(max(box_w, box_h) / face_fill))
    left = int(cx - side // 2)
    top = int(cy - side // 2)
    right = left + side
    bottom = top + side

    crop = img_bgr[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if crop.size == 0:
        return img_bgr

    pad_top = max(0, -top)
    pad_bottom = max(0, bottom - height)
    pad_left = max(0, -left)
    pad_right = max(0, right - width)
    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )

    if crop.shape[0] != side or crop.shape[1] != side:
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    interpolation = cv2.INTER_AREA if side >= size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def _cache_key(src: Path, box: tuple[int, int, int, int], size: int, fill: float) -> str:
    payload = f"{src.resolve()}|{src.stat().st_mtime_ns}|{size}|{fill}|{box}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def prepare_flux_reference(
    path: str,
    *,
    face_crop: str = "auto",
    crop_size: int = DEFAULT_CROP_SIZE,
    face_fill: float = DEFAULT_FACE_FILL,
    cache_dir: Path | None = None,
) -> tuple[str, bool]:
    """Re-frame one identity reference around its strongest face.

    ``face_crop`` accepts "auto" (crop when a face is detected, otherwise pass
    the original through) or "off" (never crop). Returns ``(prepared_abs_path,
    cropped)`` where ``cropped`` reports whether a face re-frame was applied.
    Face detection failures never raise; the original path is returned.
    """
    if face_crop in ("off", "false", "0", ""):
        return str(Path(path).resolve()), False

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"FLUX identity reference image does not exist: {path}")

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        raise ValueError(f"FLUX identity reference image cannot be decoded: {path}")

    box = _detect_face(img_bgr)
    if box is None:
        return str(src.resolve()), False

    cache = cache_dir or _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"{_cache_key(src, box, crop_size, face_fill)}.png"
    if not out_path.exists():
        try:
            cropped = _square_face_crop(img_bgr, box, face_fill=face_fill, size=crop_size)
            ok, encoded = cv2.imencode(".png", cropped)
            if not ok:
                return str(src.resolve()), False
            out_path.write_bytes(encoded.tobytes())
        except Exception:
            return str(src.resolve()), False
    return str(out_path), True


def duplicate_anchor(
    refs: list[str],
    max_refs: int,
) -> list[str]:
    """Duplicate the first (anchor) reference into slot 1, capped at ``max_refs``.

    FLUX.2 weights earliest reference slots strongest, so weighing the persona
    anchor 2x keeps the identity channel dominant. Trailing references are
    dropped first when the cap is hit.
    """
    if not refs:
        return []
    out = list(refs)
    out.insert(1, refs[0])
    while len(out) > max_refs:
        out.pop()
    return out


def augment_identity_prompt(
    prompt: str,
    persona_name: str | None = None,
    *,
    face_focus: bool = False,
) -> str:
    """Ask FLUX.2 to preserve the referenced identity, optionally close-up.

    Mirrors the generic identity instruction with an optional close-up portrait
    framing suffix for the scene so the face stays large, centered, and sharp.
    """
    instruction = (
        "Create a new image of the same person shown in the reference images. "
        "Preserve their identity, facial structure, apparent age, skin tone, hair, "
        "and overall appearance while following the requested scene, pose, clothing, "
        "lighting, and composition."
    )
    if persona_name:
        instruction += f" The saved persona is {persona_name}."
    if face_focus:
        instruction += (
            " Face is in frame, centered, sharp focus on facial features,"
            " ears and hairline clearly visible, eyes looking at the camera,"
            " subject fills most of the frame."
        )
    return f"{instruction}\n\n{str(prompt or '').strip()}".strip()


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
    """Prepare a full reference list for conditioning.

    Returns ``(prepared_paths, stats)`` where ``stats`` records how many
    references were re-framed around a face and whether the anchor was boosted.
    """
    prepared: list[str] = []
    crops = 0
    for path in paths:
        value, cropped = prepare_flux_reference(
            path,
            face_crop=face_crop,
            crop_size=crop_size,
            face_fill=face_fill,
            cache_dir=cache_dir,
        )
        prepared.append(value)
        crops += int(cropped)

    if anchor_dup:
        prepared = duplicate_anchor(prepared, max_refs)

    stats = {
        "face_crop_mode": "auto" if face_crop not in ("off", "false", "0", "") else "off",
        "reference_face_crops": crops,
        "reference_anchor_dup": bool(anchor_dup),
        "reference_prepared": len(prepared),
    }
    return prepared, stats
