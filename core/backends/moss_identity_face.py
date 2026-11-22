"""Face-only reference preparation for MOSS-Image identity conditioning.

MOSS-Image's edit adapter copies whatever the reference photo shows: background,
ears, clothing and pose come through along with the face. For Persona work the
useful signal is the face, so by default the reference is cropped around the
detected face and everything outside a soft oval is replaced with neutral gray.
Images with no detectable face are passed through unchanged.

Measured on Dirty-MossV2 (768^2, 50 steps, seed 1234, armV5): full photo 0.602
face similarity with the costume copied; masked face 0.681 with the costume gone.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

FACE_CROP_MODES = ("face", "full")
DEFAULT_FACE_CROP = "face"
FACE_FILL = 0.60
OUTPUT_SIDE = 1024
_GRAY = 128


def _cache_dir() -> Path:
    base = os.getenv("WEBBDUCK_OUTPUT_DIR")
    root = (
        Path(base) / ".moss_identity_cache"
        if base
        else Path(tempfile.gettempdir()) / "webbduck_moss_identity"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(src: Path) -> Path:
    stat = src.stat()
    payload = f"v1|{src.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{FACE_FILL}|{OUTPUT_SIDE}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return _cache_dir() / f"{digest}.png"


def _largest_face(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    from core.backends.flux_identity import _detect_face

    box = _detect_face(img_bgr)
    if box is None:
        return None
    x, y, w, h = box
    return int(x), int(y), int(w), int(h)


def _face_only(img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = img_bgr.shape[:2]
    x, y, box_w, box_h = box
    cx = x + box_w / 2.0
    cy = y + box_h / 2.0
    side = int(max(box_w, box_h) / FACE_FILL)
    side = max(max(box_w, box_h), min(side, width, height))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0 + 0.1 * box_h))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    crop = img_bgr[top : top + side, left : left + side]

    mask = np.zeros(crop.shape[:2], np.float32)
    ch, cw = crop.shape[:2]
    cv2.ellipse(
        mask,
        (int(cw * 0.50), int(ch * 0.46)),
        (int(cw * 0.34), int(ch * 0.42)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, ch * 0.02))[..., None]
    out = (crop.astype(np.float32) * mask + _GRAY * (1.0 - mask)).astype(np.uint8)
    return cv2.resize(out, (OUTPUT_SIDE, OUTPUT_SIDE), interpolation=cv2.INTER_AREA)


def prepare_face_reference(path: str | Path) -> tuple[str, bool]:
    """Return ``(reference_path, masked)`` for one MOSS identity reference.

    ``masked`` is False when no face was found and the original file is used.
    Prepared images are cached per source file, so repeated generations reuse
    the same reference.
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"MOSS-Image identity reference does not exist: {path}")

    cached = _cache_path(src)
    if cached.is_file():
        return str(cached), True

    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"MOSS-Image identity reference cannot be decoded: {path}")
    box = _largest_face(img)
    if box is None:
        return str(src.resolve()), False

    cv2.imwrite(str(cached), _face_only(img, box))
    return str(cached), True
