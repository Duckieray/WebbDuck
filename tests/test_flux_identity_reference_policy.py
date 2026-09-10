from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.backends.flux_identity import (
    DEFAULT_FACE_FILL,
    _normalize_long_edge,
    _should_auto_face_crop,
    _square_face_crop,
    normalize_flux_prompt,
    prepare_flux_reference,
)


def test_large_reference_downscales_without_changing_aspect_ratio(tmp_path: Path):
    src = tmp_path / "large.png"
    image = np.zeros((2000, 3000, 3), dtype=np.uint8)
    assert cv2.imwrite(str(src), image)

    prepared, cropped = prepare_flux_reference(str(src), face_crop="off", crop_size=1024)
    result = cv2.imread(prepared)

    assert cropped is False
    assert result is not None
    assert result.shape[:2] == (683, 1024)


def test_small_reference_is_never_upscaled(tmp_path: Path):
    src = tmp_path / "small.png"
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cv2.imwrite(str(src), image)

    prepared, cropped = prepare_flux_reference(str(src), face_crop="off", crop_size=1024)

    assert cropped is False
    assert prepared == str(src.resolve())


def test_auto_crop_only_triggers_when_face_is_small():
    shape = (1000, 1000, 3)
    assert _should_auto_face_crop((400, 300, 300, 300), shape) is False
    assert _should_auto_face_crop((460, 360, 100, 100), shape) is True


def test_face_crop_is_generous_and_does_not_upscale():
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    cropped = _square_face_crop(
        image,
        (300, 250, 160, 160),
        face_fill=DEFAULT_FACE_FILL,
        size=1024,
    )
    # 160 / .42 ~= 381px: preserve that native crop instead of inflating it to 1024.
    assert 360 <= cropped.shape[0] <= 400
    assert cropped.shape[0] == cropped.shape[1]


def test_long_edge_helper_never_upscales():
    image = np.zeros((512, 768, 3), dtype=np.uint8)
    normalized, changed = _normalize_long_edge(image, 1024)
    assert changed is False
    assert normalized.shape == image.shape


def test_flux_prompt_removes_sdxl_break():
    assert normalize_flux_prompt("portrait BREAK bedroom lighting") == "portrait. bedroom lighting"
