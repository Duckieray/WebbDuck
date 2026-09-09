from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from core.backends import krea2
from core.backends import krea2_identity_face as face
from core.backends.krea2_identity_reference import (
    face_occupancy,
    prepare_identity_reference,
    should_auto_face_crop,
)


def test_small_face_reference_auto_crops_before_downscale():
    source = Image.new("RGB", (2000, 3000), "white")
    detected = (800, 300, 300, 300)

    prepared, meta = prepare_identity_reference(
        source,
        max_long_edge=1024,
        auto_face_crop=True,
        face_box=detected,
        detector="test",
    )

    assert meta["face_detected"] is True
    assert meta["auto_face_cropped"] is True
    assert meta["crop_box"] is not None
    assert prepared.width == prepared.height
    assert max(prepared.size) <= 1024
    assert meta["prepared_face_width_ratio"] > meta["face_width_ratio"]
    assert meta["face_box_prepared"] is not None


def test_large_reference_without_small_face_only_downscales():
    source = Image.new("RGB", (3000, 2000), "white")
    # Large enough that the automatic crop should not trigger.
    detected = (900, 450, 900, 900)

    prepared, meta = prepare_identity_reference(
        source,
        max_long_edge=1024,
        auto_face_crop=True,
        face_box=detected,
        detector="test",
    )

    assert meta["auto_face_cropped"] is False
    assert meta["downscaled"] is True
    assert prepared.size == (1024, 683)


def test_face_crop_trigger_uses_relative_face_size():
    small = (400, 100, 120, 120)
    large = (250, 100, 400, 400)
    size = (1000, 1000)
    assert should_auto_face_crop(small, size) is True
    assert should_auto_face_crop(large, size) is False
    width_ratio, area_ratio = face_occupancy(small, size)
    assert width_ratio == pytest.approx(0.12)
    assert area_ratio == pytest.approx(0.0144)


def test_face_token_weights_are_soft_and_grid_aligned():
    source = Image.new("RGB", (1024, 1024), "white")
    weights = face.build_face_token_weights(
        source,
        (350.0, 250.0, 300.0, 300.0),
        target_width=1024,
        target_height=1024,
        source_grid_w=64,
        source_grid_h=64,
        fit_mode="fit",
    )

    assert len(weights) == 64 * 64
    assert min(weights) >= 0.0
    assert max(weights) <= 1.0
    assert max(weights) > 0.9
    assert 0.02 < (sum(weights) / len(weights)) < 0.60
    assert any(0.0 < value < 1.0 for value in weights)


def test_faceaware_bias_blends_background_and_face_boosts(monkeypatch):
    monkeypatch.setattr(face, "_ACTIVE_FACE_TOKEN_WEIGHTS", [0.0, 1.0])
    monkeypatch.setattr(face, "_ACTIVE_BACKGROUND_BOOST", 2.0)
    monkeypatch.setattr(face, "_ACTIVE_FACE_BOOST", 6.0)

    bias = face.faceaware_ref_boost_bias(
        text_len=1,
        src_len=2,
        tgt_len=1,
        ref_boost=4.0,
        device="cpu",
        dtype=torch.float32,
    )

    # Sequence is [text | source0 | source1 | target]. Target->source logits
    # should exponentiate back to the configured 2x and 6x attention boosts.
    source_logits = bias[0, 0, 3, 1:3]
    assert torch.exp(source_logits[0]).item() == pytest.approx(2.0)
    assert torch.exp(source_logits[1]).item() == pytest.approx(6.0)


def test_high_detail_portrait_budget_uses_3072_only_with_strong_headroom():
    assert krea2._recommended_identity_token_budget(
        15.51, 13.2, width=1024, height=1536
    ) == 3072
    assert krea2._recommended_identity_token_budget(
        15.51, 13.2, width=1024, height=1024
    ) == 3072
    # Same healthy card but landscape keeps the already-validated 2688 tier.
    assert krea2._recommended_identity_token_budget(
        15.51, 13.2, width=1536, height=768
    ) == 2688
    # Moderate desktop pressure must not jump to the high-detail tier.
    assert krea2._recommended_identity_token_budget(
        15.51, 10.2, width=1024, height=1536
    ) == 2048
