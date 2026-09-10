from __future__ import annotations

import pytest
import torch
from PIL import Image

from core.backends import krea2_identity_multiref as multiref


def test_subject_anchor_is_tight_face_reference_without_upscaling():
    source = Image.new("RGB", (2000, 3000), "white")
    subject, meta = multiref.build_subject_anchor(
        source,
        (800, 300, 300, 300),
        max_edge=640,
    )

    assert subject.width == subject.height
    assert max(subject.size) <= 640
    assert meta["crop_box"] is not None
    assert meta["face_box_original"] == [800, 300, 300, 300]
    assert meta["scale"] <= 1.0


def test_subject_anchor_does_not_enlarge_already_small_crop():
    source = Image.new("RGB", (512, 512), "white")
    subject, meta = multiref.build_subject_anchor(
        source,
        (160, 120, 180, 180),
        max_edge=640,
    )

    assert max(subject.size) <= 512
    assert meta["scale"] == pytest.approx(1.0)


def test_multiref_position_ids_add_independent_subject_frame(monkeypatch):
    monkeypatch.setattr(
        multiref,
        "_ACTIVE_SOURCE_LAYOUT",
        [
            {"reference_grid": [2, 3], "offset": [1.0, 0.5], "tokens": 6},
            {"reference_grid": [2, 2], "offset": [1.0, 1.0], "tokens": 4},
        ],
    )

    ids = multiref.multiref_position_ids(
        text_seq_len=2,
        target_grid_h=4,
        target_grid_w=3,
        source_grid_h=2,
        source_grid_w=3,
        source_offset_h=1.0,
        source_offset_w=0.5,
        device="cpu",
    )

    # [2 text | 6 full | 4 subject | 12 target]
    assert ids.shape == (24, 3)
    assert torch.all(ids[2:8, 0] == 1.0)
    assert torch.all(ids[8:12, 0] == 2.0)
    assert torch.all(ids[12:, 0] == 0.0)


def test_multiref_bias_preserves_full_boost_and_strengthens_subject(monkeypatch):
    monkeypatch.setattr(
        multiref,
        "_ACTIVE_SOURCE_LAYOUT",
        [
            {"reference_grid": [1, 2], "tokens": 2},
            {"reference_grid": [1, 1], "tokens": 1},
        ],
    )
    monkeypatch.setattr(multiref, "_ACTIVE_SUBJECT_BOOST", 6.0)

    bias = multiref.multiref_boost_bias(
        text_len=1,
        src_len=3,
        tgt_len=1,
        ref_boost=4.0,
        device="cpu",
        dtype=torch.float32,
    )

    # Sequence: [text | full0 full1 | subject | target]
    logits = bias[0, 0, 4, 1:4]
    assert torch.exp(logits[0]).item() == pytest.approx(4.0)
    assert torch.exp(logits[1]).item() == pytest.approx(4.0)
    assert torch.exp(logits[2]).item() == pytest.approx(6.0)


def test_subject_boost_tracks_stronger_global_identity_setting():
    assert multiref._subject_boost({"ref_boost": 4.0}) == pytest.approx(6.0)
    assert multiref._subject_boost({"ref_boost": 6.0}) == pytest.approx(8.0)
    assert multiref._subject_boost({"ref_boost": 1.0}) == pytest.approx(1.0)
    assert multiref._subject_boost(
        {"ref_boost": 4.0, "subject_ref_boost": 7.5}
    ) == pytest.approx(7.5)
