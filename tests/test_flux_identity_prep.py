from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.backends import flux_identity
from core.backends.flux import _resolve_flux_identity


def _image(path: Path, color: tuple[int, int, int] = (120, 120, 120), size: int = 256) -> Path:
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _compress(paths: list[Path]) -> list[str]:
    return [str(p.resolve()) for p in paths]


def test_duplicate_anchor_keeps_anchor_first():
    refs = ["a", "b", "c"]
    assert flux_identity.duplicate_anchor(refs, max_refs=5) == ["a", "a", "b", "c"]


def test_duplicate_anchor_caps_total_at_max():
    refs = ["a", "b", "c", "d", "e"]
    out = flux_identity.duplicate_anchor(refs, max_refs=5)
    assert out[0] == "a" and out[1] == "a"
    assert len(out) == 5
    assert out == ["a", "a", "b", "c", "d"]


def test_duplicate_anchor_empty():
    assert flux_identity.duplicate_anchor([], max_refs=5) == []
    assert flux_identity.duplicate_anchor(["solo"], max_refs=5) == ["solo", "solo"]


def test_augment_identity_prompt_identity_preserved_without_face_focus():
    prompt = flux_identity.augment_identity_prompt("A studio portrait")
    assert "same person shown in the reference images" in prompt
    assert "A studio portrait" in prompt
    assert "large in frame" not in prompt


def test_augment_identity_prompt_face_focus_appends_closeup():
    prompt = flux_identity.augment_identity_prompt(
        "A studio portrait",
        persona_name="Della",
        face_focus=True,
    )
    assert "The saved persona is Della." in prompt
    assert "Face is in frame, centered, sharp focus on facial features" in prompt


def test_face_crop_off_returns_original(monkeypatch, tmp_path):
    src = _image(tmp_path / "ref.png")
    prepared, cropped = flux_identity.prepare_flux_reference(
        str(src), face_crop="off", cache_dir=tmp_path / "cache"
    )
    assert prepared == str(src.resolve())
    assert cropped is False


def test_no_face_detected_keeps_original(monkeypatch, tmp_path):
    monkeypatch.setattr(flux_identity, "_detect_face", lambda img: None)
    src = _image(tmp_path / "ref.png")
    prepared, cropped = flux_identity.prepare_flux_reference(
        str(src), face_crop="auto", cache_dir=tmp_path / "cache"
    )
    assert prepared == str(src.resolve())
    assert cropped is False


def test_face_crop_squares_and_caches(monkeypatch, tmp_path):
    src = _image(tmp_path / "ref.png", size=800)
    monkeypatch.setattr(
        flux_identity,
        "_detect_face",
        lambda img: (300, 300, 400, 400),
    )
    cache = tmp_path / "cache"
    prepared, cropped = flux_identity.prepare_flux_reference(
        str(src), face_crop="auto", cache_dir=cache
    )
    assert cropped is True
    assert Path(prepared).is_file()
    img = cv2.imread(prepared)
    assert img.shape[0] == 1024 and img.shape[1] == 1024

    cache_files = list(cache.glob("*.png"))
    assert len(cache_files) == 1

    prepared2, cropped2 = flux_identity.prepare_flux_reference(
        str(src), face_crop="auto", cache_dir=cache
    )
    assert prepared2 == prepared
    assert cropped2 is True
    assert len(list(cache.glob("*.png"))) == 1


def test_face_crop_pads_with_replication_not_squish(monkeypatch, tmp_path):
    edge = np.zeros((800, 800, 3), dtype=np.uint8)
    edge[:, :200] = (255, 0, 0)
    edge[:, 200:] = (0, 255, 0)
    src = tmp_path / "edge.png"
    cv2.imwrite(str(src), edge)
    # Face box pushed hard against the left edge so the crop must pad.
    monkeypatch.setattr(flux_identity, "_detect_face", lambda img: (0, 350, 200, 200))
    prepared, cropped = flux_identity.prepare_flux_reference(
        str(src), face_crop="auto", cache_dir=tmp_path / "cache"
    )
    assert cropped is True
    out = cv2.imread(prepared)
    assert out.shape[0] == 1024 and out.shape[1] == 1024
    left_col = out[:, 0, :]
    assert (left_col == (255, 0, 0)).all()


def test_prepare_flux_references_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(
        flux_identity,
        "_detect_face",
        lambda img: (30, 30, 80, 80),
    )
    refs = [
        _image(tmp_path / "a.png"),
        _image(tmp_path / "b.png"),
        _image(tmp_path / "c.png"),
    ]
    prepared, stats = flux_identity.prepare_flux_references(
        _compress(refs),
        face_crop="auto",
        anchor_dup=True,
        max_refs=5,
        cache_dir=tmp_path / "cache",
    )
    assert len(prepared) == 4
    assert prepared[0] == prepared[1]
    assert stats["reference_face_crops"] == 3
    assert stats["reference_anchor_dup"] is True
    assert stats["face_crop_mode"] == "auto"
    assert stats["reference_prepared"] == 4


def test_prepare_flux_references_anchor_dup_respects_worker_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(flux_identity, "_detect_face", lambda img: None)
    refs = [_image(tmp_path / f"r{i}.png") for i in range(5)]
    prepared, stats = flux_identity.prepare_flux_references(
        _compress(refs),
        face_crop="auto",
        anchor_dup=True,
        max_refs=5,
        cache_dir=tmp_path / "cache",
    )
    assert len(prepared) == 5
    assert prepared[0] == prepared[1]
    assert len(set(prepared)) == 4


def test_resolve_flux_identity_duplicates_anchor_and_reports_debug(tmp_path):
    a = _image(tmp_path / "a.png")
    b = _image(tmp_path / "b.png")
    settings = {
        "identity_adapter": {
            "enabled": True,
            "preset_name": "Della",
            "reference_images": [str(a), str(b)],
            "face_crop": "off",
            "flux2_anchor_dup": True,
            "face_focus": True,
        }
    }
    refs, persona_name = _resolve_flux_identity(settings)
    assert persona_name == "Della"
    assert refs == [str(a.resolve()), str(a.resolve()), str(b.resolve())]
    debug = settings["identity_adapter_debug"]
    assert debug["face_crop_mode"] == "off"
    assert debug["reference_anchor_dup"] is True
    assert debug["reference_face_crops"] == 0
    assert debug["reference_prepared"] == 3
    assert debug["reference_images_resolved"] == 3


def test_resolve_flux_identity_defaults_to_auto_face_crop(tmp_path):
    a = _image(tmp_path / "a.png")
    settings = {
        "identity_adapter": {
            "enabled": True,
            "reference_images": [str(a)],
        }
    }
    refs, _ = _resolve_flux_identity(settings)
    # No synthetic face detected -> original passed through.
    assert refs == [str(a.resolve())]
    assert settings["identity_adapter_debug"]["face_crop_mode"] == "auto"
