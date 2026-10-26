"""Focused coverage for the Krea 2 face/skin detailer (TASK-013).

Covers the crop/paste geometry, the compositing math, request resolution, and the
guarantee that a missing detector or an empty face list is a no-op rather than a
failed generation. No GPU, no cv2 model, and no torch required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.backends import krea2_face_detailer as fd


# ---------------------------------------------------------------------------
# Request resolution (host)
# ---------------------------------------------------------------------------


def test_detailer_is_opt_in():
    assert fd.normalize_detailer_request({}) is None
    assert fd.normalize_detailer_request({"face_detailer": {}}) is None
    assert fd.normalize_detailer_request({"face_detailer": {"enabled": False}}) is None
    assert fd.normalize_detailer_request({"face_detailer": {"enabled": "0"}}) is None
    assert fd.normalize_detailer_request({"face_detailer": {"enabled": "off"}}) is None


def test_enabled_detailer_resolves_defaults():
    payload = fd.normalize_detailer_request({"face_detailer": {"enabled": True}})
    assert payload is not None
    assert payload["enabled"] is True
    assert payload["strength"] == fd.DEFAULT_STRENGTH
    assert payload["min_face_px"] == fd.DEFAULT_MIN_FACE_PX
    assert payload["max_faces"] == fd.DEFAULT_MAX_FACES
    assert payload["target_megapixels"] == fd.DEFAULT_TARGET_MEGAPIXELS
    assert payload["detector_path"] is None


def test_detailer_values_are_clamped():
    payload = fd.normalize_detailer_request(
        {
            "face_detailer": {
                "enabled": True,
                "strength": 99,
                "min_face_px": -5,
                "max_faces": 0,
                "target_megapixels": 1000,
                "pad_ratio": "junk",
            }
        }
    )
    assert payload is not None
    assert payload["strength"] == 1.0
    assert payload["min_face_px"] == 8.0
    assert payload["max_faces"] == 1
    assert payload["target_megapixels"] == 4.0
    # Junk falls back to the default instead of raising.
    assert payload["pad_ratio"] == fd.DEFAULT_PAD_RATIO


def test_detailer_payload_is_json_serializable():
    payload = fd.normalize_detailer_request({"face_detailer": {"enabled": "true"}})
    assert json.loads(json.dumps(payload)) == payload


def test_non_dict_config_is_ignored():
    assert fd.normalize_detailer_request({"face_detailer": "yes"}) is None
    assert fd.normalize_detailer_request({"face_detailer": 1}) is None


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------


def test_expand_face_box_pads_around_the_detection():
    box = fd.expand_face_box((100, 100, 40, 40), (400, 400), pad_ratio=0.5)
    assert box == (80, 80, 160, 160)


def test_expand_face_box_clamps_to_the_frame():
    box = fd.expand_face_box((0, 0, 40, 40), (400, 400), pad_ratio=1.0)
    assert box[0] == 0 and box[1] == 0
    assert box[2] <= 400 and box[3] <= 400

    box = fd.expand_face_box((380, 380, 40, 40), (400, 400), pad_ratio=1.0)
    assert box[2] == 400 and box[3] == 400


def test_expand_face_box_keeps_a_positive_extent_at_the_edge():
    # A detection flush against the frame edge must not collapse to an empty crop.
    x0, y0, x1, y1 = fd.expand_face_box((399, 399, 30, 30), (400, 400), pad_ratio=0.0)
    assert x1 > x0 and y1 > y0
    assert (x1, y1) == (400, 400)


def test_expand_face_box_zero_padding_is_identity():
    assert fd.expand_face_box((10, 20, 30, 40), (200, 200), pad_ratio=0.0) == (10, 20, 40, 60)


def test_expand_face_box_survives_degenerate_image():
    x0, y0, x1, y1 = fd.expand_face_box((0, 0, 5, 5), (1, 1), pad_ratio=1.0)
    assert x1 > x0 and y1 > y0


# ---------------------------------------------------------------------------
# Refine target sizing
# ---------------------------------------------------------------------------


def test_refine_target_snaps_to_the_latent_grid():
    for crop in [(60, 80), (101, 97), (37, 43)]:
        w, h = fd.refine_target_size(crop, target_megapixels=1.0)
        assert w % 16 == 0 and h % 16 == 0


def test_refine_target_upscales_a_small_face():
    w, h = fd.refine_target_size((48, 48), target_megapixels=1.0)
    assert w > 48 and h > 48
    assert 0.9e6 <= float(w) * float(h) <= 1.35e6


def test_refine_target_respects_the_edge_cap():
    w, h = fd.refine_target_size((2048, 2048), target_megapixels=4.0, max_edge=1536)
    assert max(w, h) <= 1536
    assert w % 16 == 0 and h % 16 == 0


def test_refine_target_never_shrinks_below_the_grid():
    w, h = fd.refine_target_size((2, 2), target_megapixels=0.05)
    assert w >= 16 and h >= 16


def test_refine_target_preserves_aspect_roughly():
    w, h = fd.refine_target_size((100, 50), target_megapixels=1.0)
    assert 1.7 <= (w / h) <= 2.3


# ---------------------------------------------------------------------------
# Face selection
# ---------------------------------------------------------------------------


def test_small_faces_are_skipped():
    assert fd.plan_face_passes([(10, 10, 20, 20)], (1024, 1024), min_face_px=32) == []


def test_large_closeups_are_skipped():
    # Wider than the max fraction of the frame: already resolved by the main pass.
    assert fd.plan_face_passes([(0, 0, 900, 900)], (1024, 1024), max_face_fraction=0.4) == []


def test_faces_are_selected_largest_first():
    faces = [(10, 10, 60, 60), (100, 100, 200, 200), (300, 300, 90, 90)]
    selected = fd.plan_face_passes(faces, (2048, 2048), min_face_px=32)
    assert selected[0] == (100, 100, 200, 200)
    assert len(selected) == 3


def test_face_budget_keeps_the_largest():
    faces = [(i * 50, 0, 40, 40) for i in range(6)]
    selected = fd.plan_face_passes(faces, (1024, 1024), min_face_px=32, max_faces=2)
    assert len(selected) == 2


def test_no_faces_is_a_no_op():
    assert fd.plan_face_passes([], (1024, 1024)) == []


def test_zero_faces_skipped():
    assert fd.plan_face_passes([(10, 10, 0, 50), (10, 10, 50, 0)], (1024, 1024)) == []


def test_degenerate_image_selects_nothing():
    assert fd.plan_face_passes([(0, 0, 40, 40)], (0, 0)) == []


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def test_feather_alpha_ramps_from_zero_at_the_border():
    alpha = fd.feather_alpha((64, 64), feather_px=8)
    assert alpha.shape == (64, 64)
    assert alpha.dtype == np.float32
    assert alpha[0, 0] == pytest.approx(0.0)
    assert alpha[-1, -1] == pytest.approx(0.0)
    assert alpha[32, 32] == pytest.approx(1.0)


def test_feather_alpha_zero_is_fully_opaque():
    alpha = fd.feather_alpha((16, 16), feather_px=0)
    assert np.all(alpha == 1.0)


def test_feather_alpha_clamps_to_half_the_short_side():
    alpha = fd.feather_alpha((16, 16), feather_px=9999)
    assert alpha.shape == (16, 16)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_color_match_aligns_mean_and_std():
    rng = np.random.default_rng(0)
    original = rng.normal(120.0, 30.0, size=(32, 32, 3)).astype(np.float32)
    refined = (original * 0.6 + 200.0).astype(np.float32)
    matched = fd.color_match(refined, original)
    assert matched.mean() == pytest.approx(original.mean(), abs=0.5)
    assert matched.std() == pytest.approx(original.std(), abs=0.5)


def test_color_match_leaves_a_flat_crop_alone():
    original = np.full((8, 8, 3), 90.0, dtype=np.float32)
    refined = np.full((8, 8, 3), 10.0, dtype=np.float32)
    # Zero variance must not be amplified into a hard contrast change.
    assert np.allclose(fd.color_match(refined, original), refined)


def test_color_match_ignores_shape_mismatch():
    refined = np.zeros((4, 4, 3), dtype=np.float32)
    original = np.zeros((8, 8, 3), dtype=np.float32)
    assert fd.color_match(refined, original).shape == (4, 4, 3)


def test_paste_crop_blends_the_interior_and_spares_the_border():
    base = np.zeros((40, 40, 3), dtype=np.float32)
    crop = np.full((20, 20, 3), 255.0, dtype=np.float32)
    out = fd.paste_crop(base, crop, (10, 10, 30, 30), feather_px=4)

    assert out[20, 20, 0] == pytest.approx(255.0)  # interior fully replaced
    assert out[10, 20, 0] == pytest.approx(0.0)  # border untouched
    # Somewhere on the ramp the blend must be strictly between the two.
    assert 0.0 < out[12, 20, 0] < 255.0


def test_paste_crop_does_not_mutate_the_base():
    base = np.zeros((20, 20, 3), dtype=np.float32)
    fd.paste_crop(base, np.full((10, 10, 3), 255.0, dtype=np.float32), (0, 0, 10, 10))
    assert base.max() == 0.0


def test_paste_crop_ignores_a_mismatched_crop():
    base = np.zeros((20, 20, 3), dtype=np.float32)
    out = fd.paste_crop(base, np.full((5, 5, 3), 255.0, dtype=np.float32), (0, 0, 10, 10))
    assert out.max() == 0.0


def test_paste_crop_ignores_an_empty_box():
    base = np.zeros((20, 20, 3), dtype=np.float32)
    out = fd.paste_crop(base, np.zeros((1, 1, 3), dtype=np.float32), (10, 10, 10, 10))
    assert out.shape == base.shape


# ---------------------------------------------------------------------------
# Failure tolerance (AC #5)
# ---------------------------------------------------------------------------


def test_missing_detector_skips_the_pass(monkeypatch, tmp_path):
    from PIL import Image

    monkeypatch.setenv("WEBBDUCK_KREA2_FACE_DETECTOR", str(tmp_path / "absent.onnx"))
    image = Image.new("RGB", (64, 64), (10, 20, 30))
    out, note = fd.run_face_detailer(None, None, image, {"enabled": True})
    assert out is image
    assert note["skipped"] == "detector-unavailable"
    assert note["faces_refined"] == 0
    assert "error" in note


def test_detector_failure_skips_the_pass(monkeypatch, tmp_path):
    from PIL import Image

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"not a real model")
    monkeypatch.setenv("WEBBDUCK_KREA2_FACE_DETECTOR", str(model))
    image = Image.new("RGB", (64, 64), (10, 20, 30))
    out, note = fd.run_face_detailer(None, None, image, {"enabled": True})
    assert out is image
    assert note["skipped"] in {"detector-failed", "detector-unavailable"}


def test_no_faces_returns_the_original_image(monkeypatch, tmp_path):
    from PIL import Image

    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            return (1, None)  # (retval, faces) - OpenCV's real shape

    monkeypatch.setattr(fd, "resolve_detector_path", lambda *_a, **_k: tmp_path / "m.onnx")
    monkeypatch.setattr(fd, "create_detector", lambda *_a, **_k: _Detector())

    image = Image.new("RGB", (64, 64), (10, 20, 30))
    out, note = fd.run_face_detailer(None, None, image, {"enabled": True})
    assert out is image
    assert note["skipped"] == "no-faces"
    assert note["faces_detected"] == 0


def test_size_gated_faces_are_distinguished_from_a_detector_miss(monkeypatch, tmp_path):
    """A face the size window rejects must not be reported as 'no faces'."""
    from PIL import Image

    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            # One huge face: detected, but wider than max_face_fraction.
            row = np.zeros((1, 15), dtype=np.float32)
            row[0, :5] = (10, 10, 500, 500, 0.95)
            return (1, row)

    monkeypatch.setattr(fd, "resolve_detector_path", lambda *_a, **_k: tmp_path / "m.onnx")
    monkeypatch.setattr(fd, "create_detector", lambda *_a, **_k: _Detector())

    image = Image.new("RGB", (512, 512), (10, 20, 30))
    out, note = fd.run_face_detailer(None, None, image, {"enabled": True})
    assert out is image
    assert note["faces_detected"] == 1
    assert note["skipped"] == "faces-below-threshold"
    assert note["faces_refined"] == 0
    # The rejected box is still reported so the threshold can be diagnosed.
    assert note["boxes"] == [{"face": [10, 10, 500, 500]}]


def test_crop_failure_still_reports_and_keeps_the_image(monkeypatch, tmp_path):
    from PIL import Image

    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            # One face inside the size window, in OpenCV's (retval, faces) shape.
            row = np.zeros((1, 15), dtype=np.float32)
            row[0, :5] = (8, 8, 40, 40, 0.9)
            return (1, row)

    def _boom(*_a, **_k):
        raise RuntimeError("transformer exploded")

    monkeypatch.setattr(fd, "resolve_detector_path", lambda *_a, **_k: tmp_path / "m.onnx")
    monkeypatch.setattr(fd, "create_detector", lambda *_a, **_k: _Detector())
    monkeypatch.setattr(fd, "refine_face", _boom)

    image = Image.new("RGB", (256, 256), (10, 20, 30))
    out, note = fd.run_face_detailer(None, None, image, {"enabled": True})
    assert out is image
    assert note["skipped"] == "crop-failed"
    assert note["faces_refined"] == 0
    assert note["errors"]


def test_detect_faces_unpacks_the_opencv_tuple():
    """OpenCV returns ``(retval, faces)``; iterating it directly yields ints."""

    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            return (1, None)  # (retval, faces) - OpenCV's real shape

    array = np.zeros((32, 32, 3), dtype=np.uint8)
    assert fd.detect_faces(_Detector(), array) == []


def test_detect_faces_reads_x_y_w_h_from_the_row():
    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            row = np.zeros((2, 15), dtype=np.float32)
            row[0, :5] = (10.4, 20.6, 30.2, 40.8, 0.9)
            row[1, :5] = (100.0, 200.0, 50.0, 60.0, 0.8)
            return (2, row)

    array = np.zeros((300, 300, 3), dtype=np.uint8)
    assert fd.detect_faces(_Detector(), array) == [(10, 21, 30, 41), (100, 200, 50, 60)]


def test_detect_faces_tolerates_a_bare_array():
    class _Detector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _array):
            row = np.zeros((1, 15), dtype=np.float32)
            row[0, :5] = (5, 6, 7, 8, 0.9)
            return row

    array = np.zeros((64, 64, 3), dtype=np.uint8)
    assert fd.detect_faces(_Detector(), array) == [(5, 6, 7, 8)]


# ---------------------------------------------------------------------------
# Host wiring
# ---------------------------------------------------------------------------


def test_explicit_detector_path_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_FACE_DETECTOR", raising=False)
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"weights")
    assert fd.resolve_detector_path(str(model)) == model


def test_detector_env_var_is_used(tmp_path, monkeypatch):
    model = tmp_path / "from-env.onnx"
    model.write_bytes(b"weights")
    monkeypatch.setenv("WEBBDUCK_KREA2_FACE_DETECTOR", str(model))
    assert fd.resolve_detector_path() == model
    assert fd.resolve_detector_path(None) == model


def test_detector_path_from_weights_root_is_preferred(tmp_path, monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_FACE_DETECTOR", raising=False)
    monkeypatch.setenv("WEBBDUCK_WEIGHTS_DIR", str(tmp_path))
    cached = tmp_path / "faces" / fd.DEFAULT_DETECTOR_FILE
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    assert fd.resolve_detector_path() == cached


def test_detector_download_uses_the_real_repo_layout(tmp_path, monkeypatch):
    """opencv_zoo nests the file under models/face_detection_yunet/."""
    seen = {}

    def _fake_download(*, repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        staged = tmp_path / "staged.onnx"
        staged.write_bytes(b"downloaded")
        return str(staged)

    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", type("M", (), {"hf_hub_download": _fake_download})
    )
    monkeypatch.setenv("WEBBDUCK_WEIGHTS_DIR", str(tmp_path / "w"))
    monkeypatch.delenv("WEBBDUCK_KREA2_FACE_DETECTOR", raising=False)

    resolved = fd.resolve_detector_path()
    assert seen["repo_id"] == fd.DEFAULT_DETECTOR_REPO
    assert seen["filename"] == (
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    assert resolved.is_file()
    assert resolved.name == fd.DEFAULT_DETECTOR_FILE
    assert resolved.parent.name == "faces"


def test_settings_accept_the_raw_json_payload():
    from server.app import _parse_face_detailer_settings

    assert _parse_face_detailer_settings("") == {}
    assert _parse_face_detailer_settings("   ") == {}
    assert _parse_face_detailer_settings("{bad json") == {}
    assert _parse_face_detailer_settings("[1, 2]") == {}
    assert _parse_face_detailer_settings('{"enabled": true}') == {"enabled": True}


def test_descriptor_advertises_the_capability_for_krea():
    from models.model_descriptor import describe_registry_model

    entry = {
        "architecture": "krea2",
        "path": "lustifyNSFWCheckpoint_v10Krea2Turbo_fp8.safetensors",
        "type": "checkpoint",
    }
    caps = describe_registry_model("lustifyNSFWCheckpoint_v10Krea2Turbo_fp8", entry).capabilities
    assert caps.img2img is True
    assert caps.face_detailer is True


# ---------------------------------------------------------------------------
# Crop refine contract (fake pipeline, no GPU)
# ---------------------------------------------------------------------------


def test_refine_face_returns_a_crop_at_the_original_pixel_size(monkeypatch):
    """The refined crop must come back at the crop's own size so paste is 1:1."""
    import torch

    from core.backends import krea2_img2img as ii
    from core.backends import krea2_worker_impl as impl

    seen = {}

    def _fake_encode(_pipe, _vae, _image, *, width, height, device):
        seen["encode_size"] = (width, height)
        latents = torch.zeros((1, 16, height // 8, width // 8))
        return latents, (height // 8) * (width // 8)

    def _fake_schedule(_pipe, *, steps, strength, image_seq_len, device):
        seen["schedule"] = (steps, strength, image_seq_len)
        return [0.6, 0.3], 0.6, 2

    def _fake_noise(clean, sigma, *, seed, device):
        seen["noise"] = (sigma, seed)
        return clean + float(sigma)

    def _fake_denoise(_pipe, **kwargs):
        seen["denoise"] = kwargs
        return torch.zeros((1, 16, kwargs["height"] // 8, kwargs["width"] // 8))

    def _fake_decode(_pipe, _vae, _latents, *, width, height, device):
        seen["decode_size"] = (width, height)
        from PIL import Image

        # Cool, *textured* output. Texture matters: a flat crop has zero variance
        # and color_match deliberately leaves it alone.
        rng = np.random.default_rng(3)
        noise = rng.integers(-25, 25, size=(height, width, 3))
        cool = np.array([40, 90, 220], dtype=np.float32) + noise
        return [Image.fromarray(cool.clip(0, 255).astype(np.uint8), mode="RGB")]

    monkeypatch.setattr(ii, "encode_source_latent", _fake_encode)
    monkeypatch.setattr(ii, "img2img_schedule", _fake_schedule)
    monkeypatch.setattr(ii, "noise_init_latent", _fake_noise)
    monkeypatch.setattr(ii, "release_vae", lambda *a, **k: None)
    monkeypatch.setattr(impl, "_denoise_one", _fake_denoise)
    monkeypatch.setattr(impl, "_decode_latents", _fake_decode)

    # A warm, textured 50x60 crop taken from a warmer frame: after color match it
    # must read warm again, not like the cool blue the fake decoder produced.
    rng = np.random.default_rng(11)
    frame = np.empty((300, 300, 3), dtype=np.float32)
    frame[:, :, 0] = 210
    frame[:, :, 1] = 150
    frame[:, :, 2] = 120
    frame += rng.integers(-20, 20, size=frame.shape)
    frame = frame.clip(0, 255).astype(np.uint8)

    crop, _info = fd.refine_face(
        None,
        None,
        image_array=frame,
        box=(10, 20, 60, 80),  # 50x60
        face_box=(18, 26, 34, 48),
        config={"strength": 0.4, "target_megapixels": 0.05},
        prompt_embeds=None,
        prompt_mask=None,
        negative_embeds=None,
        negative_mask=None,
        guidance=4.0,
        seed=7,
        device="cpu",
        steps=8,
    )

    assert crop.shape == (60, 50, 3)
    assert crop.dtype == np.uint8
    # Refined up to the grid for the img2img job, then decoded at that size.
    assert seen["encode_size"] == seen["decode_size"]
    assert seen["encode_size"][0] % 16 == 0 and seen["encode_size"][1] % 16 == 0
    # The crop was small, so the refine size must be larger than the crop.
    assert seen["encode_size"][0] >= 50 and seen["encode_size"][1] >= 60
    # Strength flows through, and progress uses the truncated step count.
    assert seen["schedule"][1] == pytest.approx(0.4)
    assert seen["denoise"]["total_denoise_steps"] == 2
    assert seen["denoise"]["init_latents"] is not None
    assert seen["denoise"]["init_sigmas"] == [0.6, 0.3]
    # Color match pulled the cool output back toward the warm original, per channel.
    original_crop = frame[20:80, 10:60].astype(np.float32)
    assert crop.mean(axis=(0, 1)) == pytest.approx(original_crop.mean(axis=(0, 1)), abs=2.0)
    assert crop[:, :, 0].mean() > crop[:, :, 2].mean()


# ---------------------------------------------------------------------------
# Worker lifecycle (structural, matching the repo's existing _run conventions)
# ---------------------------------------------------------------------------


def _run_source() -> str:
    """Read the worker implementation from disk.

    Deliberately not ``inspect.getsource``: that resolves through linecache, so
    another test in a full-suite run can hand back the wrong text and turn these
    source assertions into false failures.
    """
    return (_ROOT / "core" / "backends" / "krea2_worker_impl.py").read_text(encoding="utf-8")


def test_detailer_is_opt_in_at_the_worker_boundary():
    """A missing/disabled payload must not change the established hot path."""
    source = _run_source()
    assert 'request.get("face_detailer")' in source
    assert 'detailer_enabled = bool(detailer_cfg and detailer_cfg.get("enabled"))' in source


def test_transformer_is_released_before_decode_without_the_detailer():
    """Without the detailer we keep releasing the denoiser first (memory behavior)."""
    source = _run_source()
    guard = source.index("if not detailer_enabled:")
    release = source.index("_release_transformer(pipe, device)", guard)
    decode = source.index('report("Decoding Krea image"', guard)
    assert guard < release < decode


def test_transformer_stays_resident_through_the_detailer():
    """With the detailer, the transformer must outlive the decode it refines."""
    source = _run_source()
    guard = source.index("if detailer_enabled:", source.index('report("Decoding Krea image"'))
    refine = source.index("krea2_face_detailer.run_face_detailer", guard)
    release = source.index("_release_transformer(pipe, device)", refine)
    assert guard < refine < release


def test_prompt_embeds_survive_until_the_detailer_runs():
    """The crop re-denoise reuses the already-encoded prompt, so it must not be freed."""
    source = _run_source()
    free_early = source.index("del prompt_embeds, prompt_mask, negative_embeds, negative_mask")
    refine = source.index("krea2_face_detailer.run_face_detailer")
    assert free_early < refine
    # And a late free still exists for the detailer path, i.e. the embeds are held
    # across the refine call instead of being dropped before it.
    assert source.index("del prompt_embeds, prompt_mask, negative_embeds, negative_mask", refine) > refine


def test_detailer_pass_is_fully_guarded():
    source = _run_source()
    call = source.index("krea2_face_detailer.run_face_detailer")
    block = source[call : call + 2000]
    assert "except Exception" in block
    assert "detailer-failed" in block


def test_detailer_is_recorded_in_result_metadata():
    from core.backends import krea2_worker_impl as impl

    source = _run_source()
    assert '"face_detailer": detailer_notes or None' in source
    assert 'timing["face_detailer_seconds"]' in source
    assert hasattr(impl, "_decode_latents")


def test_detailer_enables_vae_tiling_to_bound_decode_memory():
    """Keeping the transformer resident makes decode memory the limiting factor."""
    source = _run_source()
    assert "max(width, height) > 1536 or detailer_enabled" in source


# ---------------------------------------------------------------------------
# UI / server wiring contract (mirrors the realism decoder's contract tests)
# ---------------------------------------------------------------------------


_ROOT = Path(__file__).resolve().parents[1]


def test_detailer_controls_live_in_advanced_not_parameters():
    """The whole point: no new knobs in the main flow, an Advanced escape hatch."""
    html = (_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    app_js = (_ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    state_js = (_ROOT / "ui" / "core" / "state.js").read_text(encoding="utf-8")

    assert 'id="section-advanced"' in html
    assert 'data-section="section-advanced"' in html
    for control in (
        "realism_mode",
        "face_detailer_enabled",
        "face_detailer_strength",
        "face_detailer_min_face_px",
        "face_detailer_max_faces",
        "face_detailer_detector_path",
    ):
        assert f'id="{control}"' in html

    # Advanced starts collapsed.
    line = next(l for l in html.splitlines() if 'id="section-advanced"' in l)
    assert "collapsed" in line
    # The override is always sent so the backend can decide.
    assert "formData.append('realism_mode'" in app_js
    assert "realismMode: 'auto'" in state_js


def test_user_never_needs_to_touch_the_detailer():
    """Auto mode must reach the backend with no Advanced interaction at all."""
    app_js = (_ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    assert "formData.append('realism_mode', realismMode.value || 'auto')" in app_js
    # The explicit flag is what tells the host to stop auto-enabling.
    assert "formData.append('face_detailer_explicit', '1')" in app_js


def test_parameters_shows_style_status_not_knobs():
    """A read-only one-liner in Parameters replaces the old slider block."""
    html = (_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="realism-status-text"' in html
    assert 'id="realism-status-reason"' in html
    # No detailer controls anywhere inside the Parameters section.
    params = html[html.index('id="section-params"'):html.index('id="section-advanced"')]
    assert "face_detailer_strength" not in params
    assert "face_detailer_enabled" not in params


def test_server_forwards_realism_fields():
    source = (_ROOT / "server" / "app.py").read_text(encoding="utf-8")
    assert source.count('face_detailer: str = Form("")') == 2
    assert source.count("_parse_face_detailer_settings(face_detailer)") == 2
    assert source.count('realism_mode: str = Form("auto")') == 2
    assert source.count("face_detailer_explicit: bool = Form(False)") == 2


def test_detailer_slider_labels_are_wired():
    app_js = (_ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    html = (_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    for pair in (
        ("face_detailer_strength", "face-detailer-strength-value"),
        ("face_detailer_min_face_px", "face-detailer-min-face-value"),
        ("face_detailer_max_faces", "face-detailer-max-faces-value"),
    ):
        assert f"['{pair[0]}', '{pair[1]}']" in app_js
        assert f'id="{pair[1]}"' in html


def test_face_feature_mask_spares_the_crop_corners():
    """Hair/background live in the corners and must not be regenerated."""
    mask = fd.face_feature_mask((462, 539), (126, 126, 210, 258), feather_px=12.0)
    assert mask.shape == (539, 462)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    for corner in ((8, 8), (8, 453), (530, 8), (530, 453)):
        assert mask[corner] == pytest.approx(0.0)
    # The facial centre is fully replaced.
    assert mask[126 + int(258 * 0.56), 126 + 210 // 2] == pytest.approx(1.0)


def test_face_feature_mask_leaves_most_of_the_crop_alone():
    """Only a small fraction should change, or the likeness drifts."""
    mask = fd.face_feature_mask((462, 539), (126, 126, 210, 258), feather_px=12.0)
    changed = float((mask > 0.5).mean())
    assert changed < 0.20


def test_face_feature_mask_is_symmetric_about_the_face_centre():
    mask = fd.face_feature_mask((200, 200), (50, 50, 100, 100), feather_px=8.0)
    assert mask[100, 100] == pytest.approx(1.0)
    assert mask[100, 100] >= mask[100, 40] >= mask[100, 10]


def test_face_feature_mask_handles_a_degenerate_box():
    mask = fd.face_feature_mask((64, 64), (10, 10, 0, 0), feather_px=8.0)
    assert mask.shape == (64, 64)
    assert np.all(np.isfinite(mask))


def test_face_feature_mask_zero_feather_is_still_bounded():
    mask = fd.face_feature_mask((128, 128), (20, 20, 80, 80), feather_px=0.0)
    assert mask.shape == (128, 128)
    assert mask.max() <= 1.0 and mask.min() >= 0.0


def test_refine_face_keeps_surrounding_context(monkeypatch):
    """The refined crop must equal the original outside the facial mask."""
    import torch

    from core.backends import krea2_img2img as ii
    from core.backends import krea2_worker_impl as impl

    def _fake_encode(_p, _v, _img, *, width, height, device):
        return torch.zeros((1, 16, height // 8, width // 8)), 64

    def _fake_decode(_p, _v, _l, *, width, height, device):
        from PIL import Image

        # Completely different content: anything surviving must come from masking.
        return [Image.new("RGB", (width, height), (5, 250, 5))]

    monkeypatch.setattr(ii, "encode_source_latent", _fake_encode)
    monkeypatch.setattr(ii, "img2img_schedule", lambda *a, **k: ([0.5], 0.5, 1))
    monkeypatch.setattr(ii, "noise_init_latent", lambda c, s, **k: c + float(s))
    monkeypatch.setattr(ii, "release_vae", lambda *a, **k: None)
    monkeypatch.setattr(impl, "_denoise_one", lambda _p, **k: torch.zeros((1, 16, k["height"] // 8, k["width"] // 8)))
    monkeypatch.setattr(impl, "_decode_latents", _fake_decode)

    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    crop, _info = fd.refine_face(
        None, None,
        image_array=frame,
        box=(50, 50, 350, 350),
        face_box=(120, 120, 200, 200),
        config={"strength": 0.3, "target_megapixels": 0.05, "feather_px": 10.0},
        prompt_embeds=None, prompt_mask=None,
        negative_embeds=None, negative_mask=None,
        guidance=0.0, seed=1, device="cpu", steps=4,
    )
    assert crop.shape == (300, 300, 3)
    # Corners (hair/background) are untouched original grey.
    assert crop[2, 2].tolist() == [128, 128, 128]
    assert crop[-3, -3].tolist() == [128, 128, 128]
    # The facial centre took the refinement instead.
    assert crop[150, 150].tolist() != [128, 128, 128]


def test_capabilities_keep_decoder_knobs_out_of_parameters():
    """Automatic routing means the decoder selects itself; don't re-expose it."""
    caps_js = (_ROOT / "ui" / "core" / "modelCapabilities.js").read_text(encoding="utf-8")
    body = caps_js[caps_js.index("function applyFeatureSections"):]
    body = body[:body.index("\n}\n")] if "\n}\n" in body else body
    assert "setControlVisible('vae_decoder', false)" in body
    assert "setControlVisible('vae_decoder', realismDecoder)" not in body
    # ...and the Advanced controls must actually be revealed for capable models.
    for control in ("realism-status-text", "realism_mode", "face_detailer_enabled"):
        assert f"setControlVisible('{control}'," in body


def test_advanced_group_is_revealed_not_permanently_hidden():
    html = (_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    # The group ships hidden and is unhidden by capability gating; if that gating
    # is removed the controls become unreachable.
    assert 'class="form-group hidden" id="face-detailer-group"' in html
    caps_js = (_ROOT / "ui" / "core" / "modelCapabilities.js").read_text(encoding="utf-8")
    assert "capabilityEnabled(caps, 'face_detailer')" in caps_js


def test_realism_status_is_wired_to_the_preview_endpoint():
    app_js = (_ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    assert "/realism/preview?" in app_js
    assert "function updateRealismStatus()" in app_js
    assert "updateTokenCounter(promptEl.value), 300" in app_js
    # Prompt, negative, mode and model changes all refresh the read-out.
    for id_ in ("'prompt'", "'negative'", "'realism_mode'", "'base_model'"):
        assert id_ in app_js


def test_server_exposes_realism_preview_endpoint():
    source = (_ROOT / "server" / "app.py").read_text(encoding="utf-8")
    assert '@app.get("/realism/preview")' in source
    assert "from core.realism_profile import resolve_realism" in source


def test_blank_decoder_fields_do_not_block_automatic_selection():
    """Regression: the Studio always submits vae_decoder, as "" when hidden.

    setdefault() kept that empty string, so resolve_decoder_request() returned
    None and the realism decoder silently never auto-selected in production.
    """
    host = (_ROOT / "core" / "backends" / "krea2_host_impl.py").read_text(encoding="utf-8")
    assert 'if not str(auto_settings.get(key) or "").strip():' in host
    assert 'auto_settings.setdefault("vae_decoder"' not in host
    assert 'auto_settings.setdefault("vae_decoder_output"' not in host


def test_auto_detailer_uses_the_documented_conservative_strength():
    """Auto mode must use the gentle refine budget, not the repainting default."""
    host = (_ROOT / "core" / "backends" / "krea2_host_impl.py").read_text(encoding="utf-8")
    block = host[host.index('auto_settings["face_detailer"] = {'):]
    block = block[:block.index("else:")]
    assert f'"refine_strength": {fd.DEFAULT_REFINE_STRENGTH}' in block
    assert f'"refine_steps": {fd.DEFAULT_REFINE_STEPS}' in block
    # The legacy repainting key must not be reintroduced into the auto path.
    assert '"strength"' not in block
    assert '"enabled": True' in block
    assert '"max_faces": 3' in block


def test_refine_defaults_start_below_the_old_repainting_noise_floor():
    """Guard the regression that produced coloured discs on faces.

    Reusing the main pass's step count put the refine at shifted sigma ~0.575
    with 3 steps, which re-rendered the face and shifted skin blue. The default
    budget must stay well under that floor while still executing steps.
    """
    import math

    mu = 1.15
    shift = math.exp(mu)

    def shifted(base_sigma: float) -> float:
        return shift * base_sigma / (1 + (shift - 1) * base_sigma)

    steps = fd.DEFAULT_REFINE_STEPS
    init_steps = min(int(steps * fd.DEFAULT_REFINE_STRENGTH), steps)
    t_start = steps - init_steps
    base_sigma = 1.0 - t_start * (1.0 / steps)
    init_sigma = shifted(base_sigma)

    assert init_sigma < 0.40, init_sigma
    assert init_steps >= 3, init_steps
    # The old auto path: main steps=10 at strength 0.30.
    old_steps, old_strength = 10, 0.30
    old_t = old_steps - min(int(old_steps * old_strength), old_steps)
    old_sigma = shifted(1.0 - old_t * (1.0 / old_steps))
    assert init_sigma < old_sigma * 0.6, (init_sigma, old_sigma)
