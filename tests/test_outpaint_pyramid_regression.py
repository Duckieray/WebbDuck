"""Deterministic regression tests for pyramid smart-extend geometry logic."""

import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from modes import outpaint as pyramid_outpaint


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakeInpaint:
    """Cheap inpaint stub: paints masked pixels with deterministic solid colors."""

    device = "cpu"

    def __init__(self):
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        base = kwargs["image"].convert("RGB")
        mask = kwargs["mask_image"].convert("L")
        color = (
            (self.call_count * 53) % 255,
            (self.call_count * 97) % 255,
            (self.call_count * 149) % 255,
        )
        overlay = Image.new("RGB", base.size, color)
        out = Image.composite(overlay, base, mask)
        return _FakeResult(out)


class _FakeGenerator:
    def __init__(self, seed=12345):
        self._seed = int(seed)

    def initial_seed(self):
        return self._seed


def _make_source_image(width, height):
    """Build a deterministic non-flat source image."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    band_h = max(8, height // 24)
    for y in range(0, height, band_h):
        color = ((y * 5) % 255, (y * 3) % 255, (200 + y) % 255)
        draw.rectangle((0, y, width, min(height, y + band_h)), fill=color)
    return image


def _load_pass_events(debug_dir):
    events_path = Path(debug_dir) / "pyramid" / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    return [e for e in events if e.get("type") == "pyramid_pass"]


def test_captured_payload_pass_count_regression():
    """Lock pass counts from captured small/medium/large payload geometry."""
    cases = [
        ("small", (832, 1216), (1584, 1840), 2),
        ("medium", (832, 1216), (1728, 2176), 2),
        ("large", (832, 1216), (2464, 3072), 4),
    ]
    for _name, (src_w, src_h), (dst_w, dst_h), expected_passes in cases:
        fractions = pyramid_outpaint._build_step_fractions(src_w, src_h, dst_w, dst_h, growth=1.5)
        assert len(fractions) == expected_passes
        assert fractions[-1] == 1.0
        assert all(a < b for a, b in zip(fractions[:-1], fractions[1:]))


def test_canvas_initializer_auto_regression_from_large_payload():
    """Large captured growth margins should resolve to soft-context initialization."""
    mode = pyramid_outpaint._resolve_canvas_initializer(
        "auto",
        left_added=218,
        right_added=102,
        top_added=40,
        bottom_added=328,
    )
    assert mode == "soft_context"
    assert pyramid_outpaint._resolve_canvas_initializer("auto", 20, 12, 8, 24) == "edge_strips"
    assert pyramid_outpaint._resolve_canvas_initializer("edge_strips", 200, 200, 200, 200) == "edge_strips"


def test_min_steps_for_strength_guard():
    """Low strengths must not yield zero effective inpaint steps."""
    assert pyramid_outpaint._ensure_min_steps_for_strength(6, 0.12) >= 9
    assert pyramid_outpaint._ensure_min_steps_for_strength(6, 0.50) >= 6
    assert pyramid_outpaint._ensure_min_steps_for_strength(1, 0.01) >= 100
    assert pyramid_outpaint._ensure_min_steps_for_strength(6, 0.0) == 6


def test_large_ratio_stage_lock_is_preserved(tmp_path):
    """Each pass must keep the entire previous stage fixed in the next pass."""
    # Scaled-down analogue of captured large payload geometry for quick deterministic tests.
    src_size = (208, 304)
    dst_size = (616, 768)
    source_box = {"x": 275, "y": 51, "w": src_size[0], "h": src_size[1]}
    source_image = _make_source_image(*src_size)

    settings = {
        "steps": 12,
        "smart_extend_feather": 12,
        "smart_extend_step_growth": 1.5,
        "smart_extend_refine": False,
        "smart_extend_pyramid_seam_strength": 0.0,
    }

    pyramid_outpaint.run_pyramid_outpaint(
        source_image=source_image,
        source_box=source_box,
        target_size=dst_size,
        prompt="regression prompt",
        negative_prompt="regression negative",
        base_inpaint=_FakeInpaint(),
        generator=_FakeGenerator(seed=777),
        settings=settings,
        debug_dir=tmp_path,
    )

    pass_events = _load_pass_events(tmp_path)
    assert len(pass_events) == 4
    assert pass_events[-1]["stage_size"] == [dst_size[0], dst_size[1]]

    for idx, event in enumerate(pass_events, start=1):
        assert event["lock_box"] == event["place_box"]
        assert "canvas_initializer" in event
        assert "seam_width" in event
        assert "seam_strength" in event
        assert "seam_box_count" in event
        assert int(event["seam_box_count"]) == 1
        assert event["canvas_initializer"] == "edge_strips"
        assert event["seam_mode"] == "final_only"
        assert event["seam_strength"] >= 0.0
        if event["seam_strength"] > 0.0:
            assert event["seam_strength"] >= 0.14
        if idx == 1:
            continue
        prev = Image.open(tmp_path / "pyramid" / f"pass_{idx - 1:02d}_13_stage_composited.png").convert("RGB")
        curr = Image.open(tmp_path / "pyramid" / f"pass_{idx:02d}_13_stage_composited.png").convert("RGB")
        box = event["place_box"]
        preserved = curr.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
        assert ImageChops.difference(preserved, prev).getbbox() is None


def test_pyramid_default_seam_is_final_only(tmp_path):
    """By default seam denoise should run only on the final pyramid pass."""
    src_size = (208, 304)
    dst_size = (616, 768)
    source_box = {"x": 275, "y": 51, "w": src_size[0], "h": src_size[1]}
    source_image = _make_source_image(*src_size)

    settings = {
        "steps": 12,
        "smart_extend_feather": 12,
        "smart_extend_step_growth": 1.5,
        "smart_extend_refine": False,
        "smart_extend_pyramid_seam_strength": 0.32,
    }

    pyramid_outpaint.run_pyramid_outpaint(
        source_image=source_image,
        source_box=source_box,
        target_size=dst_size,
        prompt="regression prompt",
        negative_prompt="regression negative",
        base_inpaint=_FakeInpaint(),
        generator=_FakeGenerator(seed=888),
        settings=settings,
        debug_dir=tmp_path,
    )

    pass_events = _load_pass_events(tmp_path)
    assert len(pass_events) == 4
    for idx, event in enumerate(pass_events, start=1):
        is_last = idx == len(pass_events)
        assert event["seam_mode"] == "final_only"
        assert bool(event["seam_applied"]) is is_last
        if is_last:
            assert float(event["seam_strength"]) > 0.0
            assert int(event["seam_steps"]) > 0
        else:
            assert float(event["seam_strength"]) == 0.0
            assert int(event["seam_steps"]) == 0
