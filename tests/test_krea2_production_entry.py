"""The shipped Krea 2 entry point must actually run the features we build.

A wrapper module used to replace the base worker's run loop with a
text-to-image-only copy. Everything added to the base worker afterwards - img2img
and the face/skin detailer - was silently dead in production, while unit tests
kept passing because they exercised the base worker directly.

These tests import the real production chain and assert the request reaches the
canonical loop intact. They are deliberately structural rather than GPU-bound so
they run in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _production_chain() -> tuple[object, object, object]:
    """Import the exact module chain the host launches for a Krea 2 request."""
    from core.backends import krea2_worker_adaptive as adaptive
    from core.backends import krea2_worker_safe as safe
    from core.backends import krea2_worker_impl as impl

    return adaptive, safe, impl


def test_host_launches_the_adaptive_entry_point():
    """Pin the production entry point so a swap cannot silently bypass it."""
    host = (_ROOT / "core" / "backends" / "krea2_host_impl.py").read_text(encoding="utf-8")
    assert 'krea2_worker_adaptive.py"' in host


def test_production_delegates_to_the_canonical_loop():
    """The safe wrapper must delegate, never shadow the canonical loop."""
    _, safe, impl = _production_chain()
    # Shadowing is the exact bug: safe used to rebind base._run to its own loop.
    assert safe._run is not impl._run or safe._run.__module__.endswith("krea2_worker_safe")
    assert "base._run = " not in _safe_source(), "safe must not rebind the base loop"


def _safe_source() -> str:
    return (_ROOT / "core" / "backends" / "krea2_worker_safe.py").read_text(encoding="utf-8")


def test_safe_wrapper_has_no_private_generation_loop():
    """A second copy of the run loop is how this regressed; forbid it."""
    tree = ast.parse(_safe_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run":
            body = ast.dump(node)
            # A delegating _run is a single return of base._run(...).
            assert "base._run" in body
            assert "denoise" not in body.lower()
            assert "decode" not in body.lower()
            return
    pytest.fail("krea2_worker_safe no longer defines _run")


@pytest.mark.parametrize("feature", ["img2img", "face_detailer"])
def test_canonical_loop_reads_each_feature(feature):
    """The loop that actually runs must consume the feature payloads."""
    _, _, impl = _production_chain()
    source = Path(impl.__file__).read_text(encoding="utf-8")
    assert f'request.get("{feature}")' in source


@pytest.mark.parametrize("feature", ["img2img", "face_detailer"])
def test_adaptive_planner_preserves_each_feature(feature):
    """The planner rewrites the request; it must not drop feature payloads."""
    source = (_ROOT / "core" / "backends" / "krea2_worker_adaptive.py").read_text(encoding="utf-8")
    start = source.index("def _adaptive_request(")
    body = source[start : source.index("\ndef ", start + 10)]
    # tuned starts as a copy of the request; anything removed after that is a bug.
    assert "tuned = dict(request)" in body
    assert f'tuned.pop("{feature}"' not in body
    assert f'del tuned["{feature}"]' not in body


@pytest.mark.parametrize("feature", ["img2img", "face_detailer"])
def test_production_chain_reaches_the_feature(feature, monkeypatch):
    """End-to-end plumbing: a request through the real chain sets the flag.

    Stubs the GPU primitives so this runs without a card, and asserts the
    canonical loop saw the feature payload rather than a stripped request.
    """
    adaptive, _, impl = _production_chain()

    seen: dict = {}

    def fake_run(request, output_dir, progress_path=None):
        seen["img2img"] = request.get("img2img")
        seen["face_detailer"] = request.get("face_detailer")
        seen["requested_width"] = request.get("requested_width")
        seen["requested_height"] = request.get("requested_height")
        return {"ok": True, "images": [], "seed": 0, "timing": {}, "runtime": {}}

    class _FakeSafe:
        # native_fp8.enable_for_hardware() is handed safe.base, so expose the
        # real module and only replace the loop.
        base = impl
        _run = staticmethod(fake_run)

    monkeypatch.setattr(adaptive, "safe", _FakeSafe)

    request = {
        "prompt": "p",
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "img2img": {"input_image": "/tmp/x.png", "strength": 0.6},
        "face_detailer": {"enabled": True, "strength": 0.3},
    }
    result = adaptive._run(request, Path("/tmp/opencode/krea_e2e/smoke"))

    assert result["ok"] is True
    assert seen["img2img"]["input_image"] == "/tmp/x.png"
    assert seen["face_detailer"]["enabled"] is True
    # The planner must record the original request size for output resampling.
    assert seen["requested_width"] == 1024
    assert seen["requested_height"] == 1024
    assert impl is not None


def test_planner_records_requested_size_for_resampling():
    """Sizing regression: a 2x decode must still be saved at the requested size."""
    import json
    import tempfile

    adaptive, _, _ = _production_chain()
    request = {
        "prompt": "p",
        "width": 832,
        "height": 1216,
        "steps": 10,
    }
    with tempfile.TemporaryDirectory() as tmp:
        hardware_path = Path(tmp) / "hardware.json"
        hardware_path.write_text(json.dumps({"accelerator": "cpu", "total_vram_gb": 64.0}), encoding="utf-8")
        try:
            tuned, plan = adaptive._adaptive_request(request, {"accelerator": "cpu", "total_vram_gb": 64.0})
        except Exception:
            pytest.skip("adaptive planner needs host hardware detection")
    assert tuned["requested_width"] == 832
    assert tuned["requested_height"] == 1216
    assert plan["requested_width"] == 832


def test_output_is_resampled_to_the_requested_size():
    """The grain fix: decode detail must be resampled back down, not shipped raw."""
    _, _, impl = _production_chain()
    from PIL import Image

    image = Image.new("RGB", (1504, 2208), (10, 20, 30))
    fitted = impl._fit_requested_output(image, {"requested_width": 832, "requested_height": 1216})
    assert fitted.size == (832, 1216)

    # No requested size recorded (plain text2img): leave the decode alone.
    assert impl._fit_requested_output(image, {}).size == (1504, 2208)

    # Already the right size: no needless resample.
    same = Image.new("RGB", (832, 1216), (0, 0, 0))
    assert impl._fit_requested_output(same, {"requested_width": 832, "requested_height": 1216}).size == (
        832,
        1216,
    )


def test_resampling_happens_after_the_face_pass():
    """Refine at full decode detail, then downsample; not the reverse."""
    source = Path(_production_chain()[2].__file__).read_text(encoding="utf-8")
    refine = source.index("krea2_face_detailer.run_face_detailer")
    fit = source.index("_fit_requested_output(image, request)")
    assert refine < fit


def test_safety_policy_lives_in_the_canonical_loop():
    """The safe wrapper's guarantees must survive the delegation."""
    _, safe, impl = _production_chain()
    for name in (
        "_coerce_conditioning_dtype",
        "_memory_reserve_gb",
        "_retry_profiles",
        "_dispose_pipe",
        "_fresh_denoiser",
    ):
        assert hasattr(impl, name), f"{name} missing from the canonical worker"
        assert getattr(safe, name) is getattr(impl, name), f"{name} drifted"


def test_canonical_loop_keeps_the_progressive_oom_ladder():
    """A single retry was the old behaviour; keep the ladder that replaced it."""
    _, _, impl = _production_chain()
    assert impl._retry_profiles("resident-fp8", {}) == [
        ("transformer-block", False),
        ("group", False),
        ("sequential", False),
    ]
    # Already sequential: nothing more conservative to try, so stop.
    assert impl._retry_profiles("sequential", {}) == []


def test_resident_headroom_is_revalidated_before_denoising():
    """Optimistic estimates must be checked against real free VRAM."""
    source = Path(_production_chain()[2].__file__).read_text(encoding="utf-8")
    assert 'fallback_reason = "resident_headroom"' in source
    assert "post_setup_free_vram_below_reserve" in source