"""Tests for per-image editor session persistence in the run's ``meta.json``."""

import asyncio
import json

import pytest
from PIL import Image

from server import app as appmod
from server import image_edit, storage
from server.image_edit import ImageEditError


def _redirect_output_root(monkeypatch, tmp_path):
    root = tmp_path / "outputs"
    run = root / "2026-01-01_00-00-00"
    run.mkdir(parents=True)

    monkeypatch.setattr(image_edit, "BASE", root)

    def _resolve(path_str):
        text = str(path_str)
        rel = text.split("outputs/", 1)[1] if "outputs/" in text else text
        return root / rel.lstrip("/")

    monkeypatch.setattr(image_edit, "resolve_web_path", _resolve)
    return run


def _make_image(run, name="0.png"):
    path = run / name
    Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
    return path


def _session(**overrides):
    session = {
        "version": 1,
        "tool": "erase",
        "rot": 1,
        "flipH": True,
        "flipV": False,
        "crop": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.6},
        "aspect": "free",
        "brightness": 1.1,
        "contrast": 1.2,
        "saturation": 0.9,
        "targetW": 64,
        "targetH": 48,
        "lockAspect": False,
        "snapEight": True,
        "brushSize": 48,
        "feather": 12,
        "model": "sdxl-inpaint",
        "prompt": "empty background",
        "negativePrompt": "object",
        "mask": "data:image/png;base64,AAAA",
        "updatedAt": 123.0,
    }
    session.update(overrides)
    return session


def test_edit_session_round_trips_in_run_meta(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    source = _make_image(run)

    saved = image_edit.save_edit_session(source, _session())
    assert saved["tool"] == "erase"
    assert saved["mask"].startswith("data:image/png;base64,")

    assert image_edit.load_edit_session(source) == saved

    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["edit_sessions"]["0.png"]["model"] == "sdxl-inpaint"
    assert meta["edit_sessions"]["0.png"]["mask"] == saved["mask"]


def test_edit_session_is_per_image(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    first = _make_image(run, "0.png")
    second = _make_image(run, "1.png")

    image_edit.save_edit_session(first, _session(mask=None))
    image_edit.save_edit_session(second, _session(model="flux-inpaint"))

    assert image_edit.load_edit_session(first)["model"] == "sdxl-inpaint"
    assert image_edit.load_edit_session(first)["mask"] is None
    assert image_edit.load_edit_session(second)["model"] == "flux-inpaint"


def test_edit_session_rejects_non_png_mask(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    source = _make_image(run)

    with pytest.raises(ImageEditError):
        image_edit.save_edit_session(source, _session(mask="data:image/jpeg;base64,AAAA"))


def test_edit_session_sanitizes_fields(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    source = _make_image(run)

    saved = image_edit.save_edit_session(source, {
        **_session(),
        "brushSize": 9999,
        "feather": -5,
        "crop": {"x": "bad", "y": 0, "w": 1, "h": 1},
        "unknown": "drop me",
    })

    assert saved["brushSize"] == 400.0
    assert saved["feather"] == 0.0
    assert saved["crop"] is None
    assert "unknown" not in saved


def test_remove_edit_session_drops_the_entry(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    source = _make_image(run)

    image_edit.save_edit_session(source, _session())
    image_edit.remove_edit_session(source)

    assert image_edit.load_edit_session(source) is None
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert "edit_sessions" not in meta


def test_sanitize_settings_drops_edit_sessions():
    clean = storage.sanitize_settings({"prompt": "x", "edit_sessions": {"0.png": {}}})
    assert "edit_sessions" not in clean
    assert clean["prompt"] == "x"


def test_edit_session_endpoints_round_trip(monkeypatch, tmp_path):
    run = _redirect_output_root(monkeypatch, tmp_path)
    _make_image(run)
    web_path = "outputs/2026-01-01_00-00-00/0.png"

    request = appmod.EditSessionRequest(path=web_path, session=_session())
    saved = asyncio.run(appmod.save_image_edit_session(request))
    assert saved["status"] == "ok"

    loaded = asyncio.run(appmod.get_image_edit_session(web_path))
    assert loaded["status"] == "ok"
    assert loaded["session"]["prompt"] == "empty background"


def test_edit_session_endpoint_rejects_outside_paths(monkeypatch, tmp_path):
    _redirect_output_root(monkeypatch, tmp_path)

    request = appmod.EditSessionRequest(path="../etc/passwd", session=_session())
    response = asyncio.run(appmod.save_image_edit_session(request))
    assert response.status_code == 400
