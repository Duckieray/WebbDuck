"""Regression tests for thumbnail generation race safety."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from webbduck.server import thumbnails


def test_ensure_thumbnail_is_thread_safe(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (1600, 1200), (120, 80, 40)).save(source)

    monkeypatch.setattr(thumbnails, "resolve_web_path", lambda _web_path: source)

    def _run_once():
        return thumbnails.ensure_thumbnail("/outputs/source.png")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _run_once(), range(24)))

    assert results
    expected = thumbnails.get_thumbnail_path(source)
    assert all(Path(p) == expected for p in results)
    assert expected.exists()

    with Image.open(expected) as thumb:
        assert thumb.width <= thumbnails.THUMB_SIZE[0]
        assert thumb.height <= thumbnails.THUMB_SIZE[1]

    # No temp artifacts should remain after atomic write path.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
