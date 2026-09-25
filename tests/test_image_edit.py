"""Tests for non-destructive gallery image editing.

Covers path containment, edit-name allocation, Pillow transform application, and the
two edit endpoints. Output storage is redirected into ``tmp_path`` so no real gallery is
touched.
"""

import asyncio
import io
import json
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.datastructures import UploadFile

from server import app as appmod
from server import image_edit


# ── fixtures / helpers ──────────────────────────────────────────────────────

def _redirect_output_root(monkeypatch, tmp_path) -> Path:
    """Point ``image_edit`` at a throwaway output root.

    ``image_edit`` imports ``BASE`` and ``resolve_web_path`` by value, so both names must
    be rebound for the containment check to see the temporary tree.
    """
    root = tmp_path / "outputs"
    run = root / "2026-01-01_00-00-00"
    run.mkdir(parents=True)

    monkeypatch.setattr(image_edit, "BASE", root)

    def _resolve(path_str: str) -> Path:
        text = str(path_str)
        rel = text.split("outputs/", 1)[1] if "outputs/" in text else text
        return root / rel.lstrip("/")

    monkeypatch.setattr(image_edit, "resolve_web_path", _resolve)
    return run


@pytest.fixture
def recorded_manifest(monkeypatch):
    """Capture manifest registrations instead of writing the real manifest file."""
    calls: list[tuple[Path, dict]] = []

    def _record(paths, run_dir, meta):
        for path in paths:
            calls.append((Path(path), meta))

    monkeypatch.setattr(image_edit, "append_manifest_entries", _record)
    return calls


@pytest.fixture
def silent_notifications(monkeypatch):
    monkeypatch.setattr(appmod, "_notify_gallery_changed", lambda *a, **k: None)


def _make_run_image(run_dir: Path, name: str = "0.png", size=(64, 32), color=(200, 40, 40)) -> Path:
    path = run_dir / name
    Image.new("RGB", size, color).save(path)
    return path


# ── path containment ────────────────────────────────────────────────────────

class TestResolveOutputImage:
    def test_resolves_image_inside_output_root(self, monkeypatch, tmp_path):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)

        assert image_edit.resolve_output_image(f"/outputs/{run.name}/0.png") == source.resolve()

    @pytest.mark.parametrize(
        "candidate",
        [
            "/outputs/../../etc/passwd",
            "/outputs/../secrets.png",
            "outputs/../../outside.png",
            "/outputs/",
        ],
    )
    def test_rejects_paths_outside_output_root(self, monkeypatch, tmp_path, candidate):
        run = _redirect_output_root(monkeypatch, tmp_path)
        (tmp_path / "outside.png").write_bytes(b"not an image")

        with pytest.raises(image_edit.ImageEditError):
            image_edit.resolve_output_image(candidate)

    def test_rejects_empty_path(self, monkeypatch, tmp_path):
        _redirect_output_root(monkeypatch, tmp_path)
        with pytest.raises(image_edit.ImageEditError):
            image_edit.resolve_output_image("   ")

    def test_rejects_directory(self, monkeypatch, tmp_path):
        run = _redirect_output_root(monkeypatch, tmp_path)
        with pytest.raises(image_edit.ImageEditError):
            image_edit.resolve_output_image(f"/outputs/{run.name}")


# ── edit name allocation ────────────────────────────────────────────────────

class TestAllocateEditPath:
    def test_first_edit_has_no_suffix(self, tmp_path):
        source = tmp_path / "0.png"
        source.write_bytes(b"x")
        assert image_edit.allocate_edit_path(source).name == "0_edit.png"

    def test_repeated_edits_do_not_collide(self, tmp_path):
        source = tmp_path / "0.png"
        source.write_bytes(b"x")

        first = image_edit.allocate_edit_path(source)
        first.write_bytes(b"x")
        second = image_edit.allocate_edit_path(source)
        second.write_bytes(b"x")
        third = image_edit.allocate_edit_path(source)

        assert [first.name, second.name, third.name] == ["0_edit.png", "0_edit_1.png", "0_edit_2.png"]
        assert source.read_bytes() == b"x"

    def test_editing_an_edit_does_not_double_the_marker(self, tmp_path):
        source = tmp_path / "3_edit.png"
        source.write_bytes(b"x")
        assert image_edit.allocate_edit_path(source).name == "3_edit_1.png"


# ── op normalization ────────────────────────────────────────────────────────

class TestStoreEditedUpload:
    def test_writes_new_image_and_keeps_original(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        original_bytes = source.read_bytes()

        payload = io.BytesIO()
        Image.new("RGB", (32, 16), (10, 20, 30)).save(payload, "PNG")
        dest = image_edit.store_edited_upload(source, payload.getvalue())

        assert dest != source
        assert dest.name == "0_edit.png"
        assert source.read_bytes() == original_bytes
        with Image.open(dest) as saved:
            assert saved.size == (32, 16)

    def test_registers_manifest_entry_with_lineage(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        (run / "meta.json").write_text(json.dumps({"prompt": "a cat", "seed": 7}), encoding="utf-8")
        source = _make_run_image(run)

        payload = io.BytesIO()
        Image.new("RGB", (8, 8)).save(payload, "PNG")
        dest = image_edit.store_edited_upload(source, payload.getvalue(), {"rotate": 1})

        assert len(recorded_manifest) == 1
        path, meta = recorded_manifest[0]
        assert path == dest
        assert meta["edited_from"].endswith("0.png")
        assert meta["source"] == "image_edit"
        assert meta["edit_ops"] == {"rotate": 1}
        # The run's own metadata is copied, not mutated on disk.
        assert meta["prompt"] == "a cat"

    def test_rejects_unreadable_payload(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)

        with pytest.raises(image_edit.ImageEditError):
            image_edit.store_edited_upload(source, b"definitely not an image")

    def test_rejects_empty_payload(self, monkeypatch, tmp_path):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        with pytest.raises(image_edit.ImageEditError):
            image_edit.store_edited_upload(source, b"")

    def test_no_temp_files_left_behind(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        payload = io.BytesIO()
        Image.new("RGB", (8, 8)).save(payload, "PNG")

        image_edit.store_edited_upload(source, payload.getvalue())
        assert list(run.glob("*.tmp")) == []


def _upload(data: bytes, filename: str = "edit.png") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def _png_bytes(size=(24, 12), color=(5, 5, 5)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


class TestSaveImageEditEndpoint:
    async def test_saves_edit_and_returns_new_path(self, monkeypatch, tmp_path,
                                                   recorded_manifest, silent_notifications):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        monkeypatch.setattr(appmod, "thumb_semaphore", asyncio.Semaphore(2))

        response = await appmod.save_image_edit(
            path=f"/outputs/{run.name}/0.png",
            image=_upload(_png_bytes((24, 12))),
            ops=json.dumps({"rotate": 1}),
        )

        assert response["status"] == "ok"
        assert response["path"].endswith("/0_edit.png")
        assert Path(run / "0_edit.png").exists()
        assert source.exists()

    async def test_rejects_traversal_path(self, monkeypatch, tmp_path,
                                          recorded_manifest, silent_notifications):
        _redirect_output_root(monkeypatch, tmp_path)
        (tmp_path / "outside.png").write_bytes(_png_bytes())
        monkeypatch.setattr(appmod, "thumb_semaphore", asyncio.Semaphore(2))

        response = await appmod.save_image_edit(
            path="/outputs/../../outside.png",
            image=_upload(_png_bytes()),
            ops="",
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert (tmp_path / "outside.png").exists()

    async def test_rejects_empty_payload(self, monkeypatch, tmp_path,
                                         recorded_manifest, silent_notifications):
        run = _redirect_output_root(monkeypatch, tmp_path)
        _make_run_image(run)
        monkeypatch.setattr(appmod, "thumb_semaphore", asyncio.Semaphore(2))

        response = await appmod.save_image_edit(
            path=f"/outputs/{run.name}/0.png",
            image=_upload(b""),
            ops="",
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400


class TestGalleryNotification:
    """The UI relies on the ``gallery`` event to reload after an edit."""

    def test_broadcasts_gallery_frame_with_reason_and_images(self, monkeypatch):
        sent: list[dict] = []

        async def _fake_broadcast(message):
            sent.append(message)

        monkeypatch.setattr(appmod, "broadcast", _fake_broadcast)

        async def _drive():
            appmod._notify_gallery_changed("edit", ["/outputs/run/0_edit.png"])
            await asyncio.sleep(0)  # let the scheduled task run

        asyncio.run(_drive())

        assert sent == [{
            "type": "gallery",
            "payload": {"reason": "edit", "images": ["/outputs/run/0_edit.png"]},
        }]

    def test_is_a_noop_without_a_running_event_loop(self, monkeypatch):
        """A sync worker thread has no loop; that must not raise."""
        monkeypatch.setattr(appmod, "broadcast", None)  # would explode if awaited
        appmod._notify_gallery_changed("edit", ["/outputs/run/0_edit.png"])

    async def test_single_edit_notifies_with_edit_reason(self, monkeypatch, tmp_path,
                                                          recorded_manifest, silent_notifications):
        run = _redirect_output_root(monkeypatch, tmp_path)
        _make_run_image(run, "0.png")
        monkeypatch.setattr(appmod, "thumb_semaphore", asyncio.Semaphore(2))

        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(
            appmod, "_notify_gallery_changed",
            lambda reason, images: calls.append((reason, list(images))),
        )

        result = await appmod.save_image_edit(
            path=f"/outputs/{run.name}/0.png",
            ops={"rotate": 1},
            image=_upload(_png_bytes((32, 16))),
        )

        assert result["status"] == "ok"
        assert calls, "editing must announce itself on the gallery event"
        reason, images = calls[0]
        assert reason == "edit"
        assert images == [result["path"]]
