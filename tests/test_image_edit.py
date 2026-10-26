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

    def test_records_the_edited_size_not_the_source_size(self, monkeypatch, tmp_path, recorded_manifest):
        """Regression: the manifest inherited the source's width/height.

        The gallery builds each tile's aspect-ratio style from meta.width/meta.height,
        so a crop saved under the source's dimensions rendered visibly stretched even
        though the PNG itself was correct.
        """
        run = _redirect_output_root(monkeypatch, tmp_path)
        (run / "meta.json").write_text(
            json.dumps({"prompt": "a cat", "seed": 7, "width": 1024, "height": 1024}),
            encoding="utf-8",
        )
        source = _make_run_image(run)

        payload = io.BytesIO()
        Image.new("RGB", (800, 450)).save(payload, "PNG")
        dest = image_edit.store_edited_upload(source, payload.getvalue(), {"crop": [0, 0, 800, 450]})

        with Image.open(dest) as saved:
            assert saved.size == (800, 450)
        assert len(recorded_manifest) == 1
        _path, meta = recorded_manifest[0]
        assert meta["width"] == 800
        assert meta["height"] == 450
        assert (meta["width"], meta["height"]) == Image.open(dest).size
        # Inherited prompt/seed lineage is still preserved.
        assert meta["prompt"] == "a cat"
        assert meta["seed"] == 7

    def test_records_the_true_size_even_when_source_metadata_has_none(
        self, monkeypatch, tmp_path, recorded_manifest
    ):
        run = _redirect_output_root(monkeypatch, tmp_path)
        (run / "meta.json").write_text(json.dumps({"prompt": "no dims"}), encoding="utf-8")
        source = _make_run_image(run)

        payload = io.BytesIO()
        Image.new("RGB", (123, 45)).save(payload, "PNG")
        image_edit.store_edited_upload(source, payload.getvalue())

        _path, meta = recorded_manifest[0]
        assert (meta["width"], meta["height"]) == (123, 45)

    def test_does_not_mutate_the_runs_meta_file(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        original = json.dumps({"prompt": "a cat", "width": 1024, "height": 1024})
        meta_file = run / "meta.json"
        meta_file.write_text(original, encoding="utf-8")
        source = _make_run_image(run)

        payload = io.BytesIO()
        Image.new("RGB", (64, 32)).save(payload, "PNG")
        image_edit.store_edited_upload(source, payload.getvalue())

        assert meta_file.read_text(encoding="utf-8") == original

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


# ── manifest dimension repair ────────────────────────────────────────────────

@pytest.fixture
def redirected_manifest(monkeypatch, tmp_path):
    """Point ``storage`` at a throwaway manifest + output root, cache cleared."""
    from server import storage

    root = tmp_path / "outputs"
    run = root / "run_fix"
    run.mkdir(parents=True)
    monkeypatch.setattr(storage, "BASE", root)
    monkeypatch.setattr(storage, "MANIFEST_FILE", root / "manifest.jsonl")
    monkeypatch.setattr(storage, "_manifest_cache", None)
    monkeypatch.setattr(storage, "_manifest_cache_mtime", -1.0)
    return storage, root / "run_fix"


def _write_manifest(storage, rows):
    storage.MANIFEST_FILE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _manifest_rows(storage):
    return [json.loads(line) for line in storage.MANIFEST_FILE.read_text().splitlines() if line.strip()]


class TestRepairManifestImageDimensions:
    def test_repairs_an_edit_that_kept_the_source_dimensions(self, redirected_manifest):
        storage, run = redirected_manifest
        Image.new("RGB", (800, 450)).save(run / "0_edit.png")
        _write_manifest(storage, [{
            "image": "outputs/run_fix/0_edit.png",
            "meta": {"source": "image_edit", "width": 1024, "height": 1024, "prompt": "a duck"},
        }])

        report = storage.repair_manifest_image_dimensions(source_filter="image_edit")

        assert report["repaired"] == 1
        assert report["changes"] == [("outputs/run_fix/0_edit.png", (1024, 1024), (800, 450))]
        rows = _manifest_rows(storage)
        assert (rows[0]["meta"]["width"], rows[0]["meta"]["height"]) == (800, 450)
        # Sibling metadata must survive untouched.
        assert rows[0]["meta"]["prompt"] == "a duck"

    def test_leaves_already_correct_entries_alone(self, redirected_manifest):
        storage, run = redirected_manifest
        Image.new("RGB", (800, 450)).save(run / "0_edit.png")
        _write_manifest(storage, [{
            "image": "outputs/run_fix/0_edit.png",
            "meta": {"source": "image_edit", "width": 800, "height": 450},
        }])

        report = storage.repair_manifest_image_dimensions(source_filter="image_edit")

        assert report["repaired"] == 0
        assert report["checked"] == 1

    def test_source_filter_leaves_generated_images_untouched(self, redirected_manifest):
        storage, run = redirected_manifest
        Image.new("RGB", (300, 200)).save(run / "0.png")
        _write_manifest(storage, [{
            "image": "outputs/run_fix/0.png",
            "meta": {"source": None, "width": 4096, "height": 4096},
        }])

        report = storage.repair_manifest_image_dimensions(source_filter="image_edit")

        assert report["checked"] == 0
        assert report["repaired"] == 0
        assert _manifest_rows(storage)[0]["meta"]["width"] == 4096

    def test_missing_files_are_skipped_not_raised(self, redirected_manifest):
        storage, run = redirected_manifest
        _write_manifest(storage, [{
            "image": "outputs/run_fix/does_not_exist.png",
            "meta": {"source": "image_edit", "width": 1024, "height": 1024},
        }])

        report = storage.repair_manifest_image_dimensions(source_filter="image_edit")

        assert report["repaired"] == 0
        assert report["skipped"] == 1

    def test_repairs_a_mixed_manifest_in_one_pass(self, redirected_manifest):
        storage, run = redirected_manifest
        Image.new("RGB", (640, 480)).save(run / "0_edit.png")
        Image.new("RGB", (500, 500)).save(run / "0_edit2.png")
        _write_manifest(storage, [
            {"image": "outputs/run_fix/0_edit.png",
             "meta": {"source": "image_edit", "width": 1024, "height": 1024}},
            {"image": "outputs/run_fix/0_edit2.png",
             "meta": {"source": "image_edit", "width": 1024, "height": 1024}},
            {"image": "outputs/run_fix/missing.png",
             "meta": {"source": "image_edit", "width": 8, "height": 8}},
        ])

        report = storage.repair_manifest_image_dimensions(source_filter="image_edit")

        assert report["checked"] == 3
        assert report["repaired"] == 2
        assert report["skipped"] == 1
        rows = _manifest_rows(storage)
        assert [(r["meta"]["width"], r["meta"]["height"]) for r in rows] == [(640, 480), (500, 500), (8, 8)]


# ── per-image dimensions through the gallery API ─────────────────────────────

class TestGalleryPublishesPerImageDimensions:
    """Regression: a run shares one meta.json, so edits inherited the original size.

    ``/gallery`` returned only run-level ``meta``, and ``flattenSessions`` stamped it
    onto every image in the run, so an edit saved as an extra file in that run was laid
    out at the *generated* image's aspect ratio. The tile and the lightbox then
    rendered the edited pixels stretched. The recorded per-image size must be
    published alongside the run meta.
    """

    def _run_with_edit(self, monkeypatch, tmp_path):
        from server import app as appmod

        root = tmp_path / "outputs"
        run = root / "2026-01-01_00-00-00"
        run.mkdir(parents=True)
        Image.new("RGB", (896, 1344)).save(run / "0.png")
        Image.new("RGB", (888, 944)).save(run / "0_edit.png")
        (run / "meta.json").write_text(
            json.dumps({"prompt": "a duck", "timestamp": 1, "width": 896, "height": 1344}),
            encoding="utf-8",
        )
        monkeypatch.setattr(appmod, "BASE", root)
        from server import storage

        monkeypatch.setattr(storage, "BASE", root)
        monkeypatch.setattr(storage, "MANIFEST_FILE", root / "manifest.jsonl")
        monkeypatch.setattr(storage, "_manifest_cache", None)
        monkeypatch.setattr(storage, "_manifest_cache_mtime", -1.0)
        monkeypatch.setattr(storage, "_dims_cache", None)
        monkeypatch.setattr(storage, "_dims_cache_mtime", -1.0)
        storage.rebuild_manifest()
        return root, run

    def test_gallery_returns_per_image_dims(self, monkeypatch, tmp_path):
        from server import app as appmod

        self._run_with_edit(monkeypatch, tmp_path)
        sessions = appmod.gallery(start=0, limit=10)
        assert len(sessions) == 1
        session = sessions[0]
        assert session["dims"] == {"0.png": [896, 1344], "0_edit.png": [888, 944]}
        # The run meta is still the generated image's size, untouched.
        assert session["meta"]["width"] == 896
        assert session["meta"]["height"] == 1344

    def test_manifest_image_dims_maps_each_image_to_its_own_size(self, monkeypatch, tmp_path):
        from server import storage

        self._run_with_edit(monkeypatch, tmp_path)
        dims = storage.manifest_image_dims()
        assert dims["outputs/2026-01-01_00-00-00/0.png"] == (896, 1344)
        assert dims["outputs/2026-01-01_00-00-00/0_edit.png"] == (888, 944)


    def test_manifest_rebuild_keeps_each_images_real_size(self, monkeypatch, tmp_path):
        """Regression: rebuild_manifest() stamped the run meta onto every file.

        A rebuild overwrote an edit's recorded size with the original's, so the edit
        reverted to the wrong aspect ratio without any user action.
        """
        from server import storage

        self._run_with_edit(monkeypatch, tmp_path)
        storage.rebuild_manifest()
        dims = storage.manifest_image_dims()
        assert dims["outputs/2026-01-01_00-00-00/0.png"] == (896, 1344)
        assert dims["outputs/2026-01-01_00-00-00/0_edit.png"] == (888, 944)
