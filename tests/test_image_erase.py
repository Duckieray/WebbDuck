"""Tests for diffusion erase copies created from the gallery Edit overlay.

Erase is a GPU job, so these tests exercise the CPU-side contract: mask growth, the
new-copy storage/lineage, and the endpoint's queue wiring. The actual denoising is
covered by the existing inpaint mode tests.
"""

import asyncio
import io
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.datastructures import UploadFile

from server import app as appmod
from server import image_edit


def _redirect_output_root(monkeypatch, tmp_path) -> Path:
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
    calls: list[tuple[Path, dict]] = []

    def _record(paths, run_dir, meta):
        for path in paths:
            calls.append((Path(path), dict(meta)))

    monkeypatch.setattr(image_edit, "append_manifest_entries", _record)
    return calls


@pytest.fixture
def silent_notifications(monkeypatch):
    monkeypatch.setattr(appmod, "_notify_gallery_changed", lambda *a, **k: None)


def _make_run_image(run_dir: Path, name: str = "0.png", size=(64, 48)) -> Path:
    path = run_dir / name
    Image.new("RGB", size, (200, 40, 40)).save(path)
    return path


def _png_bytes(size=(32, 24), color=(10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def _mask_bytes(size=(32, 24), box=(8, 8, 24, 16)) -> bytes:
    buffer = io.BytesIO()
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    mask.save(buffer, "PNG")
    return buffer.getvalue()


def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def _run_erase(path, image, mask, **overrides):
    """Call the endpoint directly with concrete values (FastAPI Form defaults are
    unresolved when the coroutine is invoked without the request layer)."""
    params = {
        "path": path,
        "image": image,
        "mask": mask,
        "base_model": "sdxl-test",
        "prompt": "",
        "negative_prompt": "",
        "strength": 0.99,
        "mask_blur": 12,
        "mask_grow": 6,
        "steps": None,
        "cfg": None,
        "seed": None,
        "wait_for_result": True,
    }
    params.update(overrides)
    return asyncio.run(appmod.erase_image(**params))


# ── mask growth ─────────────────────────────────────────────────────────────

class TestGrowMask:
    def test_dilates_the_painted_region(self):
        mask = Image.new("L", (32, 32), 0)
        mask.putpixel((16, 16), 255)

        grown = image_edit.grow_mask(mask, 3)

        assert sum(1 for value in grown.getdata() if value > 0) > 1
        assert grown.getpixel((16, 16)) == 255

    def test_zero_growth_is_a_noop(self):
        mask = Image.new("L", (8, 8), 0)
        mask.putpixel((4, 4), 255)
        assert image_edit.grow_mask(mask, 0) is mask


# ── storage ─────────────────────────────────────────────────────────────────

class TestSaveErasedImages:
    def test_writes_new_copy_and_keeps_original(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        original = source.read_bytes()

        paths = image_edit.save_erased_images(
            [Image.new("RGB", (32, 16), (1, 2, 3))], source, {"prompt": "background"}
        )

        assert len(paths) == 1
        assert paths[0].endswith("0_edit.png")
        assert (run / "0_edit.png").exists()
        assert source.read_bytes() == original

    def test_records_erase_lineage(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        (run / "meta.json").write_text('{"prompt": "a cat", "seed": 7}', encoding="utf-8")
        source = _make_run_image(run)

        image_edit.save_erased_images(
            [Image.new("RGB", (20, 10))], source, {"prompt": "empty background", "mask_grow": 6}
        )

        assert len(recorded_manifest) == 1
        _path, meta = recorded_manifest[0]
        assert meta["edited_from"].endswith("0.png")
        assert meta["source"] == "image_erase"
        assert meta["edit_ops"]["erase"] is True
        assert meta["edit_ops"]["prompt"] == "empty background"
        assert (meta["width"], meta["height"]) == (20, 10)

    def test_repeated_erases_do_not_collide(self, monkeypatch, tmp_path, recorded_manifest):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)

        first = image_edit.save_erased_images([Image.new("RGB", (8, 8))], source)
        second = image_edit.save_erased_images([Image.new("RGB", (8, 8))], source)

        assert first[0].endswith("0_edit.png")
        assert second[0].endswith("0_edit_1.png")


# ── endpoint ────────────────────────────────────────────────────────────────

class TestEraseEndpoint:
    def test_enqueues_at_front_and_returns_new_copy(
        self, monkeypatch, tmp_path, recorded_manifest, silent_notifications
    ):
        run = _redirect_output_root(monkeypatch, tmp_path)
        source = _make_run_image(run)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        monkeypatch.setattr(appmod, "INPUTS_DIR", inputs)

        calls: list[dict] = []

        async def _fake_enqueue(job, wait_for_result=True, front=False):
            calls.append({"job": job, "wait": wait_for_result, "front": front})
            paths = image_edit.save_erased_images(
                [Image.new("RGB", (32, 24), (1, 2, 3))],
                Path(job["source_image"]),
                job.get("erase_ops"),
            )
            return {"images": paths, "seed": 1}

        monkeypatch.setattr(appmod, "enqueue", _fake_enqueue)

        response = _run_erase(
            f"/outputs/{run.name}/0.png",
            _upload(_png_bytes(), "base.png"),
            _upload(_mask_bytes(), "mask.png"),
        )

        assert response["status"] == "ok"
        assert response["path"].endswith("0_edit.png")
        assert (run / "0_edit.png").exists()
        assert source.exists()

        assert len(calls) == 1
        assert calls[0]["front"] is True
        job = calls[0]["job"]
        assert job["type"] == "erase"
        assert job["settings"]["inpainting_fill"] == "erase"
        assert job["settings"]["base_model"] == "sdxl-test"
        assert job["settings"]["strength"] >= 0.95
        # The default background prompt is used when the caller sends none.
        assert "background" in job["settings"]["prompt"].lower()

    def test_grows_the_mask_before_queueing(
        self, monkeypatch, tmp_path, recorded_manifest, silent_notifications
    ):
        run = _redirect_output_root(monkeypatch, tmp_path)
        _make_run_image(run)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        monkeypatch.setattr(appmod, "INPUTS_DIR", inputs)

        captured = {}

        async def _fake_enqueue(job, wait_for_result=True, front=False):
            captured["mask_path"] = job["settings"]["mask_image"]
            return {"images": [], "seed": 1}

        monkeypatch.setattr(appmod, "enqueue", _fake_enqueue)

        tiny_mask = Image.new("L", (32, 24), 0)
        tiny_mask.putpixel((16, 12), 255)
        buffer = io.BytesIO()
        tiny_mask.save(buffer, "PNG")

        _run_erase(
            f"/outputs/{run.name}/0.png",
            _upload(_png_bytes(), "base.png"),
            _upload(buffer.getvalue(), "mask.png"),
            mask_grow=4,
        )

        with Image.open(captured["mask_path"]) as saved_mask:
            assert sum(1 for value in saved_mask.getdata() if value > 0) > 1

    def test_rejects_traversal_path(self, monkeypatch, tmp_path, silent_notifications):
        _redirect_output_root(monkeypatch, tmp_path)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        monkeypatch.setattr(appmod, "INPUTS_DIR", inputs)
        (tmp_path / "outside.png").write_bytes(_png_bytes())

        response = _run_erase(
            "/outputs/../../outside.png",
            _upload(_png_bytes(), "base.png"),
            _upload(_mask_bytes(), "mask.png"),
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

    def test_rejects_empty_payloads(self, monkeypatch, tmp_path, silent_notifications):
        run = _redirect_output_root(monkeypatch, tmp_path)
        _make_run_image(run)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        monkeypatch.setattr(appmod, "INPUTS_DIR", inputs)

        response = _run_erase(
            f"/outputs/{run.name}/0.png",
            _upload(b"", "base.png"),
            _upload(_mask_bytes(), "mask.png"),
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
