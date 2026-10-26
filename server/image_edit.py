"""Non-destructive image editing primitives for gallery outputs.

This module is intentionally CPU-only: it never touches torch/CUDA and never takes the
GPU lease, so editing stays available while a generation job is queued or running.

The pixel work happens in the browser (``ui/modules/ImageEditor.js`` renders the crop,
rotate, flip, resize, and tonal adjustments onto a canvas). :func:`store_edited_upload`
persists that rendered PNG as a new file next to the source image and records the lineage
in the manifest.

Only new files are ever written; existing outputs are never overwritten.
"""

import io
import json
import time
from pathlib import Path

from PIL import Image

from server.storage import (
    BASE,
    append_manifest_entries,
    resolve_web_path,
    to_web_path,
)

EDIT_STEM_SUFFIX = "_edit"


class ImageEditError(ValueError):
    """Raised when an edit request cannot be satisfied."""


def resolve_output_image(path_str: str) -> Path:
    """Resolve a web path to a real image inside the output root.

    ``resolve_web_path`` is a plain string split with no containment guarantee, so every
    caller that mutates or reads gallery data through it must re-validate the result.
    """
    raw = (path_str or "").strip()
    if not raw:
        raise ImageEditError("Image path is required")

    candidate = resolve_web_path(raw)
    try:
        resolved = candidate.resolve()
        root = BASE.resolve()
    except OSError as exc:
        raise ImageEditError(f"Could not resolve image path: {exc}") from exc

    if resolved == root or not resolved.is_relative_to(root):
        raise ImageEditError("Image path is outside the output directory")
    if not resolved.is_file():
        raise ImageEditError("Image not found")

    return resolved


def allocate_edit_path(source: Path, attempt: int = 0) -> Path:
    """Return a free ``<stem>_edit[_N].png`` path next to ``source``.

    Repeated edits of the same image must not clobber each other, so a numeric suffix is
    appended until a free name is found. The limit guards against unbounded probing.
    """
    stem = source.stem
    if stem.endswith(EDIT_STEM_SUFFIX):
        # 0_edit -> 0_edit_edit reads badly; drop the existing marker first.
        stem = stem[: -len(EDIT_STEM_SUFFIX)]
    base = f"{stem}{EDIT_STEM_SUFFIX}"

    for index in range(attempt, attempt + 1000):
        name = f"{base}.png" if index == 0 else f"{base}_{index}.png"
        candidate = source.with_name(name)
        if not candidate.exists():
            return candidate
    raise ImageEditError("Could not allocate an output name for the edited image")


def _read_run_meta(run_dir: Path) -> dict:
    meta_file = run_dir / "meta.json"
    if not meta_file.is_file():
        return {}
    try:
        with meta_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _edited_meta(source: Path, ops: dict | None = None) -> dict:
    """Build manifest metadata for a derived image.

    The run's ``meta.json`` is shared by every image in the run, so it is copied rather
    than mutated and annotated with the edit lineage.
    """
    meta = dict(_read_run_meta(source.parent))
    meta["edited_from"] = to_web_path(source)
    meta["edited_at"] = time.time()
    meta["source"] = "image_edit"
    if ops:
        applied = {k: v for k, v in ops.items() if v not in (0, 0.0, False, "")}
        if applied:
            meta["edit_ops"] = applied
    return meta


def _register_edited(dest: Path, meta: dict):
    """Publish a newly written derived image to the manifest index."""
    try:
        append_manifest_entries([dest], dest.parent, meta)
    except Exception as exc:  # pragma: no cover - manifest writes are best effort
        print(f"[Warn] Could not index edited image {dest}: {exc}")


def _atomic_save(image: Image.Image, dest: Path):
    """Write a PNG through a temp file in the destination directory, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{time.time_ns()}.tmp")
    try:
        image.save(tmp, "PNG")
        tmp.replace(dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def store_edited_upload(source: Path, data: bytes, ops: dict | None = None) -> Path:
    """Persist a client-rendered PNG blob as a new image next to ``source``."""
    if not data:
        raise ImageEditError("Edited image payload is empty")

    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            if opened.mode in ("RGBA", "P", "LA"):
                converted = opened.convert("RGBA")
            else:
                converted = opened.convert("RGB")
    except Exception as exc:
        raise ImageEditError(f"Uploaded edit is not a readable image: {exc}") from exc

    dest = allocate_edit_path(source)
    _atomic_save(converted, dest)

    meta = _edited_meta(source, ops)
    # The edit changes the pixel dimensions, so the width/height inherited from the
    # run's meta.json no longer describe this file. The gallery builds each tile's
    # aspect ratio from these values, so leaving the source dimensions in place
    # renders the edited PNG stretched. Record the real size, the same way the
    # input-upscale path does.
    meta["width"], meta["height"] = converted.size
    _register_edited(dest, meta)
    return dest


def web_path(path: Path) -> str:
    """Return the ``/outputs/...`` URL form used by the frontend."""
    return f"/{to_web_path(path)}"
