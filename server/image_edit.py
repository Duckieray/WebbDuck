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
import threading
import time
from pathlib import Path

from PIL import Image, ImageFilter

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


def grow_mask(mask: Image.Image, pixels: int) -> Image.Image:
    """Dilate a white-on-black mask so object edges are fully covered."""
    try:
        grow = int(pixels)
    except (TypeError, ValueError):
        grow = 0
    if grow <= 0:
        return mask
    # MaxFilter needs an odd kernel and gets very expensive above ~100px.
    size = min(101, max(3, grow * 2 + 1))
    if size % 2 == 0:
        size += 1
    return mask.convert("L").filter(ImageFilter.MaxFilter(size))


def save_erased_images(images, source: Path, ops: dict | None = None) -> list[str]:
    """Persist generated erase results as new ``_edit`` copies beside ``source``.

    Erase behaves like a crop: the original is never touched and the result becomes a
    new gallery image in the same run folder, carrying ``edited_from`` lineage.
    """
    saved: list[str] = []
    for image in images:
        converted = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
        dest = allocate_edit_path(source)
        _atomic_save(converted, dest)

        erase_ops = {"erase": True}
        if ops:
            erase_ops.update({k: v for k, v in ops.items() if v not in (None, "")})
        meta = _edited_meta(source, erase_ops)
        meta["source"] = "image_erase"
        meta["width"], meta["height"] = converted.size
        _register_edited(dest, meta)
        saved.append(to_web_path(dest))
    return saved


EDIT_SESSION_KEY = "edit_sessions"
_EDIT_SESSION_LOCK = threading.Lock()
_MAX_MASK_DATA_URL = 24 * 1024 * 1024
_MAX_SESSION_TEXT = 2000


def _write_run_meta(run_dir: Path, meta: dict) -> None:
    """Atomically replace a run's ``meta.json`` so readers never see a partial file."""
    meta_file = run_dir / "meta.json"
    tmp = meta_file.with_name(f".{meta_file.name}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        tmp.replace(meta_file)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _finite_number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _clean_crop(value):
    if not isinstance(value, dict):
        return None
    x = _finite_number(value.get("x"))
    y = _finite_number(value.get("y"))
    w = _finite_number(value.get("w"))
    h = _finite_number(value.get("h"))
    if None in (x, y, w, h) or w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _clean_text(value, limit: int = _MAX_SESSION_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _clean_mask(value):
    if not value:
        return None
    if not isinstance(value, str):
        raise ImageEditError("Edit mask must be a PNG data URL")
    if not value.startswith("data:image/png;base64,"):
        raise ImageEditError("Edit mask must be a PNG data URL")
    if len(value) > _MAX_MASK_DATA_URL:
        raise ImageEditError("Edit mask is too large to store")
    return value


def _sanitize_edit_session(session: dict) -> dict:
    """Keep only the known, serializable editor fields and clamp their ranges."""
    if not isinstance(session, dict):
        raise ImageEditError("Edit session must be an object")

    clean = {
        "version": int(_finite_number(session.get("version"), 1) or 1),
        "tool": "erase" if session.get("tool") == "erase" else "crop",
        "rot": int(_finite_number(session.get("rot"), 0) or 0) % 4,
        "flipH": bool(session.get("flipH")),
        "flipV": bool(session.get("flipV")),
        "crop": _clean_crop(session.get("crop")),
        "aspect": _clean_text(session.get("aspect"), 40) or "free",
        "brightness": min(max(_finite_number(session.get("brightness"), 1.0), 0.0), 2.0),
        "contrast": min(max(_finite_number(session.get("contrast"), 1.0), 0.0), 2.0),
        "saturation": min(max(_finite_number(session.get("saturation"), 1.0), 0.0), 2.0),
        "targetW": None,
        "targetH": None,
        "lockAspect": bool(session.get("lockAspect", True)),
        "snapEight": bool(session.get("snapEight", True)),
        "brushSize": min(max(_finite_number(session.get("brushSize"), 48.0), 1.0), 400.0),
        "feather": min(max(_finite_number(session.get("feather"), 12.0), 0.0), 200.0),
        "model": _clean_text(session.get("model"), 500),
        "prompt": _clean_text(session.get("prompt")),
        "negativePrompt": _clean_text(session.get("negativePrompt")),
        "mask": _clean_mask(session.get("mask")),
        "updatedAt": _finite_number(session.get("updatedAt"), time.time()),
    }
    for key in ("targetW", "targetH"):
        number = _finite_number(session.get(key))
        if number is not None and number > 0:
            clean[key] = int(number)
    return clean


def load_edit_session(source: Path) -> dict | None:
    """Return the saved editor session for ``source``, or ``None``."""
    meta = _read_run_meta(source.parent)
    sessions = meta.get(EDIT_SESSION_KEY)
    if not isinstance(sessions, dict):
        return None
    session = sessions.get(source.name)
    return dict(session) if isinstance(session, dict) else None


def save_edit_session(source: Path, session: dict) -> dict:
    """Persist the editor session for ``source`` in the run's ``meta.json``.

    Sessions are stored under a per-image ``edit_sessions`` map so a run with several
    images can remember a different mask and tool state for each one. The map is
    stripped from manifest/gallery metadata to keep it out of search and payloads.
    """
    clean = _sanitize_edit_session(session)
    with _EDIT_SESSION_LOCK:
        meta = _read_run_meta(source.parent)
        sessions = meta.get(EDIT_SESSION_KEY)
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[source.name] = clean
        meta[EDIT_SESSION_KEY] = sessions
        _write_run_meta(source.parent, meta)
    return clean


def remove_edit_session(source: Path) -> None:
    """Drop a deleted image's saved editor session from its run's ``meta.json``."""
    with _EDIT_SESSION_LOCK:
        meta = _read_run_meta(source.parent)
        sessions = meta.get(EDIT_SESSION_KEY)
        if not isinstance(sessions, dict) or source.name not in sessions:
            return
        sessions.pop(source.name, None)
        if sessions:
            meta[EDIT_SESSION_KEY] = sessions
        else:
            meta.pop(EDIT_SESSION_KEY, None)
        _write_run_meta(source.parent, meta)


def web_path(path: Path) -> str:
    """Return the ``/outputs/...`` URL form used by the frontend."""
    return f"/{to_web_path(path)}"
