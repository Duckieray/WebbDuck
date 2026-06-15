"""Thumbnail generation logic."""

import os
import tempfile
import threading
from pathlib import Path
from PIL import Image

from server.storage import BASE, resolve_web_path

THUMB_SUFFIX = ".thumb.jpg"
THUMB_SIZE = (512, 512)
THUMB_QUALITY = 85
_thumb_locks: dict[str, threading.Lock] = {}
_thumb_locks_guard = threading.Lock()


def _lock_for_path(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _thumb_locks_guard:
        lock = _thumb_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _thumb_locks[key] = lock
        return lock

def get_thumbnail_path(original_path: Path) -> Path:
    """Get expected thumbnail path for an image."""
    return original_path.with_suffix(THUMB_SUFFIX)

def ensure_thumbnail(web_path: str) -> Path:
    """
    Ensure a thumbnail exists for the given web path.
    Returns the filesystem path to the thumbnail.
    """
    original_path = resolve_web_path(web_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Original image not found: {original_path}")
        
    thumb_path = get_thumbnail_path(original_path)
    lock = _lock_for_path(thumb_path)
    with lock:
        # Output filenames are immutable per run, so avoid rewrite churn/races:
        # once generated, keep and serve the existing thumbnail.
        if thumb_path.exists():
            return thumb_path

        # Generate thumbnail atomically via temp file + replace.
        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{thumb_path.stem}.",
                suffix=".tmp",
                dir=str(thumb_path.parent),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)

            with Image.open(original_path) as img:
                # Convert to RGB (in case of RGBA/P) before saving as JPG
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                img.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
                img.save(tmp_path, "JPEG", quality=THUMB_QUALITY)

            os.replace(tmp_path, thumb_path)
            return thumb_path
        except Exception as e:
            print(f"Error generating thumbnail for {original_path}: {e}")
            # Never fall back to the full-size original for /thumbs requests.
            # Returning large originals during thumbnail bursts can amplify memory pressure.
            if thumb_path.exists():
                return thumb_path
            raise
        finally:
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
