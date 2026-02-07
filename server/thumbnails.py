"""Thumbnail generation logic."""

from pathlib import Path
from PIL import Image
import os

from webbduck.server.storage import BASE, resolve_web_path

THUMB_SUFFIX = ".thumb.jpg"
THUMB_SIZE = (512, 512)
THUMB_QUALITY = 85

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
    
    # If thumb exists and is newer than original, return it
    if thumb_path.exists():
        if thumb_path.stat().st_mtime > original_path.stat().st_mtime:
            return thumb_path

    # Generate thumbnail
    try:
        with Image.open(original_path) as img:
            # Convert to RGB (in case of RGBA/P) before saving as JPG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
            img.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG", quality=THUMB_QUALITY)
            
        return thumb_path
    except Exception as e:
        print(f"Error generating thumbnail for {original_path}: {e}")
        # Fallback to original if generation fails
        return original_path
