"""Image and session storage."""

import json
import time
from pathlib import Path

import os

# Check for custom output directory from environment variable
custom_out = os.environ.get("WEBBDUCK_OUTPUT_DIR")
if custom_out:
    BASE = Path(custom_out)
else:
    BASE = Path("outputs")



BASE.mkdir(exist_ok=True, parents=True)


def to_web_path(path: Path) -> str:
    """Convert valid filesystem path to web-accessible path."""
    try:
        rel = path.resolve().relative_to(BASE.resolve())
        return f"outputs/{rel.as_posix()}"
    except ValueError:
        return f"outputs/{path.name}"


def resolve_web_path(path_str: str) -> Path:
    """Convert web-accessible path to filesystem path."""
    if path_str.startswith("outputs/"):
        rel = path_str[len("outputs/"):]
        return BASE / rel
    return Path(path_str)




def save_images(images, settings):
    """Save generated images and metadata."""
    run = BASE / time.strftime("%Y-%m-%d_%H-%M-%S")
    run.mkdir()

    paths = []
    for i, img in enumerate(images):
        p = run / f"{i}.png"
        img.save(p)
        paths.append(to_web_path(p))

    with open(run / "meta.json", "w") as f:
        # Sanitize settings for JSON (remove PIL Image objects)
        clean_settings = settings.copy()
        if "input_image" in clean_settings:
            del clean_settings["input_image"]
        if "mask_image" in clean_settings:
            del clean_settings["mask_image"]
        json.dump(clean_settings, f, indent=2)


    return paths



SESSION_LOG = BASE / "session_log.json"


def append_session_entry(entry: dict):
    """Append entry to session log."""
    SESSION_LOG.parent.mkdir(exist_ok=True)

    if SESSION_LOG.exists():
        data = json.loads(SESSION_LOG.read_text())
    else:
        data = []

    data.append(entry)
    SESSION_LOG.write_text(json.dumps(data, indent=2))