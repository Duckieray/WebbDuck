"""Repair gallery manifest ``width``/``height`` for edited images.

Earlier versions of the gallery editor recorded the *source* run's dimensions
for a new edit, even though the saved PNG was cropped/resized/rotated to
different pixel dimensions. The gallery derives each tile's ``aspect-ratio``
from those manifest fields, so such entries render visibly stretched.

This walks the manifest, re-reads the true size of each image file, and rewrites
only the ``width``/``height`` that disagree with the pixels on disk. Nothing
else in an entry is touched, and files that are missing or unreadable are
skipped rather than reported as errors.

Run from the repo root:

    python tools/repair_edited_image_dimensions.py --dry-run
    python tools/repair_edited_image_dimensions.py
    python tools/repair_edited_image_dimensions.py --all   # includes non-edits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import storage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute manifest image dimensions from the files on disk."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing the manifest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="check every manifest entry instead of only edited images",
    )
    args = parser.parse_args()

    if not storage.MANIFEST_FILE.exists():
        print(f"no manifest at {storage.MANIFEST_FILE}")
        return 1

    source_filter = None if args.all else "image_edit"

    if args.dry_run:
        entries = storage._load_manifest_entries()
        changes = []
        skipped = 0
        from PIL import Image

        for entry in entries:
            meta = entry.get("meta")
            if not isinstance(meta, dict):
                continue
            if source_filter is not None and meta.get("source") != source_filter:
                continue
            web_path = entry.get("image")
            if not web_path:
                continue
            try:
                with Image.open(storage.resolve_web_path(web_path)) as opened:
                    real = opened.size
            except Exception:
                skipped += 1
                continue
            current = (meta.get("width"), meta.get("height"))
            if current != real:
                changes.append((web_path, current, real))

        for web_path, current, real in changes:
            print(
                f"  {web_path}: {current[0]}x{current[1]} -> {real[0]}x{real[1]}"
            )
        print(f"would repair {len(changes)} entr{'y' if len(changes) == 1 else 'ies'}"
              f" ({skipped} skipped as unreadable) - dry run, nothing written")
        return 0

    report = storage.repair_manifest_image_dimensions(source_filter=source_filter)
    for web_path, current, real in report["changes"]:
        print(f"  {web_path}: {current[0]}x{current[1]} -> {real[0]}x{real[1]}")
    print(
        f"checked {report['checked']}, repaired {report['repaired']}, "
        f"skipped {report['skipped']} unreadable"
    )
    if report["repaired"]:
        print("reload the gallery (or restart WebbDuck) to see the corrected tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
