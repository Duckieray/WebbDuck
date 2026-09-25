"""Backfill canonical display ``title`` for LoRAs and embeddings.

Derives titles from:
  1. CivitAI model pages (via the ``url`` already stored in the metadata
     store) — the authoritative canonical name for most assets.
  2. Safetensors metadata headers (``modelspec.title``, then trainer ``name``)
     as an offline fallback when no usable URL exists or the fetch fails.

Identity is never changed: each asset keeps its file-stem ``name`` as the
generation/persistence key; ``title`` is purely the human-facing display name.

Run from the repo root:

    python tools/backfill_canonical_titles.py --dry-run
    python tools/backfill_canonical_titles.py            # fill online + offline
    python tools/backfill_canonical_titles.py --offline-only
    python tools/backfill_canonical_titles.py --force    # overwrite existing titles
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Mirrors run.py's path bootstrap so the script works from any CWD.
_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT.parent):
    _as_str = str(_path)
    if _as_str not in sys.path:
        sys.path.insert(0, _as_str)

import requests

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - used only for the offline path
    safe_open = None

from core.provider_credentials import resolve_provider_token
from models.metastore import meta_store
from models.registry import EMBEDDING_REGISTRY, LORA_REGISTRY

API_BASE = "https://civitai.red/api/v1/models/{model_id}"
_REQUEST_DELAY = 0.7  # seconds between CivitAI requests
REQUEST_TIMEOUT = 20

# Titles like ``epoch_1`` are trainer checkpoint epochs, not the LoRA name.
_TRAINER_PLACEHOLDER = re.compile(r"^epoch[_-]?\d+$", re.IGNORECASE)
# Some diffusers trainers write a timestamped placeholder title.
_TIMESTAMP_PLACEHOLDER = re.compile(r"^LoRA@\d+")
# Trainer metadata often leaks the *base model* name instead of the LoRA name.
_JUNK_TITLES = {
    "stable diffusion xl 1.0 base lora",
    "flux 2.0 lora",
    "fluxlora",
    "sdxl  v3",
    "sdxl v3",
    "lora",
}
_MAX_TITLE = 120


# ---------------------------------------------------------------------------
#  CivitAI lookups
# ---------------------------------------------------------------------------


def _civitai_model_id(url: str) -> str | None:
    match = re.search(r"(?:civitai\.com|civitai\.red)/models/(\d+)", url)
    return match.group(1) if match else None


def fetch_civitai_title(model_id: str, token: str | None = None) -> str | None:
    """Return the model page name for a CivitAI model id."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = API_BASE.format(model_id=model_id)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(1.0 + attempt)
            continue
        if resp.status_code == 429:
            time.sleep(2.0 + attempt * 2)
            continue
        if resp.status_code != 200:
            return None
        data = resp.json()
        name = (data or {}).get("name") or (data or {}).get("error")
        return _clean_title(name if isinstance(name, str) else None)
    return None


# ---------------------------------------------------------------------------
#  Safetensors metadata fallback
# ---------------------------------------------------------------------------


def _safetensors_title(path: str) -> str | None:
    if not safe_open or not path:
        return None
    try:
        with safe_open(path, framework="np") as f:
            md = f.metadata() or {}
    except Exception:
        return None
    for key in ("modelspec.title", "name"):
        value = _clean_title(md.get(key))
        if value:
            return value
    return None


def _clean_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    title = value.strip().strip(".,-_ ")
    if not title:
        return None
    if len(title) > _MAX_TITLE:
        title = title[:_MAX_TITLE].rstrip()
    low = title.lower()
    if low in _JUNK_TITLES:
        return None
    if low in ("epoch", "none", "unknown", "untitled"):
        return None
    if _TRAINER_PLACEHOLDER.match(title) or _TIMESTAMP_PLACEHOLDER.match(title):
        return None
    return title


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def _resolve_paths(asset_type: str) -> dict[str, str]:
    registry = LORA_REGISTRY if asset_type == "lora" else EMBEDDING_REGISTRY
    paths = {}
    for name, cfg in registry.items():
        path = cfg.get("path")
        if path:
            paths[name] = str(path)
    return paths


def run(force: bool, online: bool, offline: bool, dry_run: bool) -> None:
    token = resolve_provider_token("civitai")[0] if online else None
    stats = {"online_used": 0, "online_failed": 0, "offline_used": 0, "skipped_url": 0}

    for asset_type, update_fn in (
        ("lora", meta_store.update_lora_meta),
        ("embedding", meta_store.update_embedding_meta),
    ):
        paths = _resolve_paths(asset_type)

        def set_title(name: str, title: str | None, source: str) -> bool:
            if not title or title == name:
                return False
            if not force:
                existing = meta_store.get(name, asset_type=asset_type)
                if existing and existing.title:
                    return False
            if dry_run:
                print(f"  [{asset_type}] {name!r} -> {title!r} ({source})")
                return True
            if update_fn(name, title=title):
                print(f"  [{asset_type}] {name!r} -> {title!r} ({source})")
                return True
            return False

        for name, path in paths.items():
            meta = meta_store.get(name, asset_type=asset_type)
            url = (meta.url if meta else "") or ""
            model_id = _civitai_model_id(url) if url else None

            title = None
            if model_id:
                # Authoritative source when a CivitAI page is known. If the
                # fetch fails, leave the title empty instead of guessing from
                # trainer metadata (which often carries the base-model name).
                if online:
                    title = fetch_civitai_title(model_id, token)
                    if title:
                        if set_title(name, title, "civitai"):
                            stats["online_used"] += 1
                    else:
                        stats["online_failed"] += 1
                elif url:
                    stats["skipped_url"] += 1
            else:
                stats["skipped_url"] += 1 if url else 0
                if offline:
                    title = _safetensors_title(path)
                    if title and set_title(name, title, "metadata"):
                        stats["offline_used"] += 1

            time.sleep(_REQUEST_DELAY if model_id else 0.0)

    print(
        "\nSummary:"
        f"\n  online titles fetched : {stats['online_used']}"
        f"\n  online fetch failures : {stats['online_failed']}"
        f"\n  offline (safetensors): {stats['offline_used']}"
        f"\n  URLs skipped (non-civ) : {stats['skipped_url']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite already-set titles")
    parser.add_argument("--dry-run", action="store_true", help="print what would change without writing")
    parser.add_argument("--online-only", action="store_true", help="only query CivitAI pages")
    parser.add_argument("--offline-only", action="store_true", help="only use safetensors metadata")
    args = parser.parse_args()

    online = not args.offline_only
    offline = not args.online_only
    run(force=args.force, online=online, offline=offline, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())