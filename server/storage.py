"""Image/session storage and searchable manifest index."""

import json
import os
import threading
import time
from pathlib import Path

# Check for custom output directory from environment variable
custom_out = os.environ.get("WEBBDUCK_OUTPUT_DIR")
if custom_out:
    BASE = Path(custom_out)
else:
    BASE = Path("outputs")

BASE.mkdir(exist_ok=True, parents=True)

MANIFEST_FILE = BASE / "manifest.jsonl"
SESSION_LOG = BASE / "session_log.json"
_manifest_lock = threading.Lock()
_manifest_cache: list[dict] | None = None
_manifest_cache_mtime: float = -1.0


def to_web_path(path: Path) -> str:
    """Convert valid filesystem path to web-accessible path."""
    try:
        rel = path.resolve().relative_to(BASE.resolve())
        return f"outputs/{rel.as_posix()}"
    except ValueError:
        return f"outputs/{path.name}"


def resolve_web_path(path_str: str) -> Path:
    """Convert web-accessible path to filesystem path."""
    if "outputs/" in path_str:
        rel = path_str.split("outputs/", 1)[1]
        return BASE / rel.lstrip("/")
    return Path(path_str)


def sanitize_settings(settings: dict) -> dict:
    """Remove non-serializable runtime fields from settings."""
    clean = dict(settings or {})
    clean.pop("input_image", None)
    clean.pop("mask_image", None)
    clean.pop("image", None)
    return clean


def _sort_image_web_paths(images: list[str]) -> list[str]:
    def sort_key(path_str: str):
        stem = Path(path_str).stem
        if stem.isdigit():
            return (0, int(stem), "")
        return (1, 0, stem)
    return sorted(images, key=sort_key)


def _make_manifest_entry(image_path: Path, run_dir: Path, meta: dict) -> dict:
    web_image = to_web_path(image_path)
    variant_path = image_path.with_name(f"{image_path.stem}_upscaled.png")
    variant_web = to_web_path(variant_path) if variant_path.exists() else None
    safe_meta = sanitize_settings(meta)
    timestamp = float(safe_meta.get("timestamp", 0.0) or 0.0)
    if timestamp <= 0:
        timestamp = time.time()
        safe_meta["timestamp"] = timestamp

    loras = safe_meta.get("loras")
    if not isinstance(loras, list):
        loras = []

    searchable_parts = [
        str(safe_meta.get("prompt", "")),
        str(safe_meta.get("negative_prompt", "")),
        str(safe_meta.get("base_model", "")),
        str(safe_meta.get("scheduler", "")),
        str(safe_meta.get("seed", "")),
        " ".join([
            item if isinstance(item, str) else str(item.get("name") or item.get("model") or "")
            for item in loras if isinstance(item, (str, dict))
        ]),
    ]

    return {
        "run": run_dir.name,
        "image": web_image,
        "variant": variant_web,
        "timestamp": timestamp,
        "meta": safe_meta,
        "searchable": " ".join(searchable_parts).lower(),
    }


def _entry_searchable(entry: dict) -> str:
    searchable = str(entry.get("searchable", "") or "").strip().lower()
    if searchable:
        return searchable

    meta = entry.get("meta", {}) or {}
    loras = meta.get("loras")
    if not isinstance(loras, list):
        loras = []
    lora_names = " ".join(
        item if isinstance(item, str) else str(item.get("name") or item.get("model") or "")
        for item in loras if isinstance(item, (str, dict))
    )
    parts = [
        str(meta.get("prompt", "")),
        str(meta.get("negative_prompt", "")),
        str(meta.get("base_model", "")),
        str(meta.get("scheduler", "")),
        str(meta.get("seed", "")),
        lora_names,
    ]
    return " ".join(parts).lower()


def _load_manifest_entries() -> list[dict]:
    if not MANIFEST_FILE.exists():
        return []

    entries = []
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def _manifest_mtime() -> float:
    try:
        return MANIFEST_FILE.stat().st_mtime
    except FileNotFoundError:
        return -1.0


def _load_manifest_entries_cached() -> list[dict]:
    global _manifest_cache, _manifest_cache_mtime
    mtime = _manifest_mtime()
    if _manifest_cache is not None and mtime == _manifest_cache_mtime:
        return _manifest_cache

    entries = _load_manifest_entries()
    _manifest_cache = entries
    _manifest_cache_mtime = mtime
    return entries


def _set_manifest_cache(entries: list[dict]):
    global _manifest_cache, _manifest_cache_mtime
    _manifest_cache = entries
    _manifest_cache_mtime = _manifest_mtime()


def _write_manifest_entries(entries: list[dict]):
    tmp_path = MANIFEST_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp_path.replace(MANIFEST_FILE)
    _set_manifest_cache(entries)


def rebuild_manifest():
    """Rebuild manifest index from output folders."""
    entries = []
    runs = sorted([p for p in BASE.iterdir() if p.is_dir()], reverse=True)

    for run_dir in runs:
        meta_file = run_dir / "meta.json"
        if not meta_file.exists():
            continue

        try:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        for image_path in sorted(run_dir.glob("*.png"), key=lambda p: p.name):
            if image_path.name.endswith("_upscaled.png"):
                continue
            entries.append(_make_manifest_entry(image_path, run_dir, meta))

    entries.sort(key=lambda x: float(x.get("timestamp", 0.0)), reverse=True)
    with _manifest_lock:
        _write_manifest_entries(entries)


def ensure_manifest():
    """Ensure manifest exists; build it lazily on first use."""
    if MANIFEST_FILE.exists():
        return
    rebuild_manifest()


def append_manifest_entries(image_paths: list[Path], run_dir: Path, meta: dict):
    """Append fresh generation outputs to manifest."""
    if not image_paths:
        return
    entries = [_make_manifest_entry(path, run_dir, meta) for path in image_paths]
    with _manifest_lock:
        MANIFEST_FILE.parent.mkdir(exist_ok=True, parents=True)
        with MANIFEST_FILE.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        cached = _load_manifest_entries_cached()
        cached.extend(entries)
        _set_manifest_cache(cached)


def remove_manifest_image(path_str: str):
    """Remove a single image (or variant) from manifest."""
    ensure_manifest()
    target = to_web_path(resolve_web_path(path_str))

    with _manifest_lock:
        entries = _load_manifest_entries_cached()
        filtered = [e for e in entries if e.get("image") != target and e.get("variant") != target]
        if len(filtered) != len(entries):
            _write_manifest_entries(filtered)


def remove_manifest_run(run_name: str):
    """Remove all entries for a deleted run."""
    ensure_manifest()
    with _manifest_lock:
        entries = _load_manifest_entries_cached()
        filtered = [e for e in entries if e.get("run") != run_name]
        if len(filtered) != len(entries):
            _write_manifest_entries(filtered)


def search_manifest_sessions(query: str, start: int = 0, limit: int = 60) -> list[dict]:
    """Search manifest globally and return gallery-compatible sessions."""
    ensure_manifest()
    q = (query or "").strip().lower()
    if not q:
        return []

    terms = [t for t in q.split() if t]
    if not terms:
        return []

    entries = _load_manifest_entries_cached()
    if not entries:
        # Recover from empty/stale manifest quickly.
        rebuild_manifest()
        entries = _load_manifest_entries_cached()

    matched = []
    for entry in entries:
        image_path = entry.get("image")
        if not image_path:
            continue
        searchable = _entry_searchable(entry)
        if not all(term in searchable for term in terms):
            continue

        score = 0
        meta = entry.get("meta", {}) or {}
        prompt = str(meta.get("prompt", "")).lower()
        negative = str(meta.get("negative_prompt", "")).lower()
        # Rank prompt hits higher than other metadata hits.
        for term in terms:
            if term in prompt:
                score += 3
            elif term in negative:
                score += 2
            else:
                score += 1
        matched.append((score, entry))

    matched.sort(
        key=lambda x: (
            -x[0],
            -float(x[1].get("timestamp", 0.0)),
        )
    )

    grouped = {}
    order = []
    for _, entry in matched:
        run = entry.get("run")
        if not run:
            continue
        if run not in grouped:
            grouped[run] = {
                "run": run,
                "images": [],
                "variants": {},
                "meta": entry.get("meta", {}),
            }
            order.append(run)

        web_image = entry.get("image")
        if web_image and web_image not in grouped[run]["images"]:
            grouped[run]["images"].append(web_image)

        variant = entry.get("variant")
        if variant and web_image:
            grouped[run]["variants"][Path(web_image).name] = variant

    sessions = []
    for run in order:
        session = grouped[run]
        session["images"] = _sort_image_web_paths(session["images"])
        sessions.append(session)

    return sessions[start:start + limit]


def save_images(images, settings):
    """Save generated images + metadata and update manifest."""
    run = BASE / time.strftime("%Y-%m-%d_%H-%M-%S")
    run.mkdir()

    image_paths = []
    web_paths = []
    for i, img in enumerate(images):
        p = run / f"{i}.png"
        img.save(p)
        image_paths.append(p)
        web_paths.append(to_web_path(p))

    clean_settings = sanitize_settings(settings)
    if "timestamp" not in clean_settings:
        clean_settings["timestamp"] = time.time()

    with (run / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(clean_settings, f, indent=2)

    append_manifest_entries(image_paths, run, clean_settings)
    return web_paths


def append_session_entry(entry: dict):
    """Append entry to session log."""
    SESSION_LOG.parent.mkdir(exist_ok=True)

    if SESSION_LOG.exists():
        data = json.loads(SESSION_LOG.read_text())
    else:
        data = []

    data.append(entry)
    SESSION_LOG.write_text(json.dumps(data, indent=2))
