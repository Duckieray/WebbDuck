"""Architecture-neutral checkpoint catalog used by runtime routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from models.discovery import (
    discover_hf_image_models,
    discover_local_image_models,
    merge_discovered_models,
    public_model_catalog,
    resolve_checkpoint_root,
    resolve_hf_cache_root,
)
from models.model_descriptor import ModelDescriptor, describe_registry_model


ROOT = Path(__file__).resolve().parent.parent


def _load_saved_defaults(checkpoint_root: Path) -> dict[str, dict[str, Any]]:
    """Read model-facing defaults without relying on the old MODEL_REGISTRY."""
    path = Path(checkpoint_root) / "checkpoints.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            continue
        defaults = value.get("defaults")
        if isinstance(defaults, dict):
            out[str(name)] = dict(defaults)
    return out


def build_runtime_registry(
    *,
    models_root: Path,
    checkpoint_root: Path | None = None,
    hf_cache: Path | None = None,
    saved_defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one checkpoint registry from generic local + HF discovery.

    The old SDXL-first ``MODEL_REGISTRY`` is deliberately not an input. Local
    discovery wins name collisions, followed by Hugging Face cache discovery.
    A neutral ``checkpoint/checkpoints.json`` sidecar may carry per-model user
    defaults without owning model discovery or architecture classification.
    """
    local_root = (
        Path(checkpoint_root)
        if checkpoint_root is not None
        else resolve_checkpoint_root(Path(models_root))
    )
    cache_root = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_root()

    local = discover_local_image_models(local_root)
    cached = discover_hf_image_models(cache_root)
    merged = merge_discovered_models(local, cached)

    defaults_by_name = (
        {str(name): dict(value) for name, value in saved_defaults.items()}
        if saved_defaults is not None
        else _load_saved_defaults(local_root)
    )
    for name, entry in merged.items():
        entry["defaults"] = dict(defaults_by_name.get(name) or {})
    return merged


def runtime_registry() -> dict[str, dict[str, Any]]:
    """Build the current runtime registry from architecture-neutral sources."""
    models_root = Path(os.getenv("WEBBDUCK_MODELS_DIR", str(ROOT))).expanduser()
    checkpoint_root = resolve_checkpoint_root(
        models_root,
        os.getenv("WEBBDUCK_CHECKPOINT_DIR"),
    )
    return build_runtime_registry(
        models_root=models_root,
        checkpoint_root=checkpoint_root,
        hf_cache=resolve_hf_cache_root(),
    )


def descriptor_for_model(
    name: str,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> ModelDescriptor:
    """Resolve a selected model name to its canonical descriptor."""
    source = registry if registry is not None else runtime_registry()
    entry = source.get(name)
    if entry is None:
        raise KeyError(f"Unknown checkpoint: {name}")
    return describe_registry_model(name, entry)


def public_runtime_catalog(
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the architecture-free public model catalog."""
    source = registry if registry is not None else runtime_registry()
    return public_model_catalog(source)
