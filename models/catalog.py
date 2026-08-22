"""Unified checkpoint catalog used by model-driven runtime routing.

This module bridges the legacy WebbDuck registry with the architecture-neutral
discovery layer. Existing registry entries remain authoritative so current SDXL
installs keep their names/defaults, while newly discovered checkpoints can be
recognized without being sent through the legacy SDXL pipeline by accident.
"""

from __future__ import annotations

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


def build_runtime_registry(
    legacy_registry: Mapping[str, Mapping[str, Any]],
    *,
    models_root: Path,
    checkpoint_root: Path | None = None,
    hf_cache: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return one registry spanning legacy, local-generic, and HF checkpoints.

    Precedence is intentionally legacy -> local generic -> HF cache. This keeps
    existing saved model names and defaults stable while still making new model
    families discoverable.
    """
    local_root = Path(checkpoint_root) if checkpoint_root is not None else resolve_checkpoint_root(Path(models_root))
    cache_root = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_root()

    legacy = {name: dict(entry) for name, entry in legacy_registry.items()}
    local = discover_local_image_models(local_root)
    cached = discover_hf_image_models(cache_root)
    merged = merge_discovered_models(legacy, local, cached)

    # Carry persisted defaults from the legacy registry onto a newly discovered
    # entry when they resolve to the same checkpoint name.
    for name, entry in merged.items():
        legacy_entry = legacy.get(name)
        if legacy_entry and "defaults" in legacy_entry:
            entry["defaults"] = dict(legacy_entry.get("defaults") or {})
        else:
            entry.setdefault("defaults", {})
    return merged


def runtime_registry() -> dict[str, dict[str, Any]]:
    """Build the current runtime registry using WebbDuck's configured roots."""
    from models import registry as legacy

    # Do not reuse legacy.CHECKPOINT_ROOT here: older installs resolve that
    # value directly to checkpoint/sdxl. Resolve from MODELS_ROOT so sibling
    # architecture folders participate automatically.
    generic_checkpoint_root = resolve_checkpoint_root(
        legacy.MODELS_ROOT,
        os.getenv("WEBBDUCK_CHECKPOINT_DIR"),
    )

    return build_runtime_registry(
        legacy.MODEL_REGISTRY,
        models_root=legacy.MODELS_ROOT,
        checkpoint_root=generic_checkpoint_root,
        hf_cache=legacy.HF_CACHE,
    )


def descriptor_for_model(name: str, registry: Mapping[str, Mapping[str, Any]] | None = None) -> ModelDescriptor:
    """Resolve a selected model name to its canonical descriptor."""
    source = registry if registry is not None else runtime_registry()
    entry = source.get(name)
    if entry is None:
        raise KeyError(f"Unknown checkpoint: {name}")
    return describe_registry_model(name, entry)


def public_runtime_catalog(registry: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return the architecture-free public model catalog."""
    source = registry if registry is not None else runtime_registry()
    return public_model_catalog(source)
