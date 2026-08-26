"""Architecture-neutral checkpoint discovery for WebbDuck.

Discovery is deliberately separate from execution. It may recognize checkpoints
whose runtime adapter is not implemented yet. Those checkpoints are described
correctly and reported as non-runnable until their adapter is wired.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from models.model_descriptor import (
    UNKNOWN_ARCHITECTURE,
    backend_for_architecture,
    constraints_for_architecture,
    describe_registry_model,
    detect_diffusers_architecture,
)
from models.single_file_inspection import inspect_single_image_checkpoint


IMAGE_ARCHITECTURES = {"sdxl", "flux", "krea2", "qwen_image"}


def resolve_hf_cache_root() -> Path:
    """Resolve the same Hugging Face hub cache conventions used by HF tooling."""
    explicit = os.getenv("WEBBDUCK_HF_CACHE_DIR")
    if explicit:
        value = Path(explicit).expanduser()
        return value if value.name.lower() == "hub" else value / "hub"

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.getenv(key)
        if raw:
            value = Path(raw).expanduser()
            return value if value.name.lower() == "hub" else value / "hub"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_checkpoint_root(models_root: Path, explicit: str | None = None) -> Path:
    """Return the architecture-neutral local checkpoint root."""
    if explicit:
        return Path(explicit).expanduser()

    root = Path(models_root).expanduser()
    for candidate in (root / "checkpoint", root / "checkpoints"):
        if candidate.exists():
            return candidate
    for legacy in (root / "checkpoint" / "sdxl", root / "checkpoints" / "sdxl"):
        if legacy.exists():
            return legacy.parent
    return root / "checkpoint"


def _architecture_hint_from_path(path: Path, root: Path) -> str:
    """Use explicit family folder names only as a conservative fallback hint."""
    try:
        parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    except ValueError:
        parts = [part.lower() for part in path.parts[:-1]]
    aliases = {
        "sdxl": "sdxl",
        "flux": "flux",
        "krea2": "krea2",
        "krea-2": "krea2",
        "qwen_image": "qwen_image",
        "qwen-image": "qwen_image",
    }
    for part in reversed(parts):
        if part in aliases:
            return aliases[part]
    return UNKNOWN_ARCHITECTURE


def _flux2_klein_gguf_detection(path: Path) -> dict[str, Any] | None:
    """Recognize released FLUX.2 Klein GGUF transformer filenames conservatively.

    GGUF is a component checkpoint, not a whole Diffusers pipeline. The filename
    therefore needs to identify a known FLUX.2 Klein family strongly enough for
    WebbDuck to select the matching support-component repository without parsing
    multi-gigabyte tensor data during catalog discovery.
    """
    filename = path.name.lower()
    normalized = filename.replace("_", "-").replace(".", "-")
    if "flux-2-klein" not in normalized:
        return None

    size_match = re.search(r"(?:^|[-_])(4b|9b)(?:[-_.]|$)", filename, re.IGNORECASE)
    if not size_match:
        return None
    parameter_size = size_match.group(1).upper()
    component_model = f"black-forest-labs/FLUX.2-klein-{parameter_size}"

    quantization = "unknown"
    quant_match = re.search(
        r"(?:^|[-_])(Q\d+(?:_[A-Z0-9]+)*|BF16|F16)$",
        path.stem.upper(),
    )
    if quant_match:
        quantization = quant_match.group(1)

    is_base = "klein-base" in normalized
    if is_base:
        component_model = f"black-forest-labs/FLUX.2-klein-base-{parameter_size}"

    return {
        "method": "gguf_filename",
        "confidence": "high",
        "family": "flux2_klein",
        "variant": "base" if is_base else "distilled",
        "parameter_size": parameter_size,
        "quantization": quantization,
        "component_model": component_model,
    }


def _registry_entry(
    *,
    path: Path,
    architecture: str,
    source: str,
    format_name: str,
    detection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": format_name,
        "arch": architecture,
        "backend": backend_for_architecture(architecture),
        "path": path,
        "source": source,
        "constraints": constraints_for_architecture(architecture),
        "detection": dict(detection or {}),
    }


def _register_unique(models: dict[str, dict[str, Any]], preferred_name: str, entry: dict[str, Any]) -> None:
    name = preferred_name
    if name not in models:
        models[name] = entry
        return
    existing = models[name]
    if str(existing.get("path")) == str(entry.get("path")):
        return
    source = str(entry.get("source") or "model")
    suffix = 2
    candidate = f"{name} [{source}]"
    while candidate in models:
        candidate = f"{name} [{source} {suffix}]"
        suffix += 1
    models[candidate] = entry


def discover_local_image_models(checkpoint_root: Path) -> dict[str, dict[str, Any]]:
    """Recursively discover local image checkpoints beneath one generic root."""
    root = Path(checkpoint_root)
    models: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return models

    def walk(current: Path) -> None:
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return
        for item in children:
            if item.is_file() and item.suffix.lower() == ".gguf":
                detection = _flux2_klein_gguf_detection(item)
                if detection is None:
                    continue
                _register_unique(
                    models,
                    item.stem,
                    _registry_entry(
                        path=item,
                        architecture="flux",
                        source="local",
                        format_name="gguf",
                        detection=detection,
                    ),
                )
                continue

            if item.is_file() and item.suffix.lower() == ".safetensors":
                architecture, detection = inspect_single_image_checkpoint(item)
                if architecture not in IMAGE_ARCHITECTURES:
                    # Structural inspection is authoritative for newly supported
                    # single-file families. Preserve the established explicit
                    # SDXL folder convention until SDXL tensor introspection is
                    # moved into the same inspector.
                    hint = _architecture_hint_from_path(item, root)
                    if hint != "sdxl":
                        continue
                    architecture = hint
                    detection = {"method": "family_folder", "confidence": "medium"}

                _register_unique(
                    models,
                    item.stem,
                    _registry_entry(
                        path=item,
                        architecture=architecture,
                        source="local",
                        format_name="single",
                        detection=detection,
                    ),
                )
                continue

            if not item.is_dir():
                continue

            if (item / "model_index.json").exists() or (item / "unet" / "config.json").exists():
                architecture, detection = detect_diffusers_architecture(item)
                if architecture in IMAGE_ARCHITECTURES:
                    _register_unique(
                        models,
                        item.name,
                        _registry_entry(
                            path=item,
                            architecture=architecture,
                            source="local",
                            format_name="diffusers",
                            detection=detection,
                        ),
                    )
                    continue
            walk(item)

    walk(root)
    return models


def _snapshot_dirs(repo: Path) -> list[Path]:
    snapshots = repo / "snapshots"
    try:
        items = [item for item in snapshots.iterdir() if item.is_dir()]
    except OSError:
        return []
    return sorted(items, key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)


def _repo_display_name(repo: Path) -> str:
    return repo.name.removeprefix("models--").replace("--", "/")


def discover_hf_image_models(hf_cache: Path) -> dict[str, dict[str, Any]]:
    """Discover recognized image Diffusers snapshots in a Hugging Face hub cache."""
    cache = Path(hf_cache)
    models: dict[str, dict[str, Any]] = {}
    if not cache.exists():
        return models

    try:
        repos = sorted(cache.glob("models--*"), key=lambda item: item.name.lower())
    except OSError:
        return models

    for repo in repos:
        name = _repo_display_name(repo)
        for snapshot in _snapshot_dirs(repo):
            if not (snapshot / "model_index.json").exists():
                continue
            architecture, detection = detect_diffusers_architecture(snapshot)
            if architecture not in IMAGE_ARCHITECTURES:
                continue
            models[name] = _registry_entry(
                path=snapshot,
                architecture=architecture,
                source="hf_cache",
                format_name="diffusers",
                detection={**detection, "repo": name},
            )
            break
    return models


def merge_discovered_models(*sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge sources in order, preferring earlier entries on name collisions."""
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for name, entry in source.items():
            if name not in merged:
                merged[name] = dict(entry)
    return merged


def public_model_catalog(registry: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert registry entries into the architecture-free API contract."""
    return [
        describe_registry_model(name, entry).to_public_dict()
        for name, entry in sorted(registry.items(), key=lambda item: item[0].lower())
    ]
