"""Model-driven catalog API for WebbDuck.

This router is additive to the legacy /models endpoints. It exposes the
architecture-free model profile produced by the runtime catalog so the Studio
can become capability-driven without breaking older clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from core.backends.krea2_lora import is_krea2_lora_entry
from models.catalog import descriptor_for_model, public_runtime_catalog, runtime_registry
from models.registry import LORA_REGISTRY, LORA_ROOT


router = APIRouter()


@router.get("/model-catalog")
def list_model_catalog():
    """Return every discovered image model with public capabilities/state."""
    return public_runtime_catalog()


def _krea_ui_loras() -> dict[str, dict[str, Any]]:
    """Return Krea LoRAs without a full filesystem header scan on every UI load.

    Most files are already in LORA_REGISTRY; compatibility checking only opens
    ambiguous transformer-family headers. For adapters the legacy scanner could
    not classify at all (notably some AI-Toolkit exports), the explicit
    ``lora/krea2`` namespace is the fast and authoritative discovery path.
    """
    found: dict[str, dict[str, Any]] = {}
    for name, cfg in LORA_REGISTRY.items():
        if is_krea2_lora_entry(cfg):
            found[str(name)] = dict(cfg)

    root = Path(LORA_ROOT).expanduser()
    for dirname in ("krea2", "krea", "krea-2"):
        folder = root / dirname
        if not folder.exists():
            continue
        try:
            candidates = folder.rglob("*.safetensors")
        except OSError:
            continue
        for path in candidates:
            if not path.is_file() or path.stem in found:
                continue
            found[path.stem] = {
                "path": path,
                "arch": "krea2",
                "weight": 1.0,
                "description": "Krea 2 LoRA",
                "source": "local",
            }
    return found


@router.get("/model-catalog/{model_name:path}/loras")
def list_model_loras(model_name: str):
    """Return LoRAs compatible with a runtime-catalog checkpoint."""
    registry = runtime_registry()
    try:
        descriptor = descriptor_for_model(model_name, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}") from exc

    if not descriptor.capabilities.lora:
        return []

    checkpoint_arch = str(descriptor.architecture or "").lower()
    if checkpoint_arch == "krea2":
        entries = _krea_ui_loras()
        return [
            {
                "name": name,
                "description": cfg.get("description", ""),
                "weight": cfg.get("weight", 1.0),
            }
            for name, cfg in sorted(entries.items())
        ]

    _FLUX_FAMILY = {"flux", "flux1", "flux2"}
    return [
        {
            "name": name,
            "description": cfg.get("description", ""),
            "weight": cfg.get("weight", 1.0),
        }
        for name, cfg in LORA_REGISTRY.items()
        if _lora_matches_checkpoint(cfg, checkpoint_arch, _FLUX_FAMILY)
    ]


def _lora_matches_checkpoint(
    lora_cfg: dict[str, Any],
    checkpoint_arch: str,
    flux_family: set[str],
) -> bool:
    """Return True if a LoRA entry is compatible with the given checkpoint arch."""
    if checkpoint_arch == "krea2":
        return is_krea2_lora_entry(lora_cfg)

    lora_arch = str(lora_cfg.get("arch") or "").lower()
    if lora_arch == checkpoint_arch:
        return True
    if checkpoint_arch == "flux" and lora_arch in flux_family:
        return True
    if lora_arch == "flux" and checkpoint_arch in flux_family:
        return True
    return False


@router.get("/model-catalog/{model_name:path}")
def get_model_profile(model_name: str):
    """Return one architecture-free model profile."""
    registry = runtime_registry()
    try:
        descriptor = descriptor_for_model(model_name, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}") from exc
    return descriptor.to_public_dict()
