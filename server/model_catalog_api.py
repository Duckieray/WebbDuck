"""Model-driven catalog API for WebbDuck.

This router is additive to the legacy /models endpoints. It exposes the
architecture-free model profile produced by the runtime catalog so the Studio
can become capability-driven without breaking older clients.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.catalog import descriptor_for_model, public_runtime_catalog, runtime_registry
from models.registry import LORA_REGISTRY


router = APIRouter()


@router.get("/model-catalog")
def list_model_catalog():
    """Return every discovered image model with public capabilities/state."""
    return public_runtime_catalog()


@router.get("/model-catalog/{model_name:path}/loras")
def list_model_loras(model_name: str):
    """Return LoRAs compatible with a runtime-catalog checkpoint.

    The legacy ``/models/{name}/loras`` endpoint is backed by MODEL_REGISTRY and
    therefore cannot see architecture-neutral checkpoints such as standalone
    FLUX GGUF transformers. Resolve the selected checkpoint through the runtime
    catalog instead, then filter the shared LoRA registry by internal arch.
    """
    registry = runtime_registry()
    try:
        descriptor = descriptor_for_model(model_name, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}") from exc

    if not descriptor.capabilities.lora:
        return []

    checkpoint_arch = str(descriptor.architecture or "").lower()
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
    """Return True if a LoRA entry is compatible with the given checkpoint arch.

    FLUX.1 and FLUX.2 are distinct architectures with incompatible transformer
    shapes, but the checkpoint is often discovered as generic ``"flux"``.  A
    generic checkpoint accepts any FLUX-family LoRA; a version-specific
    checkpoint (``"flux1"`` or ``"flux2"``) only accepts matching LoRAs.
    """
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
