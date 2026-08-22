"""Model-driven catalog API for WebbDuck.

This router is additive to the legacy /models endpoints. It exposes the
architecture-free model profile produced by the runtime catalog so the Studio
can become capability-driven without breaking older clients.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.catalog import descriptor_for_model, public_runtime_catalog, runtime_registry


router = APIRouter()


@router.get("/model-catalog")
def list_model_catalog():
    """Return every discovered image model with public capabilities/state."""
    return public_runtime_catalog()


@router.get("/model-catalog/{model_name:path}")
def get_model_profile(model_name: str):
    """Return one architecture-free model profile."""
    registry = runtime_registry()
    try:
        descriptor = descriptor_for_model(model_name, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}") from exc
    return descriptor.to_public_dict()
