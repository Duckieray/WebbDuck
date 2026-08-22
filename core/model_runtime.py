"""Checkpoint-driven runtime bridge for WebbDuck generation jobs."""

from __future__ import annotations

from typing import Any

from core.backends.base import backend_resolver
from core.backends.sdxl import ensure_registered as ensure_sdxl_registered
from models.catalog import descriptor_for_model
from models.model_descriptor import ModelDescriptor


def requested_operation(settings: dict[str, Any]) -> str:
    """Infer the requested image operation without architecture knowledge."""
    if settings.get("smart_extend"):
        return "outpaint"
    if settings.get("mask_image"):
        return "inpaint"
    if settings.get("image") or settings.get("input_image"):
        return "img2img"
    return "text2img"


def _validate_operation(descriptor: ModelDescriptor, operation: str) -> None:
    capability_name = {
        "text2img": "text2img",
        "img2img": "img2img",
        "inpaint": "inpaint",
        "outpaint": "outpaint",
    }.get(operation)
    if not capability_name:
        raise ValueError(f"Unknown image operation: {operation}")
    if not getattr(descriptor.capabilities, capability_name, False):
        raise ValueError(
            f"Checkpoint '{descriptor.name}' does not support the requested {operation} workflow."
        )


def run_selected_model(settings: dict[str, Any], cancel_event=None):
    """Resolve the selected checkpoint and execute it through a registered backend.

    Recognized checkpoints with unfinished runtime adapters fail here, before the
    legacy SDXL pipeline can see them.
    """
    model_name = str(settings.get("base_model") or "").strip()
    if not model_name:
        raise ValueError("A checkpoint must be selected before generation.")

    try:
        descriptor = descriptor_for_model(model_name)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    operation = requested_operation(settings)
    _validate_operation(descriptor, operation)

    if not descriptor.supported:
        raise RuntimeError(
            f"Checkpoint '{descriptor.name}' was discovered successfully, but its generation runtime "
            "is not available in this WebbDuck build yet."
        )

    # Existing SDXL is the first live adapter. Future adapters register through
    # their own composition modules without changing this router.
    ensure_sdxl_registered()
    backend = backend_resolver.resolve(descriptor)

    settings["model_profile"] = descriptor.to_public_dict()
    return backend.generate(descriptor, settings, cancel_event=cancel_event)
