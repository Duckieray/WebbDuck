"""Checkpoint-driven runtime routing for WebbDuck generation jobs."""

from __future__ import annotations

from typing import Any

from core.backends.base import backend_resolver
from core.backends.flux import ensure_registered as ensure_flux_registered
from core.backends.krea2 import ensure_registered as ensure_krea2_registered
from core.backends.qwen_image import ensure_registered as ensure_qwen_image_registered
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


def register_installed_backends() -> None:
    """Composition root for image backends shipped by this WebbDuck build."""
    ensure_sdxl_registered()
    ensure_flux_registered()
    ensure_krea2_registered()
    ensure_qwen_image_registered()


def run_selected_model(settings: dict[str, Any], cancel_event=None, progress_callback=None):
    """Resolve the selected checkpoint and execute its backend."""
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
            f"Checkpoint '{descriptor.name}' is recognized, but no runnable backend is installed for it."
        )

    register_installed_backends()
    backend = backend_resolver.resolve(descriptor)

    # Readiness is the hard boundary between model routing and worker launch.
    # Never discover a stale/missing isolated runtime only after a GPU job has
    # started and a backend-specific subprocess has imported half its stack.
    backend.require_ready(descriptor)

    settings["model_profile"] = descriptor.to_public_dict()
    return backend.generate(
        descriptor,
        settings,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
