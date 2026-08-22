"""Adapter exposing WebbDuck's existing SDXL runtime through the generic backend contract."""

from __future__ import annotations

from typing import Any

from core.backends.base import GenerationBackend, backend_resolver
from models.model_descriptor import ModelDescriptor


class SDXLLegacyBackend(GenerationBackend):
    """Thin compatibility adapter around the proven pre-overhaul SDXL path."""

    backend_id = "sdxl_legacy"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "sdxl"

    def generate(self, descriptor: ModelDescriptor, settings: dict[str, Any], **kwargs: Any) -> Any:
        # Lazy import avoids a module cycle: core.generation owns the existing
        # implementation and may eventually become a backend-internal module.
        from core.generation import run_generation

        selected = str(settings.get("base_model") or "")
        if selected and selected != descriptor.name:
            raise ValueError(
                f"Selected checkpoint changed during routing: {selected!r} != {descriptor.name!r}"
            )
        return run_generation(settings, cancel_event=kwargs.get("cancel_event"))

    def unload(self) -> None:
        from core.pipeline import pipeline_manager

        unload = getattr(pipeline_manager, "unload_all", None)
        if callable(unload):
            unload()


def ensure_registered() -> SDXLLegacyBackend:
    """Idempotently register the legacy SDXL adapter."""
    backend = SDXLLegacyBackend()
    if backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(backend)
    return backend
