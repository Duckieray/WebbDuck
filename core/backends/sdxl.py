"""SDXL backend exposed through WebbDuck's generic generation contract."""

from __future__ import annotations

from typing import Any

from core.backends.base import GenerationBackend, backend_resolver
from models.model_descriptor import ModelDescriptor


class SDXLDiffusersBackend(GenerationBackend):
    """Own SDXL routing while the mature SDXL implementation is split internally."""

    backend_id = "sdxl_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "sdxl"

    def generate(self, descriptor: ModelDescriptor, settings: dict[str, Any], **kwargs: Any) -> Any:
        # The backend owns the dependency on the existing SDXL generation stack.
        # Generic routing code has no SDXL pipeline knowledge.
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


_backend = SDXLDiffusersBackend()


def ensure_registered() -> SDXLDiffusersBackend:
    """Idempotently register the installed SDXL backend."""
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
