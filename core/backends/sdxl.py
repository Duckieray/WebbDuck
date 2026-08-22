"""SDXL backend exposed through WebbDuck's generic generation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.backends.base import GenerationBackend, backend_resolver
from models.model_descriptor import ModelDescriptor


def _hydrate_execution_registry() -> None:
    """Populate the mature SDXL stack from the canonical model catalog.

    The backend-owned SDXL implementation still uses a name->entry mapping for
    checkpoint/refiner/asset lookup. That mapping is no longer a discovery
    authority: rebuild its SDXL model entries from the architecture-neutral
    runtime catalog immediately before execution.
    """
    from models.catalog import runtime_registry
    from models.registry import MODEL_REGISTRY

    canonical = runtime_registry()
    sdxl_entries: dict[str, dict[str, Any]] = {}
    for name, raw_entry in canonical.items():
        entry = dict(raw_entry)
        if str(entry.get("arch") or entry.get("architecture") or "").lower() != "sdxl":
            continue
        entry["path"] = Path(str(entry.get("path") or ""))
        sdxl_entries[str(name)] = entry

    # Remove stale SDXL model entries only. Other legacy registry contents are
    # asset/runtime internals outside this backend migration and are not used as
    # canonical checkpoint discovery by model routing.
    for name in list(MODEL_REGISTRY):
        if str(MODEL_REGISTRY.get(name, {}).get("arch") or "").lower() == "sdxl":
            MODEL_REGISTRY.pop(name, None)
    MODEL_REGISTRY.update(sdxl_entries)


class SDXLDiffusersBackend(GenerationBackend):
    """Own the complete SDXL generation stack behind one backend boundary."""

    backend_id = "sdxl_diffusers"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "sdxl"

    def generate(self, descriptor: ModelDescriptor, settings: dict[str, Any], **kwargs: Any) -> Any:
        from core.backends.sdxl_runtime import run_generation

        selected = str(settings.get("base_model") or "")
        if selected and selected != descriptor.name:
            raise ValueError(
                f"Selected checkpoint changed during routing: {selected!r} != {descriptor.name!r}"
            )
        _hydrate_execution_registry()
        return run_generation(settings, cancel_event=kwargs.get("cancel_event"))

    def unload(self) -> None:
        from core.backends.sdxl_pipeline import pipeline_manager

        unload = getattr(pipeline_manager, "unload_all", None)
        if callable(unload):
            unload()


_backend = SDXLDiffusersBackend()


def ensure_registered() -> SDXLDiffusersBackend:
    """Idempotently register the installed SDXL backend."""
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
