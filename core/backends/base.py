"""Backend contract for checkpoint-driven image generation.

Backends encapsulate architecture/runtime-specific behavior. Higher layers
should route by ModelDescriptor rather than branching on architecture names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models.model_descriptor import ModelDescriptor


class GenerationBackend(ABC):
    """Architecture/runtime adapter used by the generic generation layer."""

    backend_id: str

    @abstractmethod
    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        """Return True when this backend can execute the selected checkpoint."""

    @abstractmethod
    def generate(self, descriptor: ModelDescriptor, settings: dict[str, Any], **kwargs: Any) -> Any:
        """Execute a generation request and return backend-specific result data."""

    def unload(self) -> None:
        """Release backend-owned runtime resources when applicable."""


class BackendResolver:
    """Registry that resolves a checkpoint descriptor to one backend adapter."""

    def __init__(self) -> None:
        self._backends: dict[str, GenerationBackend] = {}

    def register(self, backend: GenerationBackend) -> None:
        backend_id = str(getattr(backend, "backend_id", "") or "").strip()
        if not backend_id:
            raise ValueError("Generation backend must define backend_id")
        self._backends[backend_id] = backend

    def unregister(self, backend_id: str) -> None:
        self._backends.pop(str(backend_id), None)

    def resolve(self, descriptor: ModelDescriptor) -> GenerationBackend:
        preferred = self._backends.get(descriptor.backend)
        if preferred is not None and preferred.can_handle(descriptor):
            return preferred

        for backend in self._backends.values():
            if backend.can_handle(descriptor):
                return backend

        raise LookupError(
            f"No generation backend is registered for checkpoint '{descriptor.name}' "
            f"(backend={descriptor.backend!r})."
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(self._backends.keys())


backend_resolver = BackendResolver()
