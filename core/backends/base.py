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

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        """Probe runtime availability without loading model weights."""
        ready = bool(self.can_handle(descriptor))
        return {
            "ready": ready,
            "reason": None if ready else f"Installed backend cannot handle checkpoint '{descriptor.name}'.",
        }

    def require_ready(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        """Fail before model execution when the backend runtime is unavailable.

        Readiness is intentionally a backend-owned contract.  The generic
        generation layer can enforce it without learning architecture/runtime
        names, while each backend remains free to provide an actionable repair
        hint for its isolated environment.
        """
        payload = self.readiness(descriptor)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Checkpoint '{descriptor.name}' runtime readiness check returned an invalid result."
            )
        if bool(payload.get("ready")):
            return payload

        parts = [
            f"Checkpoint '{descriptor.name}' runtime is not ready.",
            str(payload.get("reason") or "The selected backend runtime failed its readiness check.").strip(),
        ]

        missing = payload.get("missing")
        if isinstance(missing, list) and missing:
            labels: list[str] = []
            for item in missing:
                if not isinstance(item, dict):
                    continue
                module = str(item.get("module") or "").strip()
                symbol = str(item.get("symbol") or "").strip()
                label = f"{module}.{symbol}" if module and symbol else module
                if label and label not in labels:
                    labels.append(label)
            if labels:
                parts.append("Missing runtime imports: " + ", ".join(labels) + ".")

        python_path = str(payload.get("python") or "").strip()
        if python_path:
            parts.append(f"Runtime Python: {python_path}.")

        repair_hint = str(payload.get("repair_hint") or "").strip()
        if repair_hint:
            parts.append(repair_hint)

        raise RuntimeError(" ".join(part for part in parts if part).strip())

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

    def readiness(self, descriptor: ModelDescriptor) -> dict[str, Any]:
        backend = self.resolve(descriptor)
        payload = backend.readiness(descriptor)
        if not isinstance(payload, dict):
            return {"ready": False, "reason": "Backend returned an invalid readiness payload."}
        return {
            **payload,
            "ready": bool(payload.get("ready")),
        }

    def ids(self) -> tuple[str, ...]:
        return tuple(self._backends.keys())


backend_resolver = BackendResolver()
