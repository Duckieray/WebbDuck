"""Non-loading runtime readiness surface for checkpoint-driven generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from core.backends.base import backend_resolver
from core.model_runtime import register_installed_backends
from models.catalog import descriptor_for_model, runtime_registry


router = APIRouter()


def _source_state(path_value: str) -> dict[str, Any]:
    path = Path(str(path_value or "")).expanduser()
    if not path_value:
        return {"present": False, "path": ""}
    return {
        "present": path.exists(),
        "path": str(path),
        "kind": "file" if path.is_file() else ("directory" if path.is_dir() else "missing"),
    }


@router.get("/runtime-readiness")
def runtime_readiness() -> dict[str, Any]:
    """Probe installed backend environments without loading model weights."""
    registry = runtime_registry()
    register_installed_backends()
    items: list[dict[str, Any]] = []

    for name in sorted(registry, key=str.lower):
        descriptor = descriptor_for_model(name, registry)
        public = descriptor.to_public_dict()
        if not descriptor.supported:
            readiness = {
                "ready": False,
                "reason": "Model is discovered, but this build has no runnable backend for its format/workflow.",
            }
        else:
            try:
                readiness = backend_resolver.readiness(descriptor)
            except Exception as exc:
                readiness = {"ready": False, "reason": str(exc or exc.__class__.__name__)}

        items.append(
            {
                **public,
                "source_state": _source_state(descriptor.path),
                "runtime": readiness,
                "ready": bool(public.get("supported") and readiness.get("ready")),
            }
        )

    return {
        "ready": bool(items) and all(item["ready"] for item in items if item.get("supported")),
        "count": len(items),
        "items": items,
    }
