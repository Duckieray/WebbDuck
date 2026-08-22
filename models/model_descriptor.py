"""Architecture-agnostic checkpoint description and introspection.

This module is intentionally independent from Diffusers pipeline construction.
It answers two questions only:

1. What kind of model is this checkpoint?
2. What can WebbDuck safely offer when that checkpoint is selected?

The user-facing product remains checkpoint-driven. Architecture/backend values
are internal routing metadata and must not become required UI choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


UNKNOWN_ARCHITECTURE = "unknown"
UNSUPPORTED_BACKEND = "unsupported"


@dataclass(frozen=True)
class ModelCapabilities:
    text2img: bool = False
    img2img: bool = False
    inpaint: bool = False
    outpaint: bool = False
    lora: bool = False
    embeddings: bool = False
    second_pass: bool = False
    negative_prompt: bool = False
    prompt_2: bool = False
    clip_skip: bool = False
    identity_adapter: bool = False
    tokenize: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    path: str
    format: str
    source: str
    architecture: str = UNKNOWN_ARCHITECTURE
    backend: str = UNSUPPORTED_BACKEND
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    defaults: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        """Whether the selected checkpoint is runnable in the current build."""
        return backend_is_implemented(self.backend) and self.capabilities.text2img

    def to_internal_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supported"] = self.supported
        return payload

    def to_public_dict(self) -> dict[str, Any]:
        """Return the model-driven contract the UI needs."""
        return {
            "name": self.name,
            "type": self.format,
            "defaults": dict(self.defaults),
            "capabilities": self.capabilities.to_dict(),
            "constraints": dict(self.constraints),
            "supported": self.supported,
        }


_CAPABILITIES: dict[str, ModelCapabilities] = {
    "sdxl": ModelCapabilities(
        text2img=True,
        img2img=True,
        inpaint=True,
        outpaint=True,
        lora=True,
        embeddings=True,
        second_pass=True,
        negative_prompt=True,
        prompt_2=True,
        clip_skip=True,
        identity_adapter=True,
        tokenize=True,
    ),
    # FLUX.2 klein uses one pipeline for text generation and image editing.
    # Asset capabilities stay false until the backend actually wires them.
    "flux": ModelCapabilities(text2img=True, img2img=True),
    "krea2": ModelCapabilities(text2img=True, lora=True),
    "qwen_image": ModelCapabilities(text2img=True, negative_prompt=True),
}

_BACKENDS: dict[str, str] = {
    "sdxl": "sdxl_legacy",
    "flux": "flux_diffusers",
    "krea2": "krea2_diffusers",
    "qwen_image": "qwen_image_diffusers",
}

_IMPLEMENTED_BACKENDS = {"sdxl_legacy", "flux_diffusers"}

_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "sdxl": {"dimension_multiple": 8},
    "flux": {"dimension_multiple": 8},
    "krea2": {"dimension_multiple": 16},
    "qwen_image": {"dimension_multiple": 16},
}

_DEFAULTS: dict[str, dict[str, Any]] = {
    "flux": {
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 1.0,
    },
}


def capabilities_for_architecture(architecture: str | None) -> ModelCapabilities:
    return _CAPABILITIES.get((architecture or "").lower(), ModelCapabilities())


def backend_for_architecture(architecture: str | None) -> str:
    return _BACKENDS.get((architecture or "").lower(), UNSUPPORTED_BACKEND)


def backend_is_implemented(backend: str | None) -> bool:
    return (backend or "") in _IMPLEMENTED_BACKENDS


def constraints_for_architecture(architecture: str | None) -> dict[str, Any]:
    return dict(_CONSTRAINTS.get((architecture or "").lower(), {}))


def defaults_for_architecture(architecture: str | None) -> dict[str, Any]:
    return dict(_DEFAULTS.get((architecture or "").lower(), {}))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _class_tokens(config: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in ("_class_name", "architectures", "model_type"):
        raw = config.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return " ".join(values).lower()


def detect_diffusers_architecture(path: Path) -> tuple[str, dict[str, Any]]:
    """Infer an image architecture from standard model configuration files."""
    path = Path(path)
    model_index = _read_json(path / "model_index.json")
    transformer_config = _read_json(path / "transformer" / "config.json")
    unet_config = _read_json(path / "unet" / "config.json")

    tokens = " ".join(
        part
        for part in (
            _class_tokens(model_index),
            _class_tokens(transformer_config),
            _class_tokens(unet_config),
        )
        if part
    )

    if "krea2" in tokens or "krea-2" in tokens:
        return "krea2", {"method": "config_class", "confidence": "high"}
    if "qwenimage" in tokens or "qwen_image" in tokens or "qwen-image" in tokens:
        return "qwen_image", {"method": "config_class", "confidence": "high"}
    if "flux" in tokens:
        return "flux", {"method": "config_class", "confidence": "high"}
    if "stablediffusionxl" in tokens or "stable_diffusion_xl" in tokens:
        return "sdxl", {"method": "config_class", "confidence": "high"}

    if (path / "unet" / "config.json").exists() and (path / "text_encoder_2").exists():
        return "sdxl", {"method": "directory_structure", "confidence": "medium"}

    return UNKNOWN_ARCHITECTURE, {"method": "unrecognized", "confidence": "none"}


def describe_registry_model(name: str, entry: Mapping[str, Any]) -> ModelDescriptor:
    """Build a canonical descriptor from a registry entry."""
    architecture = str(
        entry.get("arch") or entry.get("architecture") or UNKNOWN_ARCHITECTURE
    ).lower()
    path = Path(str(entry.get("path") or ""))
    detection = dict(entry.get("detection") or {})

    if architecture == UNKNOWN_ARCHITECTURE and path.is_dir():
        architecture, detected = detect_diffusers_architecture(path)
        detection = {**detected, **detection}

    defaults = defaults_for_architecture(architecture)
    defaults.update(dict(entry.get("defaults") or {}))

    return ModelDescriptor(
        name=name,
        path=str(path),
        format=str(entry.get("type") or entry.get("format") or "unknown"),
        source=str(entry.get("source") or "unknown"),
        architecture=architecture,
        backend=str(entry.get("backend") or backend_for_architecture(architecture)),
        capabilities=capabilities_for_architecture(architecture),
        defaults=defaults,
        constraints=dict(
            entry.get("constraints") or constraints_for_architecture(architecture)
        ),
        detection=detection,
    )
