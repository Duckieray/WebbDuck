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


_IDENTITY_PROVIDER_BY_ARCHITECTURE = {
    "sdxl": "faceid_sdxl",
    # The generic ``flux`` descriptor is the backward-compatible FLUX.2 Klein
    # route in this build (see models.model_descriptor), so it owns native refs.
    "flux": "flux2_native",
    "flux2": "flux2_native",
    "krea2": "krea2_identity_edit",
}


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


def _identity_reference_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep only provider-neutral identity fields while changing architecture.

    Saved Personas are intentionally reusable across model families, but their
    stored provider-specific knobs are not.  When a preset created under SDXL
    is loaded while Krea is selected (or vice versa), retain the references and
    preset identity only; reconstruct provider-specific defaults below.
    """
    refs = cfg.get("reference_images")
    if refs is None:
        refs = cfg.get("refs")
    out: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", True)),
        "reference_images": list(refs) if isinstance(refs, (list, tuple)) else [],
    }
    preset_name = str(cfg.get("preset_name") or "").strip()
    if preset_name:
        out["preset_name"] = preset_name
    return out


def _provider_defaults(provider: str) -> dict[str, Any]:
    if provider == "krea2_identity_edit":
        return {
            "ref_boost": 4.0,
            "grounding_px": 768,
            "fit_mode": "fit",
            "lora_scale": 1.0,
            # Omit lora_rank intentionally. The Krea identity contract picks
            # r64/r128/full from live GPU VRAM when no explicit rank is supplied.
        }
    if provider == "flux2_native":
        return {
            "adapter_scale": 1.0,
            "lora_scale": 0.60,
            "face_crop": "auto",
            "flux2_anchor_dup": False,
            "face_focus": False,
        }
    return {
        "embedder": "buffalo_l",
        "adapter_scale": 1.0,
        "lora_scale": 0.60,
        "reference_mode": "primary_only",
    }


def _canonicalize_provider_fields(provider: str, cfg: dict[str, Any]) -> None:
    """Normalize provider-local sentinel values before backend validation."""
    if provider == "krea2_identity_edit":
        # The UI/preset schema historically used ``auto`` as the Krea rank
        # sentinel, while the identity backend expects either an explicit
        # concrete rank (full/r128/r64) or no field so it can choose based on
        # detected VRAM. Treat auto/default as absence rather than a rank name.
        rank = str(cfg.get("lora_rank") or "").strip().lower()
        if rank in {"", "auto", "default", "gpu", "gpu_auto"}:
            cfg.pop("lora_rank", None)


def _normalize_identity_adapter_for_descriptor(
    descriptor: ModelDescriptor,
    settings: dict[str, Any],
) -> None:
    """Enforce the one valid identity provider for the selected architecture.

    Identity provider selection is not user choice: it is part of the model
    runtime contract.  This backend guard protects API clients, old saved
    Personas, stale browser state, and future UI regressions from ever routing
    an SDXL FaceID payload into Krea or a Krea adapter into FLUX.2.
    """
    cfg = settings.get("identity_adapter")
    if not isinstance(cfg, dict) or not cfg or cfg.get("enabled") is False:
        return

    architecture = str(descriptor.architecture or "").strip().lower()
    required = _IDENTITY_PROVIDER_BY_ARCHITECTURE.get(architecture)
    if not required or not bool(descriptor.capabilities.identity_adapter):
        raise ValueError(
            f"Checkpoint '{descriptor.name}' does not expose a supported identity adapter."
        )

    current = str(cfg.get("type") or cfg.get("provider") or "").strip()
    if current == required:
        # Even correct payloads are canonicalized to one key so downstream
        # backends never have to arbitrate between `type` and `provider`.
        cfg["type"] = required
        cfg.pop("provider", None)
        _canonicalize_provider_fields(required, cfg)
        settings["identity_adapter"] = cfg
        return

    # Provider changed because the selected checkpoint architecture changed.
    # Do not carry incompatible tuning (e.g. SDXL LoRA 0.60 into Krea, where
    # the identity LoRA baseline is 1.0). References survive; knobs reset.
    normalized = _identity_reference_fields(cfg)
    normalized["type"] = required
    normalized.update(_provider_defaults(required))
    _canonicalize_provider_fields(required, normalized)
    settings["identity_adapter"] = normalized
    settings["identity_adapter_provider_corrected"] = {
        "from": current or None,
        "to": required,
        "architecture": architecture,
    }


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

    # Provider ownership is architecture-driven and enforced before backend
    # resolution. The UI also locks the selector, but this is the hard safety
    # boundary for stale presets and direct API clients.
    _normalize_identity_adapter_for_descriptor(descriptor, settings)

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
