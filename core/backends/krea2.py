"""Krea backend entry module with identity-quality defaults layered in."""
from __future__ import annotations

import os
import sys
from typing import Any

from core.backends import krea2_host_impl as _impl
from core.backends import krea2_identity as _identity

# Current upstream v1.2 likeness baseline. Functions defined in krea2_identity
# resolve this module global at call time, so changing it here fixes API/default
# requests without duplicating the full identity contract module.
_identity.DEFAULT_REF_BOOST = 4.0


def _host_total_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / (1024.0**3)
    except Exception:
        pass
    return None


def _quality_identity_worker_payload(settings: dict[str, Any]) -> dict[str, Any] | None:
    adapter_cfg = settings.get("identity_adapter")
    snapshot = _identity.identity_settings_snapshot(
        adapter_cfg,
        total_vram_gb=_host_total_vram_gb(),
    )
    if snapshot is None:
        return None

    # The branch previously shipped 2.0 as its Krea default; the current v1.2
    # reference baseline is ~4. Preserve genuinely custom values, but migrate
    # old/default-looking requests automatically.
    try:
        raw_boost = float((adapter_cfg or {}).get("ref_boost"))
    except (TypeError, ValueError):
        raw_boost = None
    if raw_boost is None or abs(raw_boost - 2.0) < 1e-9:
        snapshot.ref_boost = 4.0

    if str(os.getenv("WEBBDUCK_KREA2_IDENTITY_QUALITY_BASELINE") or "").lower() in {
        "1", "true", "yes", "on"
    }:
        snapshot.lora_rank = "full"

    token, _source = _impl._huggingface_token()

    def _download(repo_id: str, filename: str):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename, token=token or None)

    try:
        weight = _identity.resolve_identity_weight(
            rank=snapshot.lora_rank,
            hf_hub_download=_download,
        )
    except _identity.KreaIdentityError as exc:
        raise _identity.KreaIdentityError(
            f"Krea identity persona for the current request could not be resolved: {exc}"
        ) from exc

    return {
        "provider": "krea2_identity_edit",
        "weight_path": weight.path,
        "weight_source": weight.source,
        "reference_image": snapshot.reference_image,
        "reference_count_used": snapshot.reference_count_used,
        "ref_boost": snapshot.ref_boost,
        "grounding_px": snapshot.grounding_px,
        "fit_mode": snapshot.fit_mode,
        "lora_scale": snapshot.lora_scale,
        "lora_rank": snapshot.lora_rank,
        "max_megapixels": snapshot.max_megapixels,
        "face_crop": snapshot.face_crop,
        "token_budget": snapshot.token_budget,
        "quality_recipe": "krea2edit-v1.2.4",
        "grounded_required": True,
    }


def _identity_recipe_defaults(variant: str) -> tuple[int, float]:
    return (10, 0.0) if str(variant).lower() == "turbo" else (20, 3.0)


def _identity_enabled(settings: dict[str, Any]) -> bool:
    cfg = settings.get("identity_adapter")
    if not isinstance(cfg, dict) or not cfg or cfg.get("enabled") is False:
        return False
    provider = str(cfg.get("type") or cfg.get("provider") or "").strip()
    return provider in {"", "krea2_identity_edit"}


_original_generate = _impl.Krea2DiffusersBackend.generate


def _quality_generate(self: Any, descriptor: Any, settings: dict[str, Any], **kwargs: Any):
    if _identity_enabled(settings):
        variant = str(descriptor.detection.get("variant") or "base").lower()
        recommended_steps, recommended_cfg = _identity_recipe_defaults(variant)
        defaults = descriptor.defaults or {}

        current_steps = settings.get("steps")
        default_steps = defaults.get("steps")
        try:
            default_like_steps = (
                current_steps is None
                or default_steps is not None
                and int(current_steps) == int(default_steps)
            )
        except (TypeError, ValueError):
            default_like_steps = True
        if default_like_steps:
            settings["steps"] = recommended_steps

        current_cfg = settings.get("cfg")
        default_cfg = defaults.get("cfg")
        try:
            default_like_cfg = (
                current_cfg is None
                or default_cfg is not None
                and abs(float(current_cfg) - float(default_cfg)) < 1e-9
            )
        except (TypeError, ValueError):
            default_like_cfg = True
        if default_like_cfg:
            settings["cfg"] = recommended_cfg

        settings["krea_identity_recipe"] = {
            "source": "krea2edit-v1.2-baseline",
            "variant": variant,
            "steps": int(settings.get("steps") or recommended_steps),
            "guidance": float(settings.get("cfg") if settings.get("cfg") is not None else recommended_cfg),
            "ref_boost_default": 4.0,
            "grounding_px_default": 768,
        }

    return _original_generate(self, descriptor, settings, **kwargs)


_impl._identity_worker_payload = _quality_identity_worker_payload
_impl._identity_recipe_defaults = _identity_recipe_defaults
_impl.Krea2DiffusersBackend.generate = _quality_generate

# Keep monkeypatch paths and function globals coherent for the existing test
# suite: imports of core.backends.krea2 resolve to the patched baseline module.
sys.modules[__name__] = _impl
