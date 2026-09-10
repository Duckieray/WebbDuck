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


def _host_vram_gb() -> tuple[float | None, float | None]:
    """Return (total, free) CUDA VRAM in GiB when the host can query it."""
    try:
        import torch

        if torch.cuda.is_available():
            total = float(torch.cuda.get_device_properties(0).total_memory) / (1024.0**3)
            free: float | None = None
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                free = float(free_bytes) / (1024.0**3)
            except Exception:
                pass
            return total, free
    except Exception:
        pass
    return None, None


def _host_total_vram_gb() -> float | None:
    return _host_vram_gb()[0]


def _recommended_identity_token_budget(
    total_vram_gb: float | None,
    free_vram_gb: float | None,
    width: int | None = None,
    height: int | None = None,
) -> int | None:
    """Choose a native Krea identity target-token budget from live headroom.

    Healthy ~16 GB cards now get a 3072-token high-detail tier for square /
    portrait-oriented identity work, where facial anatomy benefits most from
    native pixels.  The previously validated 2688 tier remains the normal
    healthy-card fallback, then 2048/1792 as desktop VRAM pressure rises.

    Explicit request-level ``identity.token_budget`` and
    ``WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET`` still win.
    """
    if total_vram_gb is None or total_vram_gb <= 0:
        return None

    if free_vram_gb is None or free_vram_gb <= 0:
        if total_vram_gb >= 15.0:
            return 2048
        return 1792 if total_vram_gb < 12.0 else 2048

    occupied_gb = max(0.0, total_vram_gb - free_vram_gb)
    free_fraction = free_vram_gb / total_vram_gb
    try:
        req_w = int(width or 0)
        req_h = int(height or 0)
    except (TypeError, ValueError):
        req_w = req_h = 0
    # Square and vertical outputs are the common portrait/persona compositions.
    portraitish = req_w > 0 and req_h > 0 and req_h >= int(req_w * 0.90)

    if total_vram_gb >= 15.0:
        # 3072 is intentionally gated more tightly than 2688 because the edit
        # sequence includes both source and target image tokens.
        if (
            portraitish
            and free_vram_gb >= 12.5
            and occupied_gb < 2.75
            and free_fraction >= 0.80
        ):
            return 3072
        if free_vram_gb >= 11.5 and occupied_gb < 3.0 and free_fraction >= 0.74:
            return 2688
        if free_vram_gb >= 9.5 and occupied_gb < 5.5 and free_fraction >= 0.60:
            return 2048
        return 1792

    if total_vram_gb >= 12.0:
        return 2048 if free_vram_gb >= 8.0 else 1792

    return 1792


def _quality_identity_worker_payload(settings: dict[str, Any]) -> dict[str, Any] | None:
    adapter_cfg = settings.get("identity_adapter")
    total_vram_gb, free_vram_gb = _host_vram_gb()
    snapshot = _identity.identity_settings_snapshot(
        adapter_cfg,
        total_vram_gb=total_vram_gb,
    )
    if snapshot is None:
        return None

    adapter = adapter_cfg if isinstance(adapter_cfg, dict) else {}

    # The branch previously shipped 2.0 as its Krea default; the current v1.2
    # reference baseline is ~4. Preserve genuinely custom values, but migrate
    # old/default-looking requests automatically.
    try:
        raw_boost = float(adapter.get("ref_boost"))
    except (TypeError, ValueError):
        raw_boost = None
    if raw_boost is None or abs(raw_boost - 2.0) < 1e-9:
        snapshot.ref_boost = 4.0

    if str(os.getenv("WEBBDUCK_KREA2_IDENTITY_QUALITY_BASELINE") or "").lower() in {
        "1", "true", "yes", "on"
    }:
        snapshot.lora_rank = "full"

    env_budget = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET") or "").strip().lower()
    if snapshot.token_budget is None and env_budget in {"", "auto", "-1"}:
        snapshot.token_budget = _recommended_identity_token_budget(
            total_vram_gb,
            free_vram_gb,
            width=int(settings.get("width") or 0),
            height=int(settings.get("height") or 0),
        )

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
        # Face-aware Krea settings are intentionally provider-specific but stay
        # optional so old callers need no changes.
        "reference_max_edge": adapter.get("reference_max_edge"),
        "auto_face_crop": adapter.get("auto_face_crop"),
        "face_focus": adapter.get("face_focus"),
        "face_ref_boost": adapter.get("face_ref_boost"),
        "background_ref_boost": adapter.get("background_ref_boost"),
        "quality_recipe": "krea2edit-v1.2.4-face-aware",
        "grounded_required": True,
    }


def _identity_recipe_defaults(variant: str) -> tuple[int, float]:
    # Krea2Edit's Turbo range is roughly 8-12 steps. WebbDuck biases the default
    # to the face-detail end of that range rather than the speed end.
    return (12, 0.0) if str(variant).lower() == "turbo" else (20, 3.0)


def _identity_enabled(settings: dict[str, Any]) -> bool:
    cfg = settings.get("identity_adapter")
    if not isinstance(cfg, dict) or not cfg or cfg.get("enabled") is False:
        return False
    provider = str(cfg.get("type") or cfg.get("provider") or "").strip()
    return provider in {"", "krea2_identity_edit"}


_original_generate = _impl.Krea2DiffusersBackend.generate


def _quality_generate(self: Any, descriptor: Any, settings: dict[str, Any], **kwargs: Any):
    identity_active = _identity_enabled(settings)
    if identity_active:
        variant = str(descriptor.detection.get("variant") or "base").lower()
        recommended_steps, recommended_cfg = _identity_recipe_defaults(variant)
        defaults = descriptor.defaults or {}

        current_steps = settings.get("steps")
        default_steps = defaults.get("steps")
        try:
            current_steps_int = int(current_steps) if current_steps is not None else None
            default_like_steps = (
                current_steps is None
                or default_steps is not None
                and current_steps_int == int(default_steps)
                or variant == "turbo"
                and current_steps_int == 10
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
            "source": "krea2edit-v1.2-face-aware",
            "variant": variant,
            "steps": int(settings.get("steps") or recommended_steps),
            "guidance": float(settings.get("cfg") if settings.get("cfg") is not None else recommended_cfg),
            "ref_boost_default": 4.0,
            "face_ref_boost_default": 6.0,
            "background_ref_boost_default": 2.0,
            "grounding_px_default": 768,
            "reference_max_edge_default": 1024,
            "auto_face_crop_default": True,
            "final_size_policy": "exact-requested-size",
        }

    result = _original_generate(self, descriptor, settings, **kwargs)

    # The adaptive worker records native denoise dimensions separately.  The
    # returned identity artifact is restored to requested dimensions after
    # decode, so saved metadata should describe both native and final sizes.
    if identity_active and isinstance(result, tuple) and len(result) == 2:
        images, _seed = result
        runtime = settings.get("krea_runtime")
        if isinstance(runtime, dict):
            identity_runtime = runtime.get("identity")
            if isinstance(identity_runtime, dict):
                try:
                    eff_w = int(identity_runtime.get("effective_width") or 0)
                    eff_h = int(identity_runtime.get("effective_height") or 0)
                except (TypeError, ValueError):
                    eff_w = eff_h = 0
                if eff_w > 0 and eff_h > 0:
                    settings["krea_effective_width"] = eff_w
                    settings["krea_effective_height"] = eff_h
        if images:
            try:
                final_w, final_h = images[0].size
                settings["width"] = int(final_w)
                settings["height"] = int(final_h)
                settings["krea_final_width"] = int(final_w)
                settings["krea_final_height"] = int(final_h)
                eff_w = int(settings.get("krea_effective_width") or final_w)
                eff_h = int(settings.get("krea_effective_height") or final_h)
                settings["krea_final_upscaled"] = (eff_w, eff_h) != (final_w, final_h)
            except Exception:
                pass

    return result


_impl._identity_worker_payload = _quality_identity_worker_payload
_impl._identity_recipe_defaults = _identity_recipe_defaults
_impl._recommended_identity_token_budget = _recommended_identity_token_budget
_impl._host_vram_gb = _host_vram_gb
_impl.Krea2DiffusersBackend.generate = _quality_generate

# Keep monkeypatch paths and function globals coherent for the existing test
# suite: imports of core.backends.krea2 resolve to the patched baseline module.
sys.modules[__name__] = _impl
