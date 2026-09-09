"""Reference preparation and final-size policy for Krea 2 identity runs.

Krea2Edit behaves more consistently when very large persona references are
normalized before either conditioning path sees them. This module therefore
provides one canonical, aspect-ratio-preserving reference for both the grounded
Qwen3-VL encode and the VAE appearance path.

The denoiser can still render below the requested output size when the adaptive
planner protects a constrained GPU. That is a *native inference* resolution,
not necessarily the desired artifact size. The final-size wrapper restores the
requested dimensions after decode without forcing the denoiser to exceed its
safe token budget.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_REFERENCE_MAX_EDGE = 1024
_OFF_VALUES = {"0", "false", "off", "no", "none", "disabled"}
_REAL_ESRGAN_VALUES = {"1", "true", "yes", "on", "auto", "realesrgan", "real-esrgan"}
_LEGACY_REAL_ESRGAN_VALUES = {"1", "true", "yes", "on"}


def _lanczos() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def reference_max_edge(identity_cfg: dict[str, Any] | None = None) -> int:
    """Return the canonical reference longest-edge cap.

    Request-level ``reference_max_edge`` wins when present. Otherwise
    ``WEBBDUCK_KREA2_IDENTITY_MAX_REFERENCE_EDGE`` may override the 1024 px
    default. A false/off/zero value disables normalization entirely.
    """
    cfg = identity_cfg if isinstance(identity_cfg, dict) else {}
    raw: Any = cfg.get("reference_max_edge")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_MAX_REFERENCE_EDGE", "")
    text = str(raw or "").strip().lower()
    if text in _OFF_VALUES:
        return 0
    if not text:
        return DEFAULT_REFERENCE_MAX_EDGE
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return DEFAULT_REFERENCE_MAX_EDGE
    return max(256, value)


def prepare_identity_reference(
    image: Image.Image,
    *,
    max_long_edge: int = DEFAULT_REFERENCE_MAX_EDGE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return a canonical RGB identity reference and preparation telemetry.

    Large images are downscaled exactly once, before Qwen and VAE conditioning.
    Smaller images are never upscaled. Aspect ratio is always preserved.
    """
    prepared = image.convert("RGB")
    original_w, original_h = prepared.size
    max_long_edge = int(max_long_edge or 0)
    longest = max(original_w, original_h)

    meta: dict[str, Any] = {
        "original_size": [original_w, original_h],
        "prepared_size": [original_w, original_h],
        "max_reference_edge": max_long_edge,
        "downscaled": False,
        "scale": 1.0,
    }
    if max_long_edge <= 0 or longest <= max_long_edge:
        return prepared, meta

    scale = float(max_long_edge) / float(longest)
    new_w = max(1, int(round(original_w * scale)))
    new_h = max(1, int(round(original_h * scale)))
    prepared = prepared.resize((new_w, new_h), _lanczos())
    meta.update(
        {
            "prepared_size": [new_w, new_h],
            "downscaled": True,
            "scale": round(scale, 6),
        }
    )
    return prepared, meta


def _prepare_request_reference(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, Path | None]:
    """Swap an oversized identity reference for a temporary canonical PNG."""
    tuned = dict(request)
    identity_cfg = tuned.get("identity")
    if not isinstance(identity_cfg, dict):
        return tuned, None, None

    identity = dict(identity_cfg)
    raw_path = str(identity.get("reference_image") or "").strip()
    if not raw_path:
        return tuned, None, None

    max_edge = reference_max_edge(identity)
    try:
        with Image.open(raw_path) as source:
            source.load()
            prepared, meta = prepare_identity_reference(source, max_long_edge=max_edge)
    except Exception:
        # Let the baseline worker raise its existing actionable reference error.
        return tuned, None, None

    if not meta.get("downscaled"):
        return tuned, meta, None

    handle = tempfile.NamedTemporaryFile(
        prefix="webbduck_krea2_identity_ref_",
        suffix=".png",
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        prepared.save(temp_path, format="PNG")
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return tuned, meta, None

    identity["reference_image"] = str(temp_path)
    tuned["identity"] = identity
    return tuned, meta, temp_path


def _attach_reference_meta(result: Any, meta: dict[str, Any] | None) -> None:
    if not isinstance(result, dict) or not isinstance(meta, dict):
        return
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        return
    identity = runtime.setdefault("identity", {})
    if isinstance(identity, dict):
        identity["reference_preprocess"] = dict(meta)


def _final_size_upscaler(original: Any):
    """Restore requested artifact dimensions after a safe lower-res denoise.

    Default policy is exact-size LANCZOS because it cannot hallucinate or alter
    facial anatomy. Set ``WEBBDUCK_KREA2_IDENTITY_UPSCALE=realesrgan`` (or 1)
    to use WebbDuck's existing Real-ESRGAN path. Set it to 0/off to retain the
    native denoise size for debugging.
    """

    def _call_realesrgan(image: Any, *, requested: Any, effective: Any, raw: str):
        # The earlier quality wrapper only recognizes the legacy truthy spellings.
        # Normalize newer explicit names ("realesrgan"/"auto") while delegating.
        key = "WEBBDUCK_KREA2_IDENTITY_UPSCALE"
        previous = os.environ.get(key)
        if raw not in _LEGACY_REAL_ESRGAN_VALUES:
            os.environ[key] = "1"
        try:
            return original(image, requested=requested, effective=effective)
        finally:
            if raw not in _LEGACY_REAL_ESRGAN_VALUES:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def _maybe_upscale(image: Any, *, requested: Any, effective: Any):
        raw = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE") or "").strip().lower()
        if raw in _OFF_VALUES:
            return image, None
        if raw in _REAL_ESRGAN_VALUES:
            return _call_realesrgan(
                image,
                requested=requested,
                effective=effective,
                raw=raw,
            )

        if requested is None:
            return image, None
        try:
            req_w, req_h = int(requested[0]), int(requested[1])
        except Exception:
            return image, None
        if req_w <= 0 or req_h <= 0:
            return image, None

        current_w, current_h = image.size
        if (current_w, current_h) == (req_w, req_h):
            return image, None

        image = image.resize((req_w, req_h), _lanczos())
        note = {
            "from": [current_w, current_h],
            "to": [req_w, req_h],
            "upscaler": "lanczos",
            "policy": "exact-requested-size",
            "native_effective_size": list(effective) if effective is not None else [current_w, current_h],
        }
        return image, note

    return _maybe_upscale


def install_identity_reference_patch(base: Any) -> None:
    """Install canonical reference prep and final-size restoration in-place."""
    if bool(getattr(base, "_krea_identity_reference_patch", False)):
        return

    original_run = base._run_identity
    original_upscale = base._maybe_upscale_identity_artifact

    def _run_with_prepared_reference(request: dict[str, Any], *args: Any, **kwargs: Any):
        tuned, meta, temp_path = _prepare_request_reference(request)
        try:
            result = original_run(tuned, *args, **kwargs)
            _attach_reference_meta(result, meta)
            return result
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    base._run_identity = _run_with_prepared_reference
    base._maybe_upscale_identity_artifact = _final_size_upscaler(original_upscale)
    base._prepare_identity_reference = prepare_identity_reference
    base._krea_identity_reference_patch = True
