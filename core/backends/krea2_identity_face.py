"""Face-aware identity conditioning for Krea2Edit.

This patch keeps WebbDuck's v1.2.4 Krea identity math intact while concentrating
reference attention where identity is most fragile: the face.  The reference
preparation layer supplies a face box in the canonical (possibly cropped /
downscaled) reference.  We transform a soft face mask through the same v1.2.4
FIT geometry as the source image, reduce it to source-token resolution, and use
it to blend a modest background reference boost with a stronger face boost.

No face detected -> byte-for-byte legacy ref_boost behavior.  This is deliberate:
face detection is an optional quality signal, never a requirement for Krea.
"""
from __future__ import annotations

import math
import os
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

_OFF_VALUES = {"0", "false", "off", "no", "none", "disabled"}

# Single-worker process state. Krea jobs are executed serially in the isolated
# worker, so keeping the active token mask here avoids threading new arguments
# through every upstream-compatible transformer forward signature.
_ACTIVE_FACE_TOKEN_WEIGHTS: list[float] | None = None
_ACTIVE_FACE_BOX: tuple[float, float, float, float] | None = None
_ACTIVE_FACE_BOOST: float | None = None
_ACTIVE_BACKGROUND_BOOST: float | None = None
_ACTIVE_FIT_MODE = "fit"
_ACTIVE_META: dict[str, Any] = {}
_ORIGINAL_REF_BOOST_BIAS: Any | None = None


def _enabled(identity_cfg: dict[str, Any]) -> bool:
    raw: Any = identity_cfg.get("face_focus")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_FACE_FOCUS", "auto")
    return str(raw).strip().lower() not in _OFF_VALUES


def _as_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def _boosts(identity_cfg: dict[str, Any]) -> tuple[float, float]:
    try:
        global_boost = float(identity_cfg.get("ref_boost", 4.0))
    except (TypeError, ValueError):
        global_boost = 4.0

    # 1.0 is the upstream "off" point; do not secretly turn identity back on.
    if global_boost <= 1.0001:
        return 1.0, 1.0

    try:
        background = float(identity_cfg.get("background_ref_boost"))
    except (TypeError, ValueError):
        background = min(2.0, global_boost)
    try:
        face = float(identity_cfg.get("face_ref_boost"))
    except (TypeError, ValueError):
        face = min(10.0, global_boost + 2.0)

    background = max(1.0, min(10.0, background))
    face = max(background, min(10.0, face))
    return background, face


def _soft_face_mask(
    size: tuple[int, int],
    face_box: tuple[float, float, float, float],
) -> Image.Image:
    """Create a feathered face/eyes/cheeks emphasis mask in source pixels."""
    width, height = size
    x, y, fw, fh = face_box

    # Generous enough to include hairline, cheeks, jaw and ears, but still much
    # tighter than the whole body/clothing/background.  The lower pad is larger
    # so chin/jaw identity is not clipped.
    left = max(0.0, x - 0.34 * fw)
    right = min(float(width), x + fw + 0.34 * fw)
    top = max(0.0, y - 0.34 * fh)
    bottom = min(float(height), y + fh + 0.42 * fh)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (int(round(left)), int(round(top)), int(round(right)), int(round(bottom))),
        fill=255,
    )
    blur = max(2.0, 0.12 * max(fw, fh))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))


def build_face_token_weights(
    source: Image.Image,
    face_box: tuple[float, float, float, float],
    *,
    target_width: int,
    target_height: int,
    source_grid_w: int,
    source_grid_h: int,
    fit_mode: str = "fit",
) -> list[float]:
    """Map a soft face mask through the exact v1.2.4 FIT path to source tokens."""
    from core.backends.krea2_identity_quality import fit_reference_pixels_v124

    mask = _soft_face_mask(source.size, face_box)
    rgb_mask = Image.merge("RGB", (mask, mask, mask))
    fitted, _geometry = fit_reference_pixels_v124(
        rgb_mask,
        target_width=int(target_width),
        target_height=int(target_height),
        fit_mode=fit_mode,
    )
    fitted_mask = fitted.convert("L")
    resampling = getattr(Image, "Resampling", Image)
    box_filter = getattr(resampling, "BOX", resampling.BILINEAR)
    token_mask = fitted_mask.resize(
        (max(1, int(source_grid_w)), max(1, int(source_grid_h))),
        box_filter,
    )
    return [float(v) / 255.0 for v in token_mask.getdata()]


def faceaware_ref_boost_bias(
    text_len: int,
    src_len: int,
    tgt_len: int,
    ref_boost: float,
    device: Any,
    dtype: Any,
) -> Any:
    """Blend background and face boosts across source keys for target queries."""
    global _ORIGINAL_REF_BOOST_BIAS
    weights = _ACTIVE_FACE_TOKEN_WEIGHTS
    if (
        not weights
        or len(weights) != int(src_len)
        or _ACTIVE_FACE_BOOST is None
        or _ACTIVE_BACKGROUND_BOOST is None
    ):
        if _ORIGINAL_REF_BOOST_BIAS is None:
            return None
        return _ORIGINAL_REF_BOOST_BIAS(
            text_len, src_len, tgt_len, ref_boost, device, dtype
        )

    import torch

    background = float(_ACTIVE_BACKGROUND_BOOST)
    face = float(_ACTIVE_FACE_BOOST)
    if background == 1.0 and face == 1.0:
        return None

    mask = torch.tensor(weights, device=device, dtype=torch.float32).clamp_(0.0, 1.0)
    boosts = background + mask * (face - background)
    boosts = boosts.clamp_min_(1e-4).log().to(dtype=dtype)

    total = int(text_len) + int(src_len) + int(tgt_len)
    bias = torch.zeros(1, 1, total, total, device=device, dtype=dtype)
    target_start = int(text_len) + int(src_len)
    source_bias = boosts.view(1, 1, 1, int(src_len)).expand(
        1, 1, int(tgt_len), int(src_len)
    )
    bias[:, :, target_start:, int(text_len):target_start] = source_bias
    return bias


def _install_encode_wrapper(base: Any) -> None:
    original_encode = base._encode_identity_reference

    def _encode_with_face_mask(pipe: Any, vae: Any, source: Image.Image, **kwargs: Any):
        global _ACTIVE_FACE_TOKEN_WEIGHTS, _ACTIVE_META

        packed = original_encode(pipe, vae, source, **kwargs)
        _ACTIVE_FACE_TOKEN_WEIGHTS = None

        if _ACTIVE_FACE_BOX is None:
            return packed
        geometry = getattr(pipe, "_krea_identity_fit_geometry", None)
        if not isinstance(geometry, dict):
            return packed
        source_grid = geometry.get("reference_grid")
        if not isinstance(source_grid, (list, tuple)) or len(source_grid) != 2:
            return packed

        try:
            source_grid_h = int(source_grid[0])
            source_grid_w = int(source_grid[1])
            weights = build_face_token_weights(
                source,
                _ACTIVE_FACE_BOX,
                target_width=int(kwargs.get("width")),
                target_height=int(kwargs.get("height")),
                source_grid_w=source_grid_w,
                source_grid_h=source_grid_h,
                fit_mode=_ACTIVE_FIT_MODE,
            )
        except Exception as exc:
            _ACTIVE_META["mask_error"] = f"{type(exc).__name__}: {exc}"
            return packed

        if len(weights) != int(packed.shape[1]):
            _ACTIVE_META["mask_error"] = (
                f"face token mask length {len(weights)} != packed source tokens {int(packed.shape[1])}"
            )
            return packed

        _ACTIVE_FACE_TOKEN_WEIGHTS = weights
        coverage = sum(weights) / max(1, len(weights))
        strong = sum(1 for value in weights if value >= 0.5) / max(1, len(weights))
        _ACTIVE_META.update(
            {
                "token_mask_tokens": len(weights),
                "token_mask_mean": round(float(coverage), 6),
                "token_mask_strong_fraction": round(float(strong), 6),
                "source_grid": [source_grid_h, source_grid_w],
            }
        )
        return packed

    base._encode_identity_reference = _encode_with_face_mask


def install_identity_face_patch(base: Any) -> None:
    """Install face-localized ref boost on the already-patched Krea worker."""
    global _ORIGINAL_REF_BOOST_BIAS
    if bool(getattr(base, "_krea_identity_face_patch", False)):
        return

    from core.backends import krea2_identity as identity

    _ORIGINAL_REF_BOOST_BIAS = identity.ref_boost_bias
    identity.ref_boost_bias = faceaware_ref_boost_bias
    _install_encode_wrapper(base)

    original_run = base._run_identity

    def _run_faceaware(request: dict[str, Any], *args: Any, **kwargs: Any):
        global _ACTIVE_FACE_TOKEN_WEIGHTS
        global _ACTIVE_FACE_BOX
        global _ACTIVE_FACE_BOOST
        global _ACTIVE_BACKGROUND_BOOST
        global _ACTIVE_FIT_MODE
        global _ACTIVE_META

        identity_cfg = request.get("identity")
        identity_cfg = identity_cfg if isinstance(identity_cfg, dict) else {}
        enabled = _enabled(identity_cfg)
        face_box = _as_box(identity_cfg.get("_prepared_face_box")) if enabled else None
        background, face = _boosts(identity_cfg)

        _ACTIVE_FACE_TOKEN_WEIGHTS = None
        _ACTIVE_FACE_BOX = face_box
        _ACTIVE_BACKGROUND_BOOST = background
        _ACTIVE_FACE_BOOST = face
        _ACTIVE_FIT_MODE = str(identity_cfg.get("fit_mode") or "fit").strip().lower()
        _ACTIVE_META = {
            "enabled": bool(enabled),
            "face_detected": face_box is not None,
            "face_box": list(face_box) if face_box is not None else None,
            "detector": identity_cfg.get("_face_detector"),
            "background_ref_boost": round(background, 4),
            "face_ref_boost": round(face, 4),
        }

        try:
            result = original_run(request, *args, **kwargs)
            if isinstance(result, dict):
                runtime = result.setdefault("runtime", {})
                if isinstance(runtime, dict):
                    ident = runtime.setdefault("identity", {})
                    if isinstance(ident, dict):
                        ident["face_focus"] = dict(_ACTIVE_META)
            return result
        finally:
            _ACTIVE_FACE_TOKEN_WEIGHTS = None
            _ACTIVE_FACE_BOX = None
            _ACTIVE_FACE_BOOST = None
            _ACTIVE_BACKGROUND_BOOST = None
            _ACTIVE_META = {}

    base._run_identity = _run_faceaware
    base._faceaware_ref_boost_bias = faceaware_ref_boost_bias
    base._build_face_token_weights = build_face_token_weights
    base._krea_identity_face_patch = True
