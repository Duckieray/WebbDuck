"""Two-reference identity conditioning for Krea2Edit.

A single Persona reference is expanded internally into the layout Krea2Edit is
best at consuming for likeness:

* reference 1: the full, canonical Persona image (scene / global identity);
* reference 2: a derived face/head anchor (subject identity, last reference).

The full reference keeps the upstream/global ref_boost (normally 4x).  The
subject anchor gets a modestly stronger boost (normally 6x).  Both are encoded
as independent source-latent blocks with independent RoPE frames, rather than
trying to make one source latent do two jobs with a token mask.

When no face can be detected, this patch is a strict fallback to the existing
single-reference v1.2.4 path.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

_OFF_VALUES = {"0", "false", "off", "no", "none", "disabled"}
DEFAULT_SUBJECT_MAX_EDGE = 640
DEFAULT_SUBJECT_REF_BOOST = 6.0
DEFAULT_GROUNDED_FACE_PX = 1024

_ACTIVE_SUBJECT_PATH: Path | None = None
_ACTIVE_SOURCE_LAYOUT: list[dict[str, Any]] | None = None
_ACTIVE_SUBJECT_BOOST: float = DEFAULT_SUBJECT_REF_BOOST
_ACTIVE_META: dict[str, Any] = {}
_ORIGINAL_POSITION_IDS: Any | None = None
_ORIGINAL_REF_BOOST_BIAS: Any | None = None


def _lanczos() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def _enabled(identity_cfg: dict[str, Any]) -> bool:
    raw: Any = identity_cfg.get("multi_reference_identity")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_MULTIREF", "1")
    return str(raw).strip().lower() not in _OFF_VALUES


def _subject_max_edge(identity_cfg: dict[str, Any]) -> int:
    raw: Any = identity_cfg.get("subject_reference_max_edge")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_SUBJECT_EDGE", "")
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_SUBJECT_MAX_EDGE
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return DEFAULT_SUBJECT_MAX_EDGE
    return max(384, min(1024, value))


def _subject_boost(identity_cfg: dict[str, Any]) -> float:
    try:
        global_boost = float(identity_cfg.get("ref_boost", 4.0))
    except (TypeError, ValueError):
        global_boost = 4.0
    raw: Any = identity_cfg.get("subject_ref_boost")
    if raw is None:
        raw = os.getenv("WEBBDUCK_KREA2_IDENTITY_SUBJECT_REF_BOOST", "")
    try:
        explicit = float(raw) if str(raw or "").strip() else None
    except (TypeError, ValueError):
        explicit = None
    if global_boost <= 1.0001:
        return 1.0
    if explicit is not None:
        return max(global_boost, min(10.0, explicit))
    return min(10.0, max(DEFAULT_SUBJECT_REF_BOOST, global_boost + 2.0))


def _detect_face(image: Image.Image) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """Detect frontal or profile faces, including mirrored profile fallback."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None, None

    rgb = image.convert("RGB")
    ow, oh = rgb.size
    if min(ow, oh) < 32:
        return None, "opencv-haar-multiview"

    detect = rgb
    scale = 1.0
    if max(ow, oh) > 1280:
        scale = 1280.0 / float(max(ow, oh))
        detect = rgb.resize(
            (max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))),
            _lanczos(),
        )

    array = np.asarray(detect)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    dh, dw = gray.shape[:2]
    min_side = max(24, min(dh, dw) // 14)

    candidates: list[tuple[int, int, int, int, str]] = []

    def _run(name: str, filename: str, img: Any, *, flipped: bool = False) -> None:
        path = os.path.join(cv2.data.haarcascades, filename)
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return
        faces = cascade.detectMultiScale(
            img,
            scaleFactor=1.07,
            minNeighbors=4,
            minSize=(min_side, min_side),
        )
        for raw_face in faces:
            x, y, w, h = (int(v) for v in raw_face)
            if flipped:
                x = dw - (x + w)
            candidates.append((x, y, w, h, name))

    _run("frontal", "haarcascade_frontalface_default.xml", gray)
    _run("profile", "haarcascade_profileface.xml", gray)
    _run("profile-flipped", "haarcascade_profileface.xml", cv2.flip(gray, 1), flipped=True)

    if not candidates:
        return None, "opencv-haar-multiview"

    # Prefer the largest plausible face.  A dedicated subject reference should
    # normally contain one person, so area is a better identity signal than
    # left-to-right ordering.
    x, y, w, h, mode = max(candidates, key=lambda f: int(f[2]) * int(f[3]))
    inv = 1.0 / scale
    box = (
        max(0, int(round(x * inv))),
        max(0, int(round(y * inv))),
        max(1, int(round(w * inv))),
        max(1, int(round(h * inv))),
    )
    return box, f"opencv-haar-{mode}"


def _subject_crop_box(
    image_size: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Square head/shoulders crop with the face occupying about half the width."""
    width, height = image_size
    x, y, fw, fh = face_box
    fill = 0.52
    side = int(round(max(fw, fh) / fill))
    side = max(max(fw, fh), min(side, width, height))
    cx = x + fw / 2.0
    cy = y + fh / 2.0 + 0.12 * fh
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return left, top, left + side, top + side


def build_subject_anchor(
    image: Image.Image,
    face_box: tuple[int, int, int, int],
    *,
    max_edge: int = DEFAULT_SUBJECT_MAX_EDGE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Build the independent subject/face reference without upscaling it."""
    rgb = image.convert("RGB")
    crop_box = _subject_crop_box(rgb.size, face_box)
    subject = rgb.crop(crop_box)
    before = subject.size
    scale = 1.0
    if max(subject.size) > int(max_edge):
        scale = float(max_edge) / float(max(subject.size))
        subject = subject.resize(
            (
                max(1, int(round(subject.width * scale))),
                max(1, int(round(subject.height * scale))),
            ),
            _lanczos(),
        )
    return subject, {
        "crop_box": list(crop_box),
        "source_crop_size": list(before),
        "prepared_size": list(subject.size),
        "scale": round(scale, 6),
        "face_box_original": list(face_box),
    }


def _save_subject(subject: Image.Image) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix="webbduck_krea2_subject_ref_",
        suffix=".png",
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    subject.save(path, format="PNG")
    return path


def _snap16(value: int) -> int:
    return max(16, (max(16, int(value)) // 16) * 16)


def _encode_subject_reference(
    quality: Any,
    pipe: Any,
    vae: Any,
    subject: Image.Image,
    *,
    target_width: int,
    target_height: int,
    device: str,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """Encode the face anchor at its own resolution, centered in target RoPE."""
    import torch

    sw = _snap16(subject.width)
    sh = _snap16(subject.height)
    if subject.size != (sw, sh):
        subject = subject.resize((sw, sh), _lanczos())

    px = pipe.image_processor.preprocess(subject, height=sh, width=sw)
    vae.to(device)
    px = px.unsqueeze(2).to(device=device, dtype=vae.dtype)
    with torch.inference_mode():
        latent = vae.encode(px).latent_dist.mode()
        latent = quality.normalize_krea_latents(latent, vae)
        latent = latent[:, :, 0]

    batch, channels, lh, lw = latent.shape
    packed = pipe._pack_latents(latent, batch, channels, lh, lw)
    patch_size = int(getattr(pipe, "patch_size", 2))
    src_gh = max(1, lh // patch_size)
    src_gw = max(1, lw // patch_size)
    target_gh = max(1, int(target_height) // 16)
    target_gw = max(1, int(target_width) // 16)
    geometry = {
        "kind": "subject",
        "reference_pixels": [sw, sh],
        "reference_grid": [src_gh, src_gw],
        "target_grid": [target_gh, target_gw],
        "offset": [
            max(0.0, (target_gh - src_gh) / 2.0),
            max(0.0, (target_gw - src_gw) / 2.0),
        ],
    }
    return packed.to(device="cpu", dtype=dtype), geometry


def multiref_position_ids(
    text_seq_len: int,
    target_grid_h: int,
    target_grid_w: int,
    source_grid_h: int,
    source_grid_w: int,
    source_offset_h: float,
    source_offset_w: float,
    device: Any,
) -> Any:
    """Build [text | full(frame=1) | subject(frame=2) | target(frame=0)] IDs."""
    layout = _ACTIVE_SOURCE_LAYOUT
    if not layout:
        if _ORIGINAL_POSITION_IDS is None:
            raise RuntimeError("Krea multiref position-id fallback is unavailable")
        return _ORIGINAL_POSITION_IDS(
            text_seq_len,
            target_grid_h,
            target_grid_w,
            source_grid_h,
            source_grid_w,
            source_offset_h,
            source_offset_w,
            device,
        )

    import torch

    blocks = [torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32)]
    for frame, item in enumerate(layout, start=1):
        gh, gw = (int(v) for v in item["reference_grid"])
        off_h, off_w = (float(v) for v in item.get("offset", (0.0, 0.0)))
        ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
        ids[..., 0] = float(frame)
        ids[..., 1] = (
            torch.arange(gh, device=device, dtype=torch.float32) + off_h
        )[:, None]
        ids[..., 2] = (
            torch.arange(gw, device=device, dtype=torch.float32) + off_w
        )[None, :]
        blocks.append(ids.reshape(-1, 3))

    target = torch.zeros(
        int(target_grid_h), int(target_grid_w), 3, device=device, dtype=torch.float32
    )
    target[..., 1] = torch.arange(
        int(target_grid_h), device=device, dtype=torch.float32
    )[:, None]
    target[..., 2] = torch.arange(
        int(target_grid_w), device=device, dtype=torch.float32
    )[None, :]
    blocks.append(target.reshape(-1, 3))
    return torch.cat(blocks, dim=0)


def multiref_boost_bias(
    text_len: int,
    src_len: int,
    tgt_len: int,
    ref_boost: float,
    device: Any,
    dtype: Any,
) -> Any:
    """Keep full-reference likeness at ref_boost and strengthen only subject ref."""
    layout = _ACTIVE_SOURCE_LAYOUT
    if not layout or len(layout) < 2:
        if _ORIGINAL_REF_BOOST_BIAS is None:
            return None
        return _ORIGINAL_REF_BOOST_BIAS(
            text_len, src_len, tgt_len, ref_boost, device, dtype
        )

    lengths = [int(item.get("tokens") or 0) for item in layout]
    if sum(lengths) != int(src_len) or any(length <= 0 for length in lengths):
        if _ORIGINAL_REF_BOOST_BIAS is None:
            return None
        return _ORIGINAL_REF_BOOST_BIAS(
            text_len, src_len, tgt_len, ref_boost, device, dtype
        )

    import torch

    total = int(text_len) + int(src_len) + int(tgt_len)
    bias = torch.zeros(1, 1, total, total, device=device, dtype=dtype)
    target_start = int(text_len) + int(src_len)
    source_start = int(text_len)
    boosts = [max(float(ref_boost), 1e-4)] * len(lengths)
    boosts[-1] = max(float(_ACTIVE_SUBJECT_BOOST), boosts[-1])
    cursor = source_start
    for length, boost in zip(lengths, boosts):
        bias[:, :, target_start:, cursor : cursor + length] = math.log(boost)
        cursor += length
    return bias


def _install_encoder(base: Any, quality: Any) -> None:
    original_encode = base._encode_identity_reference

    def _encode_multiref(pipe: Any, vae: Any, source: Image.Image, **kwargs: Any):
        global _ACTIVE_SOURCE_LAYOUT, _ACTIVE_META

        full = original_encode(pipe, vae, source, **kwargs)
        full_geometry = copy.deepcopy(getattr(pipe, "_krea_identity_fit_geometry", {}))
        full_geometry["kind"] = "full"
        full_geometry["tokens"] = int(full.shape[1])
        _ACTIVE_SOURCE_LAYOUT = None

        if _ACTIVE_SUBJECT_PATH is None:
            return full

        try:
            with Image.open(_ACTIVE_SUBJECT_PATH) as raw_subject:
                raw_subject.load()
                subject = raw_subject.convert("RGB")
            subject_packed, subject_geometry = _encode_subject_reference(
                quality,
                pipe,
                vae,
                subject,
                target_width=int(kwargs.get("width")),
                target_height=int(kwargs.get("height")),
                device=str(kwargs.get("device")),
                dtype=kwargs.get("dtype"),
            )
            subject_geometry["tokens"] = int(subject_packed.shape[1])
        except Exception as exc:
            _ACTIVE_META["encode_error"] = f"{type(exc).__name__}: {exc}"
            pipe._krea_identity_fit_geometry = full_geometry
            return full

        _ACTIVE_SOURCE_LAYOUT = [full_geometry, subject_geometry]
        pipe._krea_identity_fit_geometry = full_geometry
        _ACTIVE_META.update(
            {
                "reference_count_used": 2,
                "full_tokens": int(full.shape[1]),
                "subject_tokens": int(subject_packed.shape[1]),
                "combined_source_tokens": int(full.shape[1] + subject_packed.shape[1]),
                "full_geometry": full_geometry,
                "subject_geometry": subject_geometry,
            }
        )
        return __import__("torch").cat([full, subject_packed], dim=1)

    base._encode_identity_reference = _encode_multiref


def install_identity_multiref_patch(base: Any) -> None:
    """Install the full-reference + subject-anchor identity architecture."""
    global _ORIGINAL_POSITION_IDS, _ORIGINAL_REF_BOOST_BIAS
    if bool(getattr(base, "_krea_identity_multiref_patch", False)):
        return

    from core.backends import krea2_identity as identity
    from core.backends import krea2_identity_quality as quality

    _ORIGINAL_POSITION_IDS = quality.edit_position_ids_v124
    _ORIGINAL_REF_BOOST_BIAS = identity.ref_boost_bias
    quality.edit_position_ids_v124 = multiref_position_ids
    identity.ref_boost_bias = multiref_boost_bias
    _install_encoder(base, quality)

    original_run = base._run_identity

    def _run_multiref(request: dict[str, Any], *args: Any, **kwargs: Any):
        global _ACTIVE_SUBJECT_PATH, _ACTIVE_SOURCE_LAYOUT
        global _ACTIVE_SUBJECT_BOOST, _ACTIVE_META

        tuned = dict(request)
        identity_cfg = tuned.get("identity")
        if not isinstance(identity_cfg, dict) or not _enabled(identity_cfg):
            return original_run(tuned, *args, **kwargs)

        identity_copy = dict(identity_cfg)
        reference_path = str(identity_copy.get("reference_image") or "").strip()
        temp_subject: Path | None = None
        _ACTIVE_SOURCE_LAYOUT = None
        _ACTIVE_SUBJECT_PATH = None
        _ACTIVE_SUBJECT_BOOST = _subject_boost(identity_copy)
        _ACTIVE_META = {
            "enabled": True,
            "reference_count_used": 1,
            "full_ref_boost": float(identity_copy.get("ref_boost") or 4.0),
            "subject_ref_boost": _ACTIVE_SUBJECT_BOOST,
        }

        try:
            if reference_path:
                with Image.open(reference_path) as raw:
                    raw.load()
                    source = raw.convert("RGB")
                face_box, detector = _detect_face(source)
                _ACTIVE_META["detector"] = detector
                _ACTIVE_META["face_box"] = list(face_box) if face_box is not None else None
                if face_box is not None:
                    subject, subject_meta = build_subject_anchor(
                        source,
                        face_box,
                        max_edge=_subject_max_edge(identity_copy),
                    )
                    temp_subject = _save_subject(subject)
                    _ACTIVE_SUBJECT_PATH = temp_subject
                    identity_copy["_subject_reference_image"] = str(temp_subject)
                    identity_copy["_subject_ref_boost"] = _ACTIVE_SUBJECT_BOOST
                    identity_copy["_multiref_identity"] = True
                    # Keep the primary reference full-frame.  The older face-crop
                    # wrapper is now redundant because the dedicated subject
                    # anchor is the high-information crop.
                    identity_copy["auto_face_crop"] = False
                    try:
                        current_grounding = int(identity_copy.get("grounding_px") or 768)
                    except (TypeError, ValueError):
                        current_grounding = 768
                    if current_grounding <= 768:
                        identity_copy["grounding_px"] = DEFAULT_GROUNDED_FACE_PX
                    _ACTIVE_META["subject_anchor"] = subject_meta
                    _ACTIVE_META["grounding_px"] = int(identity_copy.get("grounding_px") or 768)

            tuned["identity"] = identity_copy
            result = original_run(tuned, *args, **kwargs)
            if isinstance(result, dict):
                runtime = result.setdefault("runtime", {})
                if isinstance(runtime, dict):
                    ident = runtime.setdefault("identity", {})
                    if isinstance(ident, dict):
                        ident["multi_reference"] = copy.deepcopy(_ACTIVE_META)
                        if _ACTIVE_SOURCE_LAYOUT:
                            ident["reference_count_used"] = len(_ACTIVE_SOURCE_LAYOUT)
            return result
        finally:
            _ACTIVE_SUBJECT_PATH = None
            _ACTIVE_SOURCE_LAYOUT = None
            _ACTIVE_META = {}
            if temp_subject is not None:
                try:
                    temp_subject.unlink(missing_ok=True)
                except Exception:
                    pass

    base._run_identity = _run_multiref
    base._krea_multiref_position_ids = multiref_position_ids
    base._krea_multiref_boost_bias = multiref_boost_bias
    base._build_krea_subject_anchor = build_subject_anchor
    base._krea_identity_multiref_patch = True
