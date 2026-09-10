"""Krea2 Identity Edit v1.2 quality/parity fixes.

This module is intentionally a thin patch layer over WebbDuck's existing Krea
worker so the large, hardware-tuned worker remains unchanged while we validate
quality against the upstream v1.2/v1.2.4 reference implementation.

The fixes here are model-math fixes, not performance tuning:
* correct VAE latent normalization: (z - mean) / std;
* use v1.2.4 pixel-space FIT geometry with /16 alignment and centered fractional
  RoPE offsets for aspect-ratio-mismatched references;
* require Qwen3-VL grounded conditioning instead of silently falling back to
  text-only identity conditioning;
* use the upstream identity recipe defaults (Turbo 10/0, Raw 20/3, ref_boost 4)
  when the request still carries ordinary Krea defaults; and
* leave post-hoc Real-ESRGAN disabled by default while parity is being debugged.
"""
from __future__ import annotations

import math
import os
from typing import Any

from PIL import Image

DEFAULT_REF_BOOST = 4.0
FIT_CROP_TOL = 0.08
_ACTIVE_FIT_MODE = "fit"


def _latent_stats(vae: Any, latents: Any) -> tuple[Any, Any]:
    import torch

    mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    std = (
        torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return mean, std


def normalize_krea_latents(latents: Any, vae: Any) -> Any:
    """Map VAE latents into Krea's normalized diffusion latent space."""
    mean, std = _latent_stats(vae, latents)
    return (latents - mean) / std


def denormalize_krea_latents(latents: Any, vae: Any) -> Any:
    """Inverse of :func:`normalize_krea_latents`."""
    mean, std = _latent_stats(vae, latents)
    return latents * std + mean


def edit_target_size_v124(
    source_size: tuple[int, int],
    requested_size: tuple[int, int] | None = None,
    *,
    fit_mode: str = "fit",
    max_megapixels: float = 2.0,
    multiple: int = 16,
) -> tuple[int, int]:
    """Return ``(height, width)`` for the TARGET grid.

    Krea2Edit v1.2 FIT no longer forces the output to the source image's aspect
    ratio. The target remains the requested composition; the reference is fit
    *inside that target coordinate system* separately before VAE encoding.
    """
    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source reference has invalid dimensions.")

    if requested_size is None:
        width, height = int(src_w), int(src_h)
    else:
        width, height = map(int, requested_size)
        if width <= 0 or height <= 0:
            raise ValueError("Requested edit dimensions must be positive.")

    cap = max(0.125, float(max_megapixels))
    mp = (width * height) / 1_000_000.0
    if mp > cap:
        scale = math.sqrt((cap * 1_000_000.0) / float(width * height))
        width = max(1, int(math.floor(width * scale)))
        height = max(1, int(math.floor(height * scale)))

    multiple = max(16, int(multiple))
    width = max(multiple, (width // multiple) * multiple)
    height = max(multiple, (height // multiple) * multiple)
    return height, width


def _crop_box(width: int, height: int, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    return x0, y0, x0 + crop_w, y0 + crop_h


def fit_reference_pixels_v124(
    source: Image.Image,
    *,
    target_width: int,
    target_height: int,
    fit_mode: str = "fit",
) -> tuple[Image.Image, dict[str, Any]]:
    """Prepare the reference image using current ComfyUI-Krea2Edit FIT geometry.

    ``fit`` performs AR-preserving pixel-space fit. Near-matched aspect ratios
    minimally center-crop to the exact target. Genuine mismatches fit inside,
    snap to the trainer's /16 grid, center-crop the source just enough to avoid
    resize squash, then place the resulting reference grid at a centered,
    possibly half-token RoPE offset inside the target grid.
    """
    image = source.convert("RGB")
    iw, ih = image.size
    if iw <= 0 or ih <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid reference/target dimensions for Krea identity FIT.")

    resampling = getattr(Image, "Resampling", Image)
    bicubic = resampling.BICUBIC
    fit_mode = str(fit_mode or "fit").strip().lower()

    if fit_mode == "fit":
        sc = min(target_height / float(ih), target_width / float(iw))
        if (
            ih * sc >= target_height * (1.0 - FIT_CROP_TOL)
            and iw * sc >= target_width * (1.0 - FIT_CROP_TOL)
        ):
            scale_fill = max(target_height / float(ih), target_width / float(iw))
            crop_h = min(ih, max(1, int(round(target_height / scale_fill))))
            crop_w = min(iw, max(1, int(round(target_width / scale_fill))))
            image = image.crop(_crop_box(iw, ih, crop_w, crop_h))
            ref_h, ref_w = target_height, target_width
        else:
            ref_h = min(
                max(16, int(ih * sc) // 16 * 16),
                max(16, target_height // 16 * 16),
            )
            ref_w = min(
                max(16, int(iw * sc) // 16 * 16),
                max(16, target_width // 16 * 16),
            )
            crop_h = min(ih, max(1, int(round(ref_h / sc))))
            crop_w = min(iw, max(1, int(round(ref_w / sc))))
            image = image.crop(_crop_box(iw, ih, crop_w, crop_h))
    else:
        scale_fill = max(target_height / float(ih), target_width / float(iw))
        crop_h = min(ih, max(1, int(round(target_height / scale_fill))))
        crop_w = min(iw, max(1, int(round(target_width / scale_fill))))
        image = image.crop(_crop_box(iw, ih, crop_w, crop_h))
        ref_h, ref_w = target_height, target_width

    if image.size != (ref_w, ref_h):
        image = image.resize((ref_w, ref_h), bicubic)

    target_gh, target_gw = target_height // 16, target_width // 16
    ref_gh, ref_gw = ref_h // 16, ref_w // 16
    off_h = max(0.0, (target_gh - ref_gh) / 2.0)
    off_w = max(0.0, (target_gw - ref_gw) / 2.0)
    geometry = {
        "fit_mode": fit_mode,
        "reference_pixels": [ref_w, ref_h],
        "target_pixels": [target_width, target_height],
        "reference_grid": [ref_gh, ref_gw],
        "target_grid": [target_gh, target_gw],
        "offset": [off_h, off_w],
    }
    return image, geometry


def edit_position_ids_v124(
    text_seq_len: int,
    target_grid_h: int,
    target_grid_w: int,
    source_grid_h: int,
    source_grid_w: int,
    source_offset_h: float,
    source_offset_w: float,
    device: Any,
) -> Any:
    """Build `[text | source(frame=1) | target(frame=0)]` v1.2.4 RoPE IDs."""
    import torch

    text_ids = torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32)

    src = torch.zeros(source_grid_h, source_grid_w, 3, device=device, dtype=torch.float32)
    src[..., 0] = 1.0
    src[..., 1] = (
        torch.arange(source_grid_h, device=device, dtype=torch.float32) + float(source_offset_h)
    )[:, None]
    src[..., 2] = (
        torch.arange(source_grid_w, device=device, dtype=torch.float32) + float(source_offset_w)
    )[None, :]

    tgt = torch.zeros(target_grid_h, target_grid_w, 3, device=device, dtype=torch.float32)
    tgt[..., 0] = 0.0
    tgt[..., 1] = torch.arange(target_grid_h, device=device, dtype=torch.float32)[:, None]
    tgt[..., 2] = torch.arange(target_grid_w, device=device, dtype=torch.float32)[None, :]

    return torch.cat(
        [text_ids, src.reshape(-1, 3), tgt.reshape(-1, 3)],
        dim=0,
    )


def _encode_identity_reference_v124(
    base: Any,
    pipe: Any,
    vae: Any,
    source: Image.Image,
    *,
    width: int,
    height: int,
    device: str,
    dtype: Any,
) -> Any:
    import torch

    fitted, geometry = fit_reference_pixels_v124(
        source,
        target_width=int(width),
        target_height=int(height),
        fit_mode=_ACTIVE_FIT_MODE,
    )
    ref_w, ref_h = geometry["reference_pixels"]
    px = pipe.image_processor.preprocess(fitted, height=ref_h, width=ref_w)
    vae.to(device)
    px = px.unsqueeze(2).to(device=device, dtype=vae.dtype)

    with torch.inference_mode():
        latent = vae.encode(px).latent_dist.mode()
        latent = normalize_krea_latents(latent, vae)
        latent = latent[:, :, 0]

    b, c, lh, lw = latent.shape
    packed = pipe._pack_latents(latent, b, c, lh, lw)

    patch_size = int(getattr(pipe, "patch_size", 2))
    src_gh, src_gw = max(1, lh // patch_size), max(1, lw // patch_size)
    target_gh, target_gw = base.grid_dims(width, height)
    geometry["reference_grid"] = [src_gh, src_gw]
    geometry["target_grid"] = [target_gh, target_gw]
    geometry["offset"] = [
        max(0.0, (target_gh - src_gh) / 2.0),
        max(0.0, (target_gw - src_gw) / 2.0),
    ]
    pipe._krea_identity_fit_geometry = geometry
    return packed.to(device="cpu", dtype=dtype)


def _decode_latents_v124(
    pipe: Any,
    vae: Any,
    latents: Any,
    *,
    width: int,
    height: int,
    device: str,
) -> list[Any]:
    import torch

    vae.to(device)
    latents = latents.to(device=device, dtype=vae.dtype)
    latents = pipe._unpack_latents(latents, height, width)
    latents = denormalize_krea_latents(latents.to(vae.dtype), vae)
    with torch.inference_mode():
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def _identity_geometry(pipe: Any, width: int, height: int) -> tuple[int, int, int, int, float, float]:
    target_gh, target_gw = height // 16, width // 16
    geometry = getattr(pipe, "_krea_identity_fit_geometry", None)
    if not isinstance(geometry, dict):
        return target_gh, target_gw, target_gh, target_gw, 0.0, 0.0
    src = geometry.get("reference_grid") or [target_gh, target_gw]
    offset = geometry.get("offset") or [0.0, 0.0]
    return (
        target_gh,
        target_gw,
        int(src[0]),
        int(src[1]),
        float(offset[0]),
        float(offset[1]),
    )


def _make_denoise_one_identity(base: Any):
    def _denoise_one_identity(
        pipe: Any,
        *,
        prompt_embeds: Any,
        prompt_mask: Any,
        negative_embeds: Any | None,
        negative_mask: Any | None,
        src_packed: Any,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        device: str,
        ref_boost: float,
        index: int,
        total_denoise_steps: int,
        report: Any,
        paired_block: bool = False,
        n_resident: int = 0,
    ) -> Any:
        import numpy as np
        import torch
        from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps
        from core.backends.krea2_identity import (
            edit_transformer_forward_nohooks,
            edit_transformer_forward_paired,
        )

        device_obj = torch.device(device)
        num_channels_latents = pipe.transformer.config.in_channels // (pipe.patch_size**2)
        generator = torch.Generator(device=device_obj).manual_seed(seed)
        latents = pipe.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device_obj,
            generator,
            None,
        )

        tgt_gh, tgt_gw, src_gh, src_gw, off_h, off_w = _identity_geometry(
            pipe, width, height
        )
        position_ids = edit_position_ids_v124(
            prompt_embeds.shape[1],
            tgt_gh,
            tgt_gw,
            src_gh,
            src_gw,
            off_h,
            off_w,
            device_obj,
        )
        neg_position_ids = None
        if negative_embeds is not None:
            neg_position_ids = edit_position_ids_v124(
                negative_embeds.shape[1],
                tgt_gh,
                tgt_gw,
                src_gh,
                src_gw,
                off_h,
                off_w,
                device_obj,
            )

        sigmas = np.linspace(1.0, 1 / int(steps), int(steps))
        timesteps, _num_steps = retrieve_timesteps(
            pipe.scheduler, int(steps), device_obj, sigmas=sigmas, mu=1.15
        )
        pipe.scheduler.set_begin_index(0)

        prompt_gpu = prompt_embeds.to(device_obj)
        mask_gpu = prompt_mask.to(device_obj)
        neg_gpu = negative_embeds.to(device_obj) if negative_embeds is not None else None
        neg_mask_gpu = negative_mask.to(device_obj) if negative_mask is not None else None
        src_gpu = src_packed.to(device_obj)

        try:
            with torch.inference_mode():
                for step_index, t in enumerate(timesteps):
                    timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(
                        latents.shape[0]
                    ).to(latents.dtype)

                    if paired_block and neg_gpu is not None:
                        out_pos, out_neg = edit_transformer_forward_paired(
                            pipe.transformer,
                            latents,
                            src_gpu,
                            prompt_gpu,
                            mask_gpu,
                            position_ids,
                            neg_gpu,
                            neg_mask_gpu,
                            neg_position_ids,
                            timestep,
                            ref_boost=ref_boost,
                            device=device,
                            n_resident=n_resident,
                        )
                        noise_pred = out_pos + float(guidance) * (out_pos - out_neg)
                        del out_pos, out_neg
                    elif paired_block:
                        noise_pred = edit_transformer_forward_nohooks(
                            pipe.transformer,
                            latents,
                            src_gpu,
                            prompt_gpu,
                            mask_gpu,
                            timestep,
                            position_ids,
                            ref_boost=ref_boost,
                            n_resident=n_resident,
                        )
                    else:
                        noise_pred = base.edit_transformer_forward(
                            pipe.transformer,
                            latents,
                            src_gpu,
                            prompt_gpu,
                            mask_gpu,
                            timestep,
                            position_ids,
                            ref_boost=ref_boost,
                        )
                        if neg_gpu is not None:
                            neg_pred = base.edit_transformer_forward(
                                pipe.transformer,
                                latents,
                                src_gpu,
                                neg_gpu,
                                neg_mask_gpu,
                                timestep,
                                neg_position_ids,
                                ref_boost=ref_boost,
                            )
                            noise_pred = noise_pred + float(guidance) * (noise_pred - neg_pred)
                            del neg_pred

                    latents = pipe.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]
                    completed = index * steps + step_index + 1
                    progress = 0.55 + (0.34 * completed / total_denoise_steps)
                    report(
                        "Editing with Krea 2 identity",
                        progress,
                        completed,
                        total_denoise_steps,
                    )
        finally:
            del prompt_gpu, mask_gpu, neg_gpu, neg_mask_gpu, src_gpu, generator
            if device == "cuda":
                torch.cuda.empty_cache()

        return latents.detach().to("cpu")

    return _denoise_one_identity


def _strict_grounded_encode(original: Any):
    def _encode(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        processor_source = result[-1] if isinstance(result, tuple) and result else None
        allow = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_ALLOW_TEXT_ONLY") or "").lower()
        if processor_source == "text-only" and allow not in {"1", "true", "yes", "on"}:
            from core.backends.krea2_identity import KreaIdentityError

            raise KreaIdentityError(
                "Krea identity requires Qwen3-VL grounded image+text conditioning, "
                "but AutoProcessor/Qwen3-VL was unavailable. Refusing the old "
                "text-only fallback because it does not reproduce Krea2Edit. "
                "Repair the Krea runtime/processor cache, or set "
                "WEBBDUCK_KREA2_IDENTITY_ALLOW_TEXT_ONLY=1 for debugging only."
            )
        return result

    return _encode


def _controlled_upscale(original: Any):
    def _maybe_upscale(image: Any, *, requested: Any, effective: Any):
        raw = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE") or "").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return image, None
        return original(image, requested=requested, effective=effective)

    return _maybe_upscale


def _quality_request(request: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(request)
    identity = dict(tuned.get("identity") or {})
    if not identity:
        return tuned

    variant = str(tuned.get("variant") or "base").lower()
    try:
        ref_boost = float(identity.get("ref_boost"))
    except (TypeError, ValueError):
        ref_boost = DEFAULT_REF_BOOST
    if "ref_boost" not in identity or abs(ref_boost - 2.0) < 1e-9:
        identity["ref_boost"] = DEFAULT_REF_BOOST

    if variant == "turbo":
        if tuned.get("steps") is None or int(tuned.get("steps") or 0) == 8:
            tuned["steps"] = 10
        if tuned.get("guidance") is None:
            tuned["guidance"] = 0.0
    else:
        if tuned.get("steps") is None or int(tuned.get("steps") or 0) == 28:
            tuned["steps"] = 20
        try:
            guidance = float(tuned.get("guidance"))
        except (TypeError, ValueError):
            guidance = 4.5
        if tuned.get("guidance") is None or abs(guidance - 4.5) < 1e-9:
            tuned["guidance"] = 3.0

    identity["quality_geometry"] = "v1.2.4-fit"
    tuned["identity"] = identity
    return tuned


def install_worker_quality_patch(base: Any) -> None:
    """Patch the loaded baseline Krea worker in-place."""
    global _ACTIVE_FIT_MODE

    if bool(getattr(base, "_krea_identity_quality_patch", False)):
        return

    from core.backends import krea2_identity as identity

    identity.DEFAULT_REF_BOOST = DEFAULT_REF_BOOST
    identity.edit_target_size = edit_target_size_v124
    base.edit_target_size = edit_target_size_v124

    original_prompt = base._encode_identity_prompt_phase
    original_run = base._run_identity
    original_upscale = base._maybe_upscale_identity_artifact

    base._normalize_krea_latents = normalize_krea_latents
    base._denormalize_krea_latents = denormalize_krea_latents
    base._fit_reference_pixels_v124 = fit_reference_pixels_v124
    base._identity_position_ids_v124 = edit_position_ids_v124
    base._encode_identity_reference = lambda pipe, vae, source, **kwargs: _encode_identity_reference_v124(
        base, pipe, vae, source, **kwargs
    )
    base._decode_latents = _decode_latents_v124
    base._encode_identity_prompt_phase = _strict_grounded_encode(original_prompt)
    base._denoise_one_identity = _make_denoise_one_identity(base)
    base._maybe_upscale_identity_artifact = _controlled_upscale(original_upscale)

    def _run_identity_quality(request: dict[str, Any], *args: Any, **kwargs: Any):
        global _ACTIVE_FIT_MODE
        tuned = _quality_request(request)
        identity_cfg = tuned.get("identity") or {}
        previous = _ACTIVE_FIT_MODE
        _ACTIVE_FIT_MODE = str(identity_cfg.get("fit_mode") or "fit").lower()
        try:
            return original_run(tuned, *args, **kwargs)
        finally:
            _ACTIVE_FIT_MODE = previous

    base._run_identity = _run_identity_quality
    base._krea_identity_quality_patch = True
