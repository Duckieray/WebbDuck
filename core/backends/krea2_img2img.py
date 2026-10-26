"""Latent-init img2img for Krea 2 with a flow-matching denoise strength.

``Krea2Pipeline`` publishes a text-to-image ``__call__`` only, but it already
accepts ``latents=`` and ``sigmas=``. That is the whole img2img primitive WebbDuck
needs: VAE-encode the source, corrupt it to the noise level implied by the
requested ``strength``, and run only the tail of the sigma schedule.

Two Krea-specific details drive the implementation:

1. ``Krea2Pipeline`` derives its own sigmas as ``linspace(1.0, 1/steps, steps)``
   and hands them to ``retrieve_timesteps(..., mu=mu)``. Its scheduler runs with
   ``use_dynamic_shifting=True``, so those sigmas are exponentially time-shifted
   once. The *effective* schedule the model actually denoises on is the shifted
   one, so that is the schedule ``strength`` must truncate.

2. Passing ``sigmas=`` back into ``__call__`` shifts them a **second** time. The
   truncated levels would be inflated toward 1.0, which nullifies ``strength``
   and re-renders the whole source. :func:`invert_krea_time_shift` pre-applies
   the inverse shift so the scheduler's own shift lands exactly on the intended
   schedule. (Same correction the FLUX img2img path needs for the same reason.)

``mu`` is 1.15 for distilled Turbo checkpoints and resolution-derived for Raw
checkpoints; both are honored so the truncated schedule matches what a plain
text-to-image run of the same shape would have denoised on.

This module is shared by the host and the isolated worker. Pure helpers live at
module scope so they stay importable (and testable) without torch; the worker
entry points import torch lazily inside the function bodies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Krea's scheduler config, used when the pipeline does not expose the attribute.
DISTILLED_SHIFT = 1.15
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 6400
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15

MIN_STRENGTH = 0.05


def normalize_strength(value: Any, default: float = 1.0) -> float:
    """Clamp a denoise strength into WebbDuck's usable ``(0, 1]`` range.

    ``1.0`` means "start from pure noise", i.e. ordinary text-to-image, which is
    why it is the default and why a caller that omits strength keeps today's
    behavior. Values are clamped rather than rejected so a stray slider value
    degrades to a sane denoise instead of failing the run.
    """
    raw = default if value is None else value
    try:
        strength = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(strength):
        return float(default)
    return max(MIN_STRENGTH, min(1.0, strength))


def resolve_img2img_request(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Build the worker ``img2img`` payload, or ``None`` for plain text-to-image.

    Mirrors the FLUX contract: an input image plus a sub-1.0 strength turns the
    source into the denoise init. A strength of 1.0 is text-to-image, so no
    payload is emitted and existing Krea requests are untouched.
    """
    raw_path = settings.get("image") or settings.get("input_image")
    image_path = str(raw_path or "").strip()
    if not image_path:
        return None

    strength = normalize_strength(settings.get("strength"), default=1.0)
    if strength >= 1.0:
        return None

    return {"input_image": image_path, "strength": strength}


def resolution_shift(
    image_seq_len: int,
    *,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    """Linear resolution-dependent shift, matching Krea's ``calculate_shift``.

    Verbatim copy of ``diffusers.pipelines.krea2.pipeline_krea2.calculate_shift``
    (itself copied from the FLUX pipeline). Inlined rather than imported because
    this module is also loaded by the WebbDuck host process, whose diffusers
    install is not required to carry the Krea pipeline.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def schedule_mu(pipe: Any, image_seq_len: int) -> float:
    """Return the ``mu`` Krea would use for this request shape.

    Distilled (Turbo) checkpoints pin ``mu=1.15``; Raw checkpoints derive it from
    the packed image sequence length. Mirrors ``Krea2Pipeline.__call__`` so an
    img2img run truncates the same schedule a text-to-image run would build.
    """
    if bool(getattr(getattr(pipe, "config", None), "is_distilled", False)):
        return DISTILLED_SHIFT

    config = getattr(getattr(pipe, "scheduler", None), "config", None)

    def _cfg(key: str, fallback: Any) -> Any:
        value = None
        if config is not None:
            try:
                value = config.get(key)
            except Exception:
                value = None
        return fallback if value is None else value

    return float(
        resolution_shift(
            int(image_seq_len),
            base_seq_len=int(_cfg("base_image_seq_len", BASE_IMAGE_SEQ_LEN)),
            max_seq_len=int(_cfg("max_image_seq_len", MAX_IMAGE_SEQ_LEN)),
            base_shift=float(_cfg("base_shift", BASE_SHIFT)),
            max_shift=float(_cfg("max_shift", MAX_SHIFT)),
        )
    )


def invert_krea_time_shift(sigmas: list[float], mu: float | None) -> list[float]:
    """Pre-apply the inverse of the scheduler's exponential time shift.

    ``FlowMatchEulerDiscreteScheduler`` with ``use_dynamic_shifting=True`` maps
    ``t -> exp(mu) / (exp(mu) + (1/t - 1))`` over any sigmas passed to
    ``set_timesteps(sigmas=...)``. The Krea pipe applies that shift to our
    truncated schedule, so the inverse is applied here first and the two cancel.
    0.0 and 1.0 are fixed points of the transform and pass through unchanged.
    """
    if mu is None:
        return [float(s) for s in sigmas]
    exp_mu = math.exp(float(mu))
    out: list[float] = []
    for value in sigmas:
        s = float(value)
        if s <= 0.0 or s >= 1.0:
            out.append(s)
            continue
        out.append(1.0 / (1.0 + exp_mu * (1.0 / s - 1.0)))
    return out


def img2img_sigma_schedule(
    steps: int,
    strength: float,
    full_sigmas: list[float],
) -> tuple[list[float], int, float]:
    """Truncate the effective (already shifted) schedule to ``strength``.

    Mirrors ``FluxImg2ImgPipeline.get_timesteps`` for flow matching: ``strength``
    keeps the noisiest ``strength`` fraction of the schedule, so ``1.0`` keeps all
    of it. Returns ``(truncated_sigmas, t_start, init_sigma)`` where
    ``init_sigma`` is the noise level used to corrupt the source latents.

    ``full_sigmas`` must be the scheduler's *effective* post-shift schedule with
    its appended terminal 0.0 removed; that trailing zero is re-appended by
    ``set_timesteps``, so keeping it here would shift the schedule length.
    """
    steps = max(1, int(steps))
    strength = normalize_strength(strength, default=1.0)
    full = [float(s) for s in full_sigmas]

    init_steps = min(int(steps * strength), steps)
    t_start = int(max(steps - init_steps, 0))
    t_start = min(t_start, max(0, len(full) - 1))

    truncated = full[t_start:]
    if not truncated:
        # Degenerate schedule: still run one step from clean latents rather than
        # handing the scheduler an empty sigma list.
        truncated = [0.0]
    return [float(s) for s in truncated], int(t_start), float(truncated[0])


def denoise_step_count(steps: int, strength: float) -> int:
    """Number of scheduler steps a ``strength``-truncated run will execute.

    Derived from :func:`img2img_sigma_schedule` so a plan or a UI estimate can
    never disagree with what the scheduler actually iterates.
    """
    truncated, _t_start, _init_sigma = img2img_sigma_schedule(steps, strength, [1.0] * max(1, int(steps)))
    return len(truncated)


def load_source_image(path_value: Any) -> Any:
    """Load an RGB PIL image without leaving a file handle open."""
    from PIL import Image

    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Krea 2 img2img input image does not exist: {path}")
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def full_shifted_sigmas(pipe: Any, steps: int, mu: float, device: Any) -> list[float]:
    """Reproduce the scheduler's effective schedule for ``steps``, minus the tail.

    Uses the pipeline's own ``linspace(1.0, 1/steps, steps)`` base sigmas and the
    pipeline's own ``mu`` so the result is byte-for-byte the schedule a plain
    text-to-image call would denoise on. ``scheduler.set_timesteps`` appends a
    terminal 0.0 that it re-appends for our truncated call too, so it is dropped.
    """
    import numpy as np
    from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

    steps = max(1, int(steps))
    sigmas = np.linspace(1.0, 1 / steps, steps)
    retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=float(mu))
    shifted = [float(s) for s in pipe.scheduler.sigmas.detach().cpu().tolist()]
    if shifted and shifted[-1] == 0.0:
        shifted = shifted[:-1]
    return shifted


def encode_source_latent(
    pipe: Any,
    vae: Any,
    source_image: Any,
    *,
    width: int,
    height: int,
    device: str,
) -> tuple[Any, int]:
    """VAE-encode the source to a clean, packed, normalized latent.

    Returns ``(clean_packed_latents_on_cpu, image_seq_len)``. The latent is packed
    exactly like the pipeline's target grid because ``prepare_latents`` passes a
    caller-supplied latent through untouched, and normalized with ``(z - mean) /
    std`` to match the space the transformer denoises in.
    """
    import torch

    patch_size = int(getattr(pipe, "patch_size", 2))
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    latent_h = max(1, int(height) // vae_scale_factor)
    latent_w = max(1, int(width) // vae_scale_factor)
    image_seq_len = max(1, latent_h // patch_size) * max(1, latent_w // patch_size)

    vae.to(device)
    px = pipe.image_processor.preprocess(source_image.convert("RGB"), height=height, width=width)
    px = px.unsqueeze(2).to(device=device, dtype=vae.dtype)  # (B, C, 1, H, W)
    with torch.inference_mode():
        latent = vae.encode(px).latent_dist.mode()  # (B, z, 1, lh, lw)
        mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(latent.device, latent.dtype)
        # Must mirror the pipelines' decode (`latents / latents_std + latents_mean`,
        # where that local `latents_std` is itself 1/config.latents_std), i.e. the
        # normalized space is `(latent - mean) / config.latents_std`.
        #
        # Dividing by `std` here instead inverted the scaling by config.latents_std
        # squared and handed the transformer a latent far outside the distribution it
        # denoises in. A face crop refined at low strength was then discarded and
        # re-rendered from scratch, which is what painted coloured discs onto faces.
        clean = (latent - mean) * std
        clean = clean[:, :, 0]  # (B, z, lh, lw)

    batch, channels, lh, lw = clean.shape
    packed = pipe._pack_latents(clean, batch, channels, lh, lw)
    del clean, latent, px
    return packed.to("cpu"), image_seq_len


def img2img_schedule(
    pipe: Any,
    *,
    steps: int,
    strength: float,
    image_seq_len: int,
    device: Any,
) -> tuple[list[float], float, int]:
    """Truncated schedule for a ``strength``, ready to hand to the pipeline.

    Returns ``(sigmas_for_pipe, init_sigma, denoise_steps)``. ``sigmas_for_pipe``
    is pre-inverted against the scheduler's time shift so the shift the pipeline
    applies lands on the intended levels.
    """
    import torch

    strength = normalize_strength(strength, default=1.0)
    mu = schedule_mu(pipe, image_seq_len)
    device_obj = torch.device(str(device))
    full = full_shifted_sigmas(pipe, steps, mu, device_obj)
    truncated, _t_start, init_sigma = img2img_sigma_schedule(steps, strength, full)
    return invert_krea_time_shift(truncated, mu), init_sigma, len(truncated)


def noise_init_latent(
    clean_packed: Any,
    init_sigma: float,
    *,
    seed: int,
    device: str,
) -> Any:
    """Corrupt a clean latent to the schedule's start noise level.

    Flow-matching interpolation ``x_t = (1 - sigma) * x0 + sigma * noise``, using
    the *post-shift* sigma the scheduler will actually denoise from.
    """
    import torch

    device_obj = torch.device(str(device))
    work = clean_packed.to(device=device_obj)
    generator = torch.Generator(device=device_obj).manual_seed(int(seed))
    noise = torch.randn(work.shape, generator=generator, device=device_obj, dtype=work.dtype)
    noised = work * (1.0 - float(init_sigma)) + noise * float(init_sigma)
    del noise, work
    return noised.to("cpu")


def release_vae(vae: Any) -> None:
    """Move the VAE off the accelerator and hand VRAM back to the denoise pass."""
    try:
        vae.to("cpu")
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def build_img2img_init(
    pipe: Any,
    vae: Any,
    source_image: Any,
    *,
    width: int,
    height: int,
    steps: int,
    strength: float,
    seed: int,
    device: str,
) -> tuple[Any, list[float], int]:
    """One-shot helper: encode, schedule, and noise for a single image.

    Equivalent to calling :func:`encode_source_latent`, :func:`img2img_schedule`,
    and :func:`noise_init_latent` in sequence. Returns
    ``(noised_packed_latents, sigmas_for_pipe, denoise_steps)``.

    The VAE encode runs while the pipeline VAE is still attached and is released
    to CPU afterwards, so the transformer denoise gets the full VRAM ceiling back
    instead of sharing it with a resident VAE.
    """
    clean, image_seq_len = encode_source_latent(
        pipe, vae, source_image, width=width, height=height, device=device
    )
    sigmas, init_sigma, denoise_steps = img2img_schedule(
        pipe,
        steps=steps,
        strength=strength,
        image_seq_len=image_seq_len,
        device=device,
    )
    noised = noise_init_latent(clean, init_sigma, seed=seed, device=device)
    del clean
    release_vae(vae)
    return noised, sigmas, denoise_steps
