"""Optional 2x "realism" VAE decoder for Krea 2.

Krea 2 decodes with the stock Qwen-Image VAE (``AutoencoderKLQwenImage``).
``spacepxl/Wan2.1-VAE-upscale2x`` is a decoder-only finetune of the Wan2.1 VAE
that shares the exact same 16-channel latent space and per-channel
normalization. Its final conv emits 12 channels that pixel-shuffle into an RGB
image at 2x resolution, for roughly the same decode cost. It was trained on real
photos with degraded-latent robustness, which noticeably improves skin and hair
texture compared with the stock decoder.

This module is shared by the host and the isolated worker:

* host: :func:`resolve_decoder_request` builds the worker payload and
  :func:`fit_realism_output` sizes the 2x decode to the final artifact;
* worker: :func:`install_worker_realism_patch` wraps ``_decode_latents`` and
  :func:`realism_decode_context` activates it per request.

Any decoder load/decode failure falls back to the stock VAE and is recorded in
the run note rather than failing the generation.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

DEFAULT_REPO = "spacepxl/Wan2.1-VAE-upscale2x"
DEFAULT_SUBFOLDER = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1"

DECODER_STOCK = "stock"
DECODER_REALISM = "realism"
OUTPUT_REQUESTED = "requested"
OUTPUT_2X = "2x"

_REALISM_ALIASES = {"realism", "realism-2x", "realism2x", "upscale2x", "wan2x", "2x"}
_OUTPUT_2X_ALIASES = {"2x", "native-2x", "native2x", "keep", "full"}
_ROOT_SUBFOLDER_VALUES = {"", ".", "none", "root"}


def normalize_decoder_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return DECODER_REALISM if raw in _REALISM_ALIASES else DECODER_STOCK


def normalize_output_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return OUTPUT_2X if raw in _OUTPUT_2X_ALIASES else OUTPUT_REQUESTED


def decoder_source() -> str:
    return str(os.getenv("WEBBDUCK_KREA2_REALISM_DECODER") or "").strip() or DEFAULT_REPO


def decoder_subfolder(source: str | None = None) -> str | None:
    """Return the diffusers subfolder for the decoder source (``None`` = root)."""
    explicit = os.getenv("WEBBDUCK_KREA2_REALISM_DECODER_SUBFOLDER")
    if explicit is not None:
        value = explicit.strip()
        return None if value.lower() in _ROOT_SUBFOLDER_VALUES else value
    source = source or decoder_source()
    if source == DEFAULT_REPO:
        return DEFAULT_SUBFOLDER
    local = Path(source).expanduser()
    if local.is_dir() and not (local / "config.json").is_file():
        nested = local / DEFAULT_SUBFOLDER
        if (nested / "config.json").is_file():
            return DEFAULT_SUBFOLDER
    return None


def resolve_decoder_request(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Build the worker ``decoder`` payload, or ``None`` for the stock VAE.

    The per-request ``vae_decoder`` field wins; an empty value falls back to
    ``WEBBDUCK_KREA2_DECODER`` so deployments can opt in globally.
    """
    requested = settings.get("vae_decoder")
    if requested is None or str(requested).strip() == "":
        requested = os.getenv("WEBBDUCK_KREA2_DECODER")
    if normalize_decoder_mode(requested) != DECODER_REALISM:
        return None

    output = settings.get("vae_decoder_output")
    if output is None or str(output).strip() == "":
        output = os.getenv("WEBBDUCK_KREA2_DECODER_OUTPUT")

    source = decoder_source()
    return {
        "mode": DECODER_REALISM,
        "output": normalize_output_mode(output),
        "source": source,
        "subfolder": decoder_subfolder(source),
    }


def fit_realism_output(
    image: Image.Image,
    *,
    requested: tuple[int, int] | None,
    output_mode: str,
) -> tuple[Image.Image, str | None]:
    """Size a 2x realism decode for saving.

    ``requested`` mode resizes (normally an anti-aliased downsample) to the
    exact requested dimensions; ``2x`` keeps the full decoder resolution.
    Returns ``(image, resize_filter_or_None)``.
    """
    if normalize_output_mode(output_mode) == OUTPUT_2X or requested is None:
        return image, None
    req_w, req_h = (int(requested[0] or 0), int(requested[1] or 0))
    if req_w <= 0 or req_h <= 0 or image.size == (req_w, req_h):
        return image, None
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((req_w, req_h), resampling.LANCZOS), "lanczos"


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

_ACTIVE: dict[str, Any] | None = None
_NOTE: dict[str, Any] | None = None
_DECODER: Any = None


def pixel_shuffle_first_frame(decoded: Any) -> Any:
    """Convert decoder output ``[B, C, F, H, W]`` to an image batch ``[B, 3, H', W']``.

    The 2x decoder emits 12 channels (3 RGB x 2x2 sub-pixels); a stock-shaped
    3-channel output passes through unchanged.
    """
    import torch.nn.functional as F

    frame = decoded[:, :, 0] if decoded.ndim == 5 else decoded
    channels = int(frame.shape[1])
    if channels == 3:
        return frame
    if channels % 3 != 0:
        raise ValueError(f"Unexpected realism decoder channel count: {channels}")
    factor = int(round((channels // 3) ** 0.5))
    if factor * factor * 3 != channels:
        raise ValueError(f"Realism decoder channels {channels} are not a pixel-shuffle layout")
    return F.pixel_shuffle(frame, factor)


def _load_decoder(config: dict[str, Any], dtype: Any, device: str) -> Any:
    from diffusers import AutoencoderKLWan

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    subfolder = config.get("subfolder")
    if subfolder:
        kwargs["subfolder"] = subfolder
    token = str(os.getenv("HF_TOKEN") or "").strip()
    if token and not Path(str(config["source"])).expanduser().exists():
        kwargs["token"] = token
    decoder = AutoencoderKLWan.from_pretrained(str(config["source"]), **kwargs)
    decoder.eval()
    # Only the decoder half is used; dropping the encoder saves VRAM.
    try:
        decoder.encoder = None
    except Exception:
        pass
    return decoder.to(device)


def _ensure_decoder(config: dict[str, Any], vae: Any, device: str) -> Any:
    global _DECODER
    if _DECODER is not None:
        return _DECODER
    started = time.perf_counter()
    decoder = _load_decoder(config, vae.dtype, device)
    z_dim = int(getattr(decoder.config, "z_dim", 0) or 0)
    if z_dim != int(vae.config.z_dim):
        raise ValueError(
            f"Realism decoder latent channels ({z_dim}) do not match Krea 2 ({vae.config.z_dim})."
        )
    if _NOTE is not None:
        _NOTE["load_seconds"] = round(time.perf_counter() - started, 3)
    _DECODER = decoder
    return decoder


def _release_decoder() -> None:
    global _DECODER
    if _DECODER is None:
        return
    _DECODER = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextmanager
def realism_decode_context(request: dict[str, Any]) -> Iterator[dict[str, Any] | None]:
    """Activate the realism decoder for one worker request.

    Yields the mutable run note (``None`` when the stock VAE is requested).
    """
    global _ACTIVE, _NOTE
    config = request.get("decoder") if isinstance(request, dict) else None
    if not isinstance(config, dict) or normalize_decoder_mode(config.get("mode")) != DECODER_REALISM:
        yield None
        return

    _ACTIVE = {
        "mode": DECODER_REALISM,
        "output": normalize_output_mode(config.get("output")),
        "source": str(config.get("source") or DEFAULT_REPO),
        "subfolder": config.get("subfolder"),
    }
    _NOTE = {
        "mode": DECODER_REALISM,
        "decoder": "wan2.1-upscale2x",
        "output": _ACTIVE["output"],
        "source": _ACTIVE["source"],
        "applied": False,
    }
    note = _NOTE
    try:
        yield note
    finally:
        _ACTIVE = None
        _NOTE = None
        _release_decoder()


def make_realism_decode(original: Any):
    """Wrap a Krea ``_decode_latents`` so active requests use the 2x decoder."""

    def _decode_latents_realism(
        pipe: Any,
        vae: Any,
        latents: Any,
        *,
        width: int,
        height: int,
        device: str,
    ) -> list[Any]:
        config = _ACTIVE
        note = _NOTE
        if not config or note is None or note.get("fallback"):
            return original(pipe, vae, latents, width=width, height=height, device=device)

        import torch

        from core.backends.krea2_identity_quality import denormalize_krea_latents

        try:
            decoder = _ensure_decoder(config, vae, device)
            if bool(getattr(vae, "use_tiling", False)):
                try:
                    decoder.enable_tiling()
                except Exception:
                    pass
            work = latents.to(device=device, dtype=decoder.dtype)
            work = pipe._unpack_latents(work, height, width)
            work = denormalize_krea_latents(work.to(decoder.dtype), vae)
            with torch.inference_mode():
                decoded = decoder.decode(work, return_dict=False)[0]
                image = pixel_shuffle_first_frame(decoded)
            del work, decoded
            images = pipe.image_processor.postprocess(image, output_type="pil")
        except Exception as exc:
            # Never fail a finished denoise because of the optional decoder:
            # record why, stop retrying for the remaining images, and decode
            # with the stock VAE instead.
            note["fallback"] = "stock-vae"
            note["error"] = f"{type(exc).__name__}: {exc}"
            _release_decoder()
            return original(pipe, vae, latents, width=width, height=height, device=device)

        note["applied"] = True
        if images:
            note["decoded"] = list(images[0].size)
        note["latent_pixels"] = [int(width), int(height)]
        return images

    _decode_latents_realism._krea_realism_wrapped = True  # type: ignore[attr-defined]
    return _decode_latents_realism


def install_worker_realism_patch(base: Any) -> None:
    """Patch the loaded Krea worker module's ``_decode_latents`` in place."""
    current = getattr(base, "_decode_latents", None)
    if current is None or getattr(current, "_krea_realism_wrapped", False):
        return
    base._decode_latents = make_realism_decode(current)
