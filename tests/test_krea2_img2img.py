"""Focused coverage for Krea 2 latent-init img2img (TASK-012).

The sigma math is the load-bearing part: Krea's scheduler runs with
``use_dynamic_shifting=True``, so any sigmas handed to ``Krea2Pipeline.__call__``
are exponentially time-shifted. These tests pin the strength -> sigma mapping, the
double-shift correction, and the request plumbing so ``strength`` cannot silently
stop meaning anything.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.backends import krea2_img2img as i2i
from models.model_descriptor import capabilities_for_architecture


def _scheduler(use_dynamic_shifting: bool = True):
    return SimpleNamespace(
        config={
            "use_dynamic_shifting": use_dynamic_shifting,
            "base_image_seq_len": 256,
            "max_image_seq_len": 6400,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
    )


def _pipe(distilled: bool = False, use_dynamic_shifting: bool = True):
    return SimpleNamespace(
        scheduler=_scheduler(use_dynamic_shifting),
        config=SimpleNamespace(is_distilled=distilled),
    )


# ---------------------------------------------------------------------------
# Strength normalization
# ---------------------------------------------------------------------------


def test_strength_defaults_to_text2img():
    assert i2i.normalize_strength(None) == 1.0
    assert i2i.normalize_strength(1.0) == 1.0


@pytest.mark.parametrize("raw", [0, -1, -0.5, 0.0, "0"])
def test_strength_clamps_to_usable_floor(raw):
    assert i2i.normalize_strength(raw) == i2i.MIN_STRENGTH


@pytest.mark.parametrize("raw", [1.5, 5, 99])
def test_strength_clamps_to_one(raw):
    assert i2i.normalize_strength(raw) == 1.0


def test_strength_survives_junk():
    assert i2i.normalize_strength("junk") == 1.0
    assert i2i.normalize_strength(float("nan"), default=0.4) == 0.4
    assert i2i.normalize_strength(float("inf")) == 1.0
    assert i2i.normalize_strength(None, default=0.25) == 0.25


def test_strength_accepts_numeric_strings():
    assert i2i.normalize_strength("0.35") == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# mu selection (Turbo fixed shift vs Raw resolution shift)
# ---------------------------------------------------------------------------


def test_distilled_variant_uses_fixed_shift():
    assert i2i.schedule_mu(_pipe(distilled=True), image_seq_len=4608) == 1.15


def test_raw_variant_shift_grows_with_sequence_length():
    small = i2i.schedule_mu(_pipe(distilled=False), image_seq_len=256)
    large = i2i.schedule_mu(_pipe(distilled=False), image_seq_len=6400)
    assert small == pytest.approx(0.5)
    assert large == pytest.approx(1.15)
    assert large > small


def test_raw_variant_reads_scheduler_config_overrides():
    pipe = _pipe(distilled=False)
    pipe.scheduler.config["base_shift"] = 0.8
    pipe.scheduler.config["max_shift"] = 1.6
    assert i2i.schedule_mu(pipe, image_seq_len=256) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Time-shift inverse
# ---------------------------------------------------------------------------


def _shift(mu: float, t: float) -> float:
    import math

    if t <= 0.0:
        return 0.0
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


@pytest.mark.parametrize("mu", [1.15, 1.2956, 0.5])
def test_invert_cancels_the_scheduler_shift(mu):
    sigmas = [0.99, 0.85, 0.7, 0.5, 0.3, 0.1]
    inverted = i2i.invert_krea_time_shift(sigmas, mu)
    for original, restored in zip(sigmas, inverted):
        assert _shift(mu, restored) == pytest.approx(original, abs=1e-9)


def test_invert_passes_fixed_points_through():
    assert i2i.invert_krea_time_shift([0.0, 1.0], 1.15) == [0.0, 1.0]


def test_invert_without_mu_is_identity():
    assert i2i.invert_krea_time_shift([0.5, 0.25], None) == [0.5, 0.25]


# ---------------------------------------------------------------------------
# strength -> truncated sigma schedule
# ---------------------------------------------------------------------------


FULL_12 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]


def test_full_strength_keeps_whole_schedule():
    truncated, t_start, init_sigma = i2i.img2img_sigma_schedule(12, 1.0, FULL_12)
    assert t_start == 0
    assert truncated == FULL_12
    assert init_sigma == pytest.approx(1.0)


def test_half_strength_keeps_noisiest_half():
    truncated, t_start, init_sigma = i2i.img2img_sigma_schedule(12, 0.5, FULL_12)
    assert t_start == 6
    assert len(truncated) == 6
    assert init_sigma == pytest.approx(0.4)


def test_lower_strength_keeps_fewer_lower_sigmas():
    low, _t_low, low_sigma = i2i.img2img_sigma_schedule(12, 0.35, FULL_12)
    high, _t_high, high_sigma = i2i.img2img_sigma_schedule(12, 0.75, FULL_12)
    assert len(low) < len(high)
    # A gentler denoise must start from *less* noise.
    assert low_sigma < high_sigma


def test_truncation_always_ends_at_the_schedule_tail():
    for strength in (1.0, 0.9, 0.75, 0.5, 0.35, 0.2, 0.05):
        truncated, _t_start, _sigma = i2i.img2img_sigma_schedule(12, strength, FULL_12)
        assert truncated[-1] == FULL_12[-1]


def test_truncation_never_returns_empty():
    # A degenerate/empty schedule must still yield one runnable sigma rather than
    # handing the scheduler an empty list.
    truncated, t_start, init_sigma = i2i.img2img_sigma_schedule(12, 0.5, [])
    assert truncated == [0.0]
    assert t_start == 0
    assert init_sigma == pytest.approx(0.0)


def test_schedule_is_monotonic_and_descending():
    for strength in (1.0, 0.8, 0.6, 0.4, 0.2):
        truncated, _t_start, _sigma = i2i.img2img_sigma_schedule(12, strength, FULL_12)
        assert truncated == sorted(truncated, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in truncated)


def test_denoise_step_count_matches_truncated_schedule():
    for strength in (1.0, 0.75, 0.5, 0.35, 0.05):
        truncated, _t_start, _sigma = i2i.img2img_sigma_schedule(12, strength, FULL_12)
        assert i2i.denoise_step_count(12, strength) == len(truncated)


def test_denoise_step_count_never_zero():
    assert i2i.denoise_step_count(12, 0.0) >= 1
    assert i2i.denoise_step_count(1, 0.05) >= 1


# ---------------------------------------------------------------------------
# Host request resolution
# ---------------------------------------------------------------------------


def test_no_input_image_is_text2img():
    assert i2i.resolve_img2img_request({}) is None
    assert i2i.resolve_img2img_request({"strength": 0.4}) is None
    assert i2i.resolve_img2img_request({"image": "   "}) is None


def test_full_strength_with_image_is_text2img():
    assert i2i.resolve_img2img_request({"image": "/tmp/a.png", "strength": 1.0}) is None
    assert i2i.resolve_img2img_request({"image": "/tmp/a.png"}) is None


def test_image_with_strength_becomes_img2img_payload():
    payload = i2i.resolve_img2img_request({"image": "/tmp/a.png", "strength": 0.45})
    assert payload == {"input_image": "/tmp/a.png", "strength": pytest.approx(0.45)}


def test_input_image_key_is_accepted():
    payload = i2i.resolve_img2img_request({"input_image": "/tmp/b.png", "strength": 0.6})
    assert payload is not None
    assert payload["input_image"] == "/tmp/b.png"


def test_payload_is_json_serializable():
    payload = i2i.resolve_img2img_request({"image": "/tmp/a.png", "strength": 0.5})
    assert json.loads(json.dumps(payload)) == payload


def test_strength_above_one_collapses_to_text2img():
    # Clamping up to 1.0 means "start from pure noise", i.e. text-to-image, so no
    # payload is emitted rather than a bogus img2img run.
    assert i2i.resolve_img2img_request({"image": "/tmp/a.png", "strength": 9}) is None


def test_strength_below_floor_still_produces_a_payload():
    payload = i2i.resolve_img2img_request({"image": "/tmp/a.png", "strength": 0})
    assert payload is not None
    assert payload["strength"] == i2i.MIN_STRENGTH


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


def test_krea2_advertises_img2img():
    caps = capabilities_for_architecture("krea2")
    assert caps.img2img is True
    # Masked editing still has no trained Krea runtime.
    assert caps.inpaint is False
    assert caps.outpaint is False


# --------------------------------------------------------------------------------------
# Latent normalization: encode must be the exact inverse of decode
# --------------------------------------------------------------------------------------


torch = pytest.importorskip("torch")


class _FakeLatents:
    def __init__(self, value):
        self._value = value

    def mode(self):
        return self._value


class _FakeVaeConfig:
    def __init__(self, mean, std):
        self.latents_mean = mean
        self.latents_std = std
        self.z_dim = len(mean)


class _FakeVae:
    """Just enough VAE surface for ``encode_source_latent``."""

    def __init__(self, latent, mean, std):
        self.config = _FakeVaeConfig(mean, std)
        self._latent = latent
        self.dtype = torch.float32

    def to(self, device):
        return self

    def encode(self, px):
        return SimpleNamespace(latent_dist=_FakeLatents(self._latent))


class _FakeImage:
    def convert(self, mode):
        return self


class _FakeProcessor:
    @staticmethod
    def preprocess(image, height=None, width=None):
        return torch.zeros(1, 3, height, width)


class _FakePipe:
    patch_size = 2
    vae_scale_factor = 8
    image_processor = _FakeProcessor()

    @staticmethod
    def _pack_latents(latents, batch, channels, lh, lw):
        return latents


def test_encode_source_latent_inverts_the_pipeline_decode():
    """``encode_source_latent`` must emit the exact space ``_decode_latents`` reads.

    Krea's decode (pipeline_krea2.py, mirrored in ``_decode_latents``) is::

        latents_std = 1.0 / config.latents_std
        latents = latents / latents_std + latents_mean

    which is ``latent = n * config.latents_std + mean`` and therefore normalizes as
    ``n = (latent - mean) / config.latents_std``.

    Encoding with ``(latent - mean) * config.latents_std`` instead inflates the
    latent by ``config.latents_std ** 2`` -- for Krea-2-Turbo that is 2.1x-7.9x per
    channel. The transformer is then handed a latent far outside the unit-variance
    space it denoises in and discards the source entirely, re-rendering it. That is
    what painted hard-edged, oddly coloured discs onto faces: the face detailer and
    every latent-init img2img run shared this encoder.
    """
    mean = [0.1, -0.2, 0.3, -0.4]
    # Real Krea-2-Turbo per-channel values, so the regression trips on magnitude.
    std = [2.8184, 1.4541, 2.3275, 2.6558]
    z_dim = len(mean)
    raw = torch.arange(z_dim, dtype=torch.float32).reshape(1, z_dim, 1, 1, 1) * 3.0 + 7.0

    vae = _FakeVae(raw, mean, std)
    encoded, _seq = i2i.encode_source_latent(
        _FakePipe(), vae, _FakeImage(), width=64, height=64, device="cpu"
    )

    # ``encode_source_latent`` drops the temporal axis: (B, z, lh, lw).
    raw2d = raw[:, :, 0]
    mean_t = torch.tensor(mean).view(1, z_dim, 1, 1)
    std_t = torch.tensor(std).view(1, z_dim, 1, 1)
    expected = (raw2d - mean_t) / std_t
    assert encoded.shape == raw2d.shape, (encoded.shape, raw2d.shape)
    assert torch.allclose(encoded, expected, atol=1e-4), (
        "encode_source_latent is not (latent - mean) / latents_std; the denoiser "
        "receives a latent scaled by latents_std**2 away from the decode space"
    )

    # And the full round trip is lossless.
    assert torch.allclose(encoded * std_t + mean_t, raw2d, atol=1e-4)


def test_encode_source_latent_does_not_inflate_by_std_squared():
    """Explicit guard on the sign/direction of the normalization."""
    mean = [0.0]
    std = [2.0]  # std**2 == 4.0
    raw = torch.full((1, 1, 1, 1, 1), 2.0)

    vae = _FakeVae(raw, mean, std)
    encoded, _seq = i2i.encode_source_latent(
        _FakePipe(), vae, _FakeImage(), width=64, height=64, device="cpu"
    )

    assert encoded.flatten().tolist() == pytest.approx([1.0])  # 2.0 / 2.0
    # The inverted form would have produced 4.0, a 4x overshoot.
    assert encoded.abs().max().item() < std[0] ** 2 * raw.abs().max().item() / std[0] ** 2 + 1.0
