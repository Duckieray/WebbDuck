from __future__ import annotations

import pytest

from core.backends.flux import _normalized_flux_detection
from core.backends.flux_worker import (
    _flux_demotion_tiers,
    _img2img_sigma_schedule,
    _invert_flux_time_shift,
)


def _full_sigmas(steps: int):
    """Descending flow-matching sigma schedule, matching FlowMatchEulerDiscrete."""
    return [1.0 - i / steps for i in range(steps)]
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture


def _descriptor(detection: dict | None = None) -> ModelDescriptor:
    return ModelDescriptor(
        name="community-merged-Q5_K_M",
        path="/models/flux/community-merged-Q5_K_M.gguf",
        format="gguf",
        source="local",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=capabilities_for_architecture("flux"),
        detection=dict(detection or {}),
    )


def test_merged_gguf_recovers_9b_from_actual_component_model():
    detection = _normalized_flux_detection(
        _descriptor({"quantization": "Q5_K_M"}),
        "black-forest-labs/FLUX.2-klein-9B",
    )

    assert detection["parameter_size"] == "9B"
    assert detection["family"] == "flux2_klein"
    assert detection["component_model"] == "black-forest-labs/FLUX.2-klein-9B"


def test_component_model_can_recover_4b_without_filename_metadata():
    detection = _normalized_flux_detection(
        _descriptor({"quantization": "Q5_K_M"}),
        "black-forest-labs/FLUX.2-klein-4B",
    )

    assert detection["parameter_size"] == "4B"


def test_valid_explicit_size_is_not_overridden():
    detection = _normalized_flux_detection(
        _descriptor({"parameter_size": "9B", "quantization": "Q5_K_M"}),
        "black-forest-labs/FLUX.2-klein-9B",
    )

    assert detection["parameter_size"] == "9B"


def test_demotion_tiers_start_with_request_and_are_monotonic():
    request = (832, 1216, 4, 2)
    tiers = _flux_demotion_tiers(*request)

    assert tiers[0] == {
        "condition_count": 4,
        "width": 832,
        "height": 1216,
        "lora_count": 2,
    }
    assert len(tiers) > 1
    area_tracked = 10**18
    refs_tracked = object()
    loras_tracked = object()
    for tier in tiers:
        this_area = tier["width"] * tier["height"]
        assert this_area <= area_tracked, f"area increased at {tier}"
        area_tracked = this_area
        assert tier["lora_count"] <= (loras_tracked if isinstance(loras_tracked, int) else 10**9)
        loras_tracked = tier["lora_count"]
        assert tier["condition_count"] <= (refs_tracked if isinstance(refs_tracked, int) else 10**9)
        refs_tracked = tier["condition_count"]
        assert tier["width"] % 8 == 0 and tier["height"] % 8 == 0
        assert tier["width"] >= 8 and tier["height"] >= 8


def test_demotion_tiers_lower_lora_count_before_0_condition_images():
    tiers = _flux_demotion_tiers(1024, 1024, 2, 2)
    with_lora = [
        tier for tier in tiers if tier["lora_count"] == 1 and tier["condition_count"] >= 1
    ]
    assert with_lora, "expected a tier dropping one LoRA while keeping conditioning"

    last = tiers[-1]
    assert last["lora_count"] == 0 and last["condition_count"] == 0

    tier0_lora = [t for t in tiers if t["condition_count"] == 2 and t["lora_count"] == 2]
    assert len(tier0_lora) == 3, "2 refs + 2 loras at full res and both scaled resolutions"


def test_demotion_tiers_single_lora_keeps_conditioning_first():
    tiers = _flux_demotion_tiers(1024, 1024, 1, 1)
    assert tiers[0] == {
        "condition_count": 1,
        "width": 1024,
        "height": 1024,
        "lora_count": 1,
    }
    with_conditioning = [t for t in tiers if t["condition_count"] >= 1]
    assert with_conditioning, "expected at least one tier keeping the request's conditioning"
    assert tiers[-1]["condition_count"] == 0


def test_demotion_tiers_shrink_resolution_proportionally():
    tiers = _flux_demotion_tiers(1600, 900, 3, 1)
    scaled = [t for t in tiers if t["condition_count"] == 3 and t["lora_count"] == 1 and t != tiers[0]]
    assert scaled, "expected resolution-only tiers"
    for tier in scaled:
        ratio = tier["width"] / tier["height"]
        assert abs(ratio - 1600 / 900) < 0.02, f"aspect drifted at {tier}"


def test_demotion_tiers_preserve_conditioning_for_img2img():
    # For img2img the identity references are the ONLY conditioning (the source
    # body is the latent init). Dropping them does not help the OOM ceiling and
    # only strips identity, so every tier must keep the full conditioning set.
    tiers = _flux_demotion_tiers(832, 1216, 4, 2, preserve_conditioning=True)
    assert tiers[0]["condition_count"] == 4
    for tier in tiers:
        assert tier["condition_count"] == 4, f"conditioning dropped at {tier}"
    assert any(t["lora_count"] < 2 for t in tiers), "expected adapter fallback tiers"
    assert any(t["width"] < 832 for t in tiers), "expected resolution fallback tiers"


def test_img2img_sigma_strength_1_keeps_entire_schedule():
    steps, strength = 8, 1.0
    sigmas, t_start, init_sigma = _img2img_sigma_schedule(steps, strength, _full_sigmas(steps))
    assert t_start == 0
    assert len(sigmas) == steps
    assert init_sigma == 1.0


def test_img2img_sigma_strength_truncates_from_noisy_end():
    steps, strength = 8, 0.65
    sigmas, t_start, init_sigma = _img2img_sigma_schedule(steps, strength, _full_sigmas(steps))
    # t_start = steps * (1 - strength) = 8 * 0.35 = 2.8 -> 2; keeps sigmas[2:].
    assert t_start == 2
    assert len(sigmas) == steps - t_start
    assert init_sigma == pytest.approx(0.75)
    # Retained schedule is strictly descending and starts where it was cut.
    assert sigmas[0] == init_sigma
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))


def test_img2img_sigma_strength_is_monotonic_in_strength():
    full = _full_sigmas(10)
    starts = {}
    for strength in (0.2, 0.5, 0.75, 1.0):
        _, t_start, init_sigma = _img2img_sigma_schedule(10, strength, full)
        starts[strength] = (t_start, init_sigma)
    # Stronger strength keeps more (noisier) steps and starts from a higher sigma.
    assert starts[0.2][0] > starts[0.5][0] > starts[0.75][0] > starts[1.0][0]
    assert starts[0.2][1] < starts[0.5][1] < starts[0.75][1] < starts[1.0][1]


def test_img2img_sigma_clamps_out_of_range_strength_and_empty_schedule():
    full = _full_sigmas(8)
    hi = _img2img_sigma_schedule(8, 1.5, full)
    assert hi == (list(full), 0, 1.0)
    # Degenerate strength 0 keeps a single (near-final) step without crashing.
    lo = _img2img_sigma_schedule(8, 0.0, full)
    assert len(lo[0]) >= 1


def _exponential_shift(sigmas, mu):
    """Mirror FlowMatchEulerDiscreteScheduler._time_shift_exponential with sigma=1.0.

    This is the transform the FLUX.2 Klein pipe re-applies to any sigmas passed to
    set_timesteps(sigmas=..., mu=mu) when use_dynamic_shifting=True.
    """
    import math

    exp_mu = math.exp(mu)
    out = []
    for s in sigmas:
        if s <= 0.0 or s >= 1.0:
            out.append(float(s))
        else:
            out.append(exp_mu / (exp_mu + (1.0 / s - 1.0)))
    return out


def test_invert_flux_time_shift_round_trips_through_forward_shift():
    mu = 9.0
    truncated = [0.9172, 0.8146, 0.5063, 0.1047, 0.0]
    deshifted = _invert_flux_time_shift(truncated, mu)
    re_shifted = _exponential_shift(deshifted, mu)
    assert len(re_shifted) == len(truncated)
    for got, want in zip(re_shifted, truncated):
        assert abs(got - want) < 1e-6, (got, want)


def test_invert_flux_time_shift_is_noop_when_mu_is_none():
    truncated = [0.9, 0.5, 0.0]
    assert _invert_flux_time_shift(truncated, None) == truncated


def test_invert_flux_time_shift_preserves_endpoint_fixed_points():
    # 1.0 -> 1.0 and 0.0 -> 0.0 under the exponential shift.
    out = _invert_flux_time_shift([1.0, 0.5, 0.0], 7.0)
    assert out[0] == 1.0
    assert out[-1] == 0.0
    assert 0.0 < out[1] < 1.0


def test_invert_flux_time_shift_reduces_midschedule_sigmas():
    # The inverse pulls non-endpoint values down so the scheduler's second shift
    # lands them back where we started instead of inflating toward 1.0.
    mu = 9.0
    original = 0.5
    deshifted = _invert_flux_time_shift([original], mu)[0]
    assert deshifted < original
    # And re-shifting recovers the original value.
    assert abs(_exponential_shift([deshifted], mu)[0] - original) < 1e-9


def test_flux_pipeline_class_selects_inpaint_variant():
    from core.backends.flux_worker import _flux_pipeline_class

    base = _flux_pipeline_class(inpaint=False)
    inpaint = _flux_pipeline_class(inpaint=True)

    assert base.__name__ == "Flux2KleinPipeline"
    assert "Inpaint" in inpaint.__name__
    assert inpaint is not base
