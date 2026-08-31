from __future__ import annotations

from core.backends.flux import _normalized_flux_detection
from core.backends.flux_worker import _flux_demotion_tiers
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
