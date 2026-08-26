from __future__ import annotations

from core.backends.flux import _normalized_flux_detection
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
