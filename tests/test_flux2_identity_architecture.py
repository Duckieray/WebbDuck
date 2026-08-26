from core.backends.flux import FluxDiffusersBackend
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture, defaults_for_architecture


def test_explicit_flux2_descriptor_supports_native_identity():
    descriptor = ModelDescriptor(
        name="black-forest-labs/FLUX.2-klein-9B",
        path="/cache/flux2",
        format="diffusers",
        source="hf_cache",
        architecture="flux2",
        backend="flux_diffusers",
        capabilities=capabilities_for_architecture("flux2"),
        defaults=defaults_for_architecture("flux2"),
        constraints={"dimension_multiple": 8},
    )

    assert descriptor.capabilities.identity_adapter is True
    assert FluxDiffusersBackend().can_handle(descriptor) is True
    assert descriptor.defaults["steps"] == 4
    assert descriptor.defaults["cfg"] == 1.0


def test_flux1_does_not_advertise_flux2_native_identity():
    assert capabilities_for_architecture("flux1").identity_adapter is False
