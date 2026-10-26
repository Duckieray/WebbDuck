from pathlib import Path

import pytest

from core.backends.base import BackendResolver, GenerationBackend
from models.model_descriptor import (
    _BACKENDS,
    _CAPABILITIES,
    ModelDescriptor,
    defaults_for_model,
    detect_diffusers_architecture,
    describe_registry_model,
)


class _Backend(GenerationBackend):
    backend_id = "test_backend"

    def can_handle(self, descriptor: ModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id

    def generate(self, descriptor, settings, **kwargs):
        return {"name": descriptor.name, "settings": settings}


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_detects_flux_from_model_index(tmp_path):
    _write_json(tmp_path / "model_index.json", '{"_class_name": "FluxPipeline"}')
    arch, detection = detect_diffusers_architecture(tmp_path)
    assert arch == "flux"
    assert detection["confidence"] == "high"


def test_detects_krea2_from_model_index(tmp_path):
    _write_json(tmp_path / "model_index.json", '{"_class_name": "Krea2Pipeline"}')
    arch, _ = detect_diffusers_architecture(tmp_path)
    assert arch == "krea2"


def test_detects_sdxl_from_existing_structure(tmp_path):
    _write_json(tmp_path / "unet" / "config.json", '{}')
    (tmp_path / "text_encoder_2").mkdir()
    arch, detection = detect_diffusers_architecture(tmp_path)
    assert arch == "sdxl"
    assert detection["method"] == "directory_structure"


def test_unknown_diffusers_model_stays_unknown(tmp_path):
    _write_json(tmp_path / "model_index.json", '{"_class_name": "FutureImagePipeline"}')
    arch, _ = detect_diffusers_architecture(tmp_path)
    assert arch == "unknown"


def test_krea_public_descriptor_is_runnable_and_exposes_lora_ui(tmp_path):
    descriptor = describe_registry_model(
        "Krea-2-Turbo",
        {
            "path": tmp_path,
            "type": "diffusers",
            "source": "local",
            "arch": "krea2",
        },
    )
    payload = descriptor.to_public_dict()
    assert payload["name"] == "Krea-2-Turbo"
    assert payload["arch"] == "krea2"
    assert payload["capabilities"]["text2img"] is True
    # TASK-012: Krea 2 gains latent-init img2img via the pipeline's
    # latents=/sigmas= entry points. Masked editing still has no trained runtime.
    assert payload["capabilities"]["img2img"] is True
    assert payload["capabilities"]["inpaint"] is False
    assert payload["capabilities"]["outpaint"] is False
    assert payload["capabilities"]["lora"] is True
    assert payload["defaults"]["steps"] == 8
    assert payload["defaults"]["cfg"] == 0.0
    assert payload["supported"] is True
    assert "architecture" not in payload
    assert "backend" not in payload


def test_sdxl_backend_has_normal_installed_identity(tmp_path):
    descriptor = describe_registry_model(
        "existing-sdxl",
        {
            "path": tmp_path,
            "type": "diffusers",
            "source": "local",
            "arch": "sdxl",
        },
    )
    assert descriptor.backend == "sdxl_diffusers"
    assert descriptor.supported is True


def test_backend_resolver_routes_by_descriptor_without_caller_arch_branching():
    resolver = BackendResolver()
    resolver.register(_Backend())
    descriptor = ModelDescriptor(
        name="example",
        path="/tmp/example",
        format="diffusers",
        source="test",
        architecture="future_arch",
        backend="test_backend",
    )
    assert resolver.resolve(descriptor).backend_id == "test_backend"


def test_backend_resolver_fails_actionably_when_unsupported():
    resolver = BackendResolver()
    descriptor = ModelDescriptor(
        name="future-model",
        path="/tmp/future",
        format="diffusers",
        source="test",
    )
    with pytest.raises(LookupError, match="future-model"):
        resolver.resolve(descriptor)


def test_every_generatable_architecture_declares_explicit_inference_defaults():
    # Regression: the UI only restores steps/cfg when the selected architecture
    # publishes defaults, so a missing entry makes the form silently keep the
    # previously selected model's values. SDXL had no entry, so switching back
    # from FLUX.2 left it at steps=4 / cfg=1.0 and outpainting came out blurry.
    generatable = sorted(set(_CAPABILITIES) | set(_BACKENDS))
    assert "sdxl" in generatable
    for architecture in generatable:
        defaults = defaults_for_model(architecture, "test-model")
        assert "steps" in defaults, f"{architecture} declares no default step count"
        assert "cfg" in defaults, f"{architecture} declares no default guidance"
        assert isinstance(defaults["steps"], int)
        assert isinstance(defaults["cfg"], (int, float))
        # Must stay inside the slider bounds declared in ui/index.html, otherwise
        # the browser silently clamps the restored value.
        assert 1 <= defaults["steps"] <= 100, architecture
        assert 0 <= defaults["cfg"] <= 20, architecture


def test_switching_back_to_sdxl_restores_the_sdxl_inference_contract():
    # The exact reported failure: FLUX.2 defaults followed the user into SDXL.
    sdxl_defaults = defaults_for_model("sdxl", "some-sdxl-checkpoint")
    assert sdxl_defaults["steps"] == 30
    assert sdxl_defaults["cfg"] == 7.0
    assert sdxl_defaults["steps"] != defaults_for_model("flux2", "flux2")["steps"]
