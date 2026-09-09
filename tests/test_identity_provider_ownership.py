from pathlib import Path

import pytest

from core import model_runtime
from models.model_descriptor import ModelCapabilities, ModelDescriptor


def _descriptor(architecture: str, backend: str, *, identity: bool = True) -> ModelDescriptor:
    return ModelDescriptor(
        name=f"test-{architecture}",
        path=architecture,
        format="diffusers",
        source="test",
        architecture=architecture,
        backend=backend,
        capabilities=ModelCapabilities(text2img=True, identity_adapter=identity),
    )


def test_sdxl_replaces_stale_krea_provider_and_resets_sdxl_defaults():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "krea2_identity_edit",
            "reference_images": ["/refs/person.png"],
            "ref_boost": 9.0,
            "lora_scale": 1.0,
        }
    }
    model_runtime._normalize_identity_adapter_for_descriptor(
        _descriptor("sdxl", "sdxl_diffusers"), settings
    )
    cfg = settings["identity_adapter"]
    assert cfg["type"] == "faceid_sdxl"
    assert cfg["reference_images"] == ["/refs/person.png"]
    assert cfg["adapter_scale"] == 1.0
    assert cfg["lora_scale"] == 0.60
    assert "ref_boost" not in cfg


def test_krea_replaces_stale_sdxl_provider_and_never_keeps_sdxl_lora_scale():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "faceid_sdxl",
            "reference_images": ["/refs/person.png"],
            "adapter_scale": 1.0,
            "lora_scale": 0.60,
            "embedder": "buffalo_l",
        }
    }
    model_runtime._normalize_identity_adapter_for_descriptor(
        _descriptor("krea2", "krea2_diffusers"), settings
    )
    cfg = settings["identity_adapter"]
    assert cfg["type"] == "krea2_identity_edit"
    assert cfg["reference_images"] == ["/refs/person.png"]
    assert cfg["ref_boost"] == 4.0
    assert cfg["grounding_px"] == 768
    assert cfg["lora_scale"] == 1.0
    assert "lora_rank" not in cfg
    assert "embedder" not in cfg
    assert "adapter_scale" not in cfg


def test_flux2_replaces_stale_faceid_provider_with_native_references():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "faceid_sdxl",
            "reference_images": ["/refs/a.png", "/refs/b.png"],
        }
    }
    model_runtime._normalize_identity_adapter_for_descriptor(
        _descriptor("flux2", "flux_diffusers"), settings
    )
    cfg = settings["identity_adapter"]
    assert cfg["type"] == "flux2_native"
    assert cfg["reference_images"] == ["/refs/a.png", "/refs/b.png"]
    assert "embedder" not in cfg
    assert "ref_boost" not in cfg


def test_correct_provider_preserves_explicit_provider_tuning():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "krea2_identity_edit",
            "reference_images": ["/refs/person.png"],
            "ref_boost": 5.0,
            "lora_scale": 0.9,
            "lora_rank": "r128",
        }
    }
    model_runtime._normalize_identity_adapter_for_descriptor(
        _descriptor("krea2", "krea2_diffusers"), settings
    )
    assert settings["identity_adapter"]["ref_boost"] == 5.0
    assert settings["identity_adapter"]["lora_scale"] == 0.9
    assert settings["identity_adapter"]["lora_rank"] == "r128"


def test_krea_auto_rank_is_removed_so_backend_can_choose_from_gpu_vram():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "krea2_identity_edit",
            "reference_images": ["/refs/person.png"],
            "lora_rank": "auto",
        }
    }
    model_runtime._normalize_identity_adapter_for_descriptor(
        _descriptor("krea2", "krea2_diffusers"), settings
    )
    assert "lora_rank" not in settings["identity_adapter"]


def test_krea_explicit_rank_is_not_removed():
    for rank in ("full", "r128", "r64"):
        settings = {
            "identity_adapter": {
                "enabled": True,
                "type": "krea2_identity_edit",
                "reference_images": ["/refs/person.png"],
                "lora_rank": rank,
            }
        }
        model_runtime._normalize_identity_adapter_for_descriptor(
            _descriptor("krea2", "krea2_diffusers"), settings
        )
        assert settings["identity_adapter"]["lora_rank"] == rank


def test_architecture_without_identity_support_rejects_adapter():
    settings = {
        "identity_adapter": {
            "enabled": True,
            "type": "faceid_sdxl",
            "reference_images": ["/refs/person.png"],
        }
    }
    with pytest.raises(ValueError, match="does not expose a supported identity adapter"):
        model_runtime._normalize_identity_adapter_for_descriptor(
            _descriptor("flux1", "flux_diffusers", identity=False), settings
        )


def test_persona_ui_exposes_only_the_required_provider_option():
    source = Path("ui/modules/PersonaIdentityUI.js").read_text(encoding="utf-8")
    assert "select.replaceChildren(onlyOption)" in source
    assert "select.disabled = true" in source
    assert "faceid_sdxl" in source
    assert "krea2_identity_edit" in source
    assert "flux2_native" in source
    assert "Identity Provider (automatic)" in source
