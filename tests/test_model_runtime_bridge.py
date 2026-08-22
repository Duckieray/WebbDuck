from pathlib import Path

import pytest

from core import model_runtime
from models.catalog import build_runtime_registry
from models.model_descriptor import ModelCapabilities, ModelDescriptor


class _Backend:
    backend_id = "sdxl_legacy"

    def generate(self, descriptor, settings, **kwargs):
        return [descriptor.name], settings.get("seed", 123)


def test_runtime_registry_preserves_legacy_and_adds_hf_model(tmp_path):
    checkpoint_root = tmp_path / "checkpoint"
    legacy_path = checkpoint_root / "sdxl" / "legacy"
    (legacy_path / "unet").mkdir(parents=True)
    (legacy_path / "unet" / "config.json").write_text("{}", encoding="utf-8")
    (legacy_path / "text_encoder_2").mkdir()

    hf_cache = tmp_path / "hub"
    flux = hf_cache / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc"
    flux.mkdir(parents=True)
    (flux / "model_index.json").write_text('{"_class_name": "FluxPipeline"}', encoding="utf-8")

    legacy = {
        "Legacy SDXL": {
            "type": "diffusers",
            "arch": "sdxl",
            "path": legacy_path,
            "source": "local",
            "defaults": {"steps": 30, "cfg": 6.0},
        }
    }
    registry = build_runtime_registry(
        legacy,
        models_root=tmp_path,
        checkpoint_root=checkpoint_root,
        hf_cache=hf_cache,
    )

    assert registry["Legacy SDXL"]["defaults"]["steps"] == 30
    assert "black-forest-labs/FLUX.2-klein-4B" in registry
    assert registry["black-forest-labs/FLUX.2-klein-4B"]["arch"] == "flux"


def test_requested_operation_is_capability_driven():
    assert model_runtime.requested_operation({}) == "text2img"
    assert model_runtime.requested_operation({"image": "/tmp/a.png"}) == "img2img"
    assert model_runtime.requested_operation({"mask_image": "/tmp/m.png"}) == "inpaint"
    assert model_runtime.requested_operation({"smart_extend": True, "image": "/tmp/a.png"}) == "outpaint"


def test_unrunnable_recognized_model_fails_before_legacy_pipeline(monkeypatch):
    descriptor = ModelDescriptor(
        name="FLUX fixture",
        path="/tmp/flux",
        format="diffusers",
        source="test",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=ModelCapabilities(text2img=True),
    )
    monkeypatch.setattr(model_runtime, "descriptor_for_model", lambda _name: descriptor)

    with pytest.raises(RuntimeError, match="discovered successfully"):
        model_runtime.run_selected_model({"base_model": "FLUX fixture", "prompt": "duck"})


def test_sdxl_routes_through_backend_resolver(monkeypatch):
    descriptor = ModelDescriptor(
        name="Legacy SDXL",
        path="/tmp/sdxl",
        format="diffusers",
        source="test",
        architecture="sdxl",
        backend="sdxl_legacy",
        capabilities=ModelCapabilities(text2img=True, img2img=True, inpaint=True, outpaint=True),
    )
    backend = _Backend()
    monkeypatch.setattr(model_runtime, "descriptor_for_model", lambda _name: descriptor)
    monkeypatch.setattr(model_runtime, "ensure_sdxl_registered", lambda: backend)
    monkeypatch.setattr(model_runtime.backend_resolver, "resolve", lambda _descriptor: backend)

    images, seed = model_runtime.run_selected_model(
        {"base_model": "Legacy SDXL", "prompt": "duck", "seed": 42}
    )
    assert images == ["Legacy SDXL"]
    assert seed == 42
