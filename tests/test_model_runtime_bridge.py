from pathlib import Path

from core import model_runtime
from models.catalog import build_runtime_registry
from models.model_descriptor import ModelCapabilities, ModelDescriptor


class _Backend:
    def __init__(self, backend_id="sdxl_diffusers"):
        self.backend_id = backend_id

    def generate(self, descriptor, settings, **kwargs):
        return [descriptor.name], settings.get("seed", 123)


def test_runtime_registry_preserves_sdxl_and_adds_hf_flux(tmp_path):
    checkpoint_root = tmp_path / "checkpoint"
    sdxl_path = checkpoint_root / "sdxl" / "existing"
    (sdxl_path / "unet").mkdir(parents=True)
    (sdxl_path / "unet" / "config.json").write_text("{}", encoding="utf-8")
    (sdxl_path / "text_encoder_2").mkdir()

    hf_cache = tmp_path / "hub"
    flux = hf_cache / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc"
    flux.mkdir(parents=True)
    (flux / "model_index.json").write_text('{"_class_name": "Flux2KleinPipeline"}', encoding="utf-8")

    registry = build_runtime_registry(
        {"Existing SDXL": {"type": "diffusers", "arch": "sdxl", "path": sdxl_path, "source": "local", "defaults": {"steps": 30, "cfg": 6.0}}},
        models_root=tmp_path,
        checkpoint_root=checkpoint_root,
        hf_cache=hf_cache,
    )

    assert registry["Existing SDXL"]["defaults"]["steps"] == 30
    assert registry["black-forest-labs/FLUX.2-klein-4B"]["arch"] == "flux"


def test_requested_operation_is_capability_driven():
    assert model_runtime.requested_operation({}) == "text2img"
    assert model_runtime.requested_operation({"image": "image.png"}) == "img2img"
    assert model_runtime.requested_operation({"mask_image": "mask.png"}) == "inpaint"
    assert model_runtime.requested_operation({"smart_extend": True, "image": "image.png"}) == "outpaint"


def _assert_routes(monkeypatch, descriptor: ModelDescriptor, backend_id: str):
    backend = _Backend(backend_id)
    monkeypatch.setattr(model_runtime, "descriptor_for_model", lambda _name: descriptor)
    monkeypatch.setattr(model_runtime, "_register_installed_backends", lambda: None)
    monkeypatch.setattr(model_runtime.backend_resolver, "resolve", lambda _descriptor: backend)
    images, seed = model_runtime.run_selected_model(
        {"base_model": descriptor.name, "prompt": "duck", "seed": 42}
    )
    assert images == [descriptor.name]
    assert seed == 42


def test_flux_routes_through_backend_resolver(monkeypatch):
    descriptor = ModelDescriptor(
        name="FLUX.2-klein-4B",
        path="flux",
        format="diffusers",
        source="test",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=ModelCapabilities(text2img=True, img2img=True),
    )
    _assert_routes(monkeypatch, descriptor, "flux_diffusers")


def test_krea_routes_through_backend_resolver(monkeypatch):
    descriptor = ModelDescriptor(
        name="Krea-2-Turbo",
        path="krea",
        format="diffusers",
        source="test",
        architecture="krea2",
        backend="krea2_diffusers",
        capabilities=ModelCapabilities(text2img=True),
    )
    _assert_routes(monkeypatch, descriptor, "krea2_diffusers")


def test_sdxl_routes_through_normal_backend_identity(monkeypatch):
    descriptor = ModelDescriptor(
        name="Existing SDXL",
        path="sdxl",
        format="diffusers",
        source="test",
        architecture="sdxl",
        backend="sdxl_diffusers",
        capabilities=ModelCapabilities(text2img=True, img2img=True, inpaint=True, outpaint=True),
    )
    _assert_routes(monkeypatch, descriptor, "sdxl_diffusers")
