from pathlib import Path

import pytest

from core.backends.base import BackendResolver, GenerationBackend
from models.model_descriptor import (
    ModelDescriptor,
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
    _write_json(
        tmp_path / "model_index.json",
        '{"_class_name": "FluxPipeline"}',
    )
    arch, detection = detect_diffusers_architecture(tmp_path)
    assert arch == "flux"
    assert detection["confidence"] == "high"


def test_detects_krea2_from_model_index(tmp_path):
    _write_json(
        tmp_path / "model_index.json",
        '{"_class_name": "Krea2Pipeline"}',
    )
    arch, _ = detect_diffusers_architecture(tmp_path)
    assert arch == "krea2"


def test_detects_sdxl_from_legacy_structure(tmp_path):
    _write_json(tmp_path / "unet" / "config.json", '{}')
    (tmp_path / "text_encoder_2").mkdir()
    arch, detection = detect_diffusers_architecture(tmp_path)
    assert arch == "sdxl"
    assert detection["method"] == "directory_structure"


def test_unknown_diffusers_model_stays_unknown(tmp_path):
    _write_json(tmp_path / "model_index.json", '{"_class_name": "FutureImagePipeline"}')
    arch, _ = detect_diffusers_architecture(tmp_path)
    assert arch == "unknown"


def test_public_descriptor_hides_architecture_and_backend(tmp_path):
    descriptor = describe_registry_model(
        "Krea-2-Turbo",
        {
            "path": tmp_path,
            "type": "diffusers",
            "source": "local",
            "arch": "krea2",
            "defaults": {"steps": 8, "cfg": 0.0},
        },
    )
    payload = descriptor.to_public_dict()
    assert payload["name"] == "Krea-2-Turbo"
    assert payload["capabilities"]["text2img"] is True
    assert payload["capabilities"]["inpaint"] is False
    assert "architecture" not in payload
    assert "backend" not in payload


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
