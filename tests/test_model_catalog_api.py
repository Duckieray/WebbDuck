from __future__ import annotations

from fastapi import HTTPException
import pytest

from models.model_descriptor import ModelCapabilities, ModelDescriptor
import server.model_catalog_api as model_api


def test_list_model_catalog_returns_public_profiles(monkeypatch):
    payload = [
        {
            "name": "RealVisXL",
            "type": "diffusers",
            "defaults": {"steps": 30},
            "capabilities": {"text2img": True, "second_pass": True},
            "constraints": {"dimension_multiple": 8},
            "supported": True,
        },
        {
            "name": "Future Model",
            "type": "diffusers",
            "defaults": {},
            "capabilities": {"text2img": True},
            "constraints": {"dimension_multiple": 16},
            "supported": False,
        },
    ]
    monkeypatch.setattr(model_api, "public_runtime_catalog", lambda: payload)

    assert model_api.list_model_catalog() == payload
    assert "architecture" not in payload[0]
    assert "backend" not in payload[0]


def test_get_model_profile_is_architecture_free(monkeypatch):
    descriptor = ModelDescriptor(
        name="Example",
        path="/models/example",
        format="diffusers",
        source="local",
        architecture="sdxl",
        backend="sdxl_legacy",
        capabilities=ModelCapabilities(text2img=True, second_pass=True),
        defaults={"steps": 30, "cfg": 6.0},
        constraints={"dimension_multiple": 8},
    )
    monkeypatch.setattr(model_api, "runtime_registry", lambda: {"Example": {}})
    monkeypatch.setattr(model_api, "descriptor_for_model", lambda name, registry: descriptor)

    profile = model_api.get_model_profile("Example")
    assert profile["name"] == "Example"
    assert profile["supported"] is True
    assert profile["capabilities"]["second_pass"] is True
    assert "architecture" not in profile
    assert "backend" not in profile


def test_get_model_profile_returns_404_for_unknown(monkeypatch):
    monkeypatch.setattr(model_api, "runtime_registry", lambda: {})

    def missing(_name, _registry):
        raise KeyError("missing")

    monkeypatch.setattr(model_api, "descriptor_for_model", missing)

    with pytest.raises(HTTPException) as exc_info:
        model_api.get_model_profile("missing")
    assert exc_info.value.status_code == 404
