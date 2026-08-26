from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.flux_lora import (
    apply_flux_loras,
    inject_lora_trigger,
    lora_trigger_phrase,
    resolve_flux_loras,
)
from models.model_descriptor import backend_for_architecture


def test_resolve_flux_loras_preserves_studio_name_weight_contract(tmp_path, monkeypatch):
    from models import registry as asset_registry

    path = tmp_path / "redcraft.safetensors"
    path.write_bytes(b"adapter")
    monkeypatch.setattr(
        asset_registry,
        "LORA_REGISTRY",
        {
            "RedCraft": {
                "path": path,
                "arch": "flux",
                "trigger": "redcraft_style",
                "weight": 1.0,
                "source": "local",
            }
        },
    )

    resolved = resolve_flux_loras([{"name": "RedCraft", "weight": 0.85}])

    assert resolved == [
        {
            "name": "RedCraft",
            "path": str(path),
            "weight": 0.85,
            "trigger": "redcraft_style",
        }
    ]
    phrase = lora_trigger_phrase(resolved)
    assert phrase == "(redcraft_style:0.85)"
    assert inject_lora_trigger("portrait", phrase) == "portrait, (redcraft_style:0.85)"


def test_resolve_flux_loras_rejects_incompatible_architecture(tmp_path, monkeypatch):
    from models import registry as asset_registry

    path = tmp_path / "sdxl.safetensors"
    path.write_bytes(b"adapter")
    monkeypatch.setattr(
        asset_registry,
        "LORA_REGISTRY",
        {"Wrong": {"path": path, "arch": "sdxl", "weight": 1.0}},
    )

    with pytest.raises(ValueError, match="not FLUX"):
        resolve_flux_loras([{"name": "Wrong", "weight": 1.0}])


def test_apply_flux_loras_uses_named_weighted_adapters_without_fusing(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    class FakePipe:
        def __init__(self):
            self.loads = []
            self.adapters = None
            self.fused = False

        def load_lora_weights(self, path, adapter_name=None, **kwargs):
            self.loads.append((path, adapter_name, kwargs))

        def set_adapters(self, names, adapter_weights=None):
            self.adapters = (list(names), list(adapter_weights or []))

        def fuse_lora(self, *args, **kwargs):
            self.fused = True

    pipe = FakePipe()
    applied = apply_flux_loras(
        pipe,
        [
            {"name": "A", "path": str(first), "weight": 0.7},
            {"name": "B", "path": str(second), "weight": 1.2},
        ],
    )

    assert len(pipe.loads) == 2
    assert pipe.adapters is not None
    assert pipe.adapters[1] == [0.7, 1.2]
    assert pipe.fused is False
    assert [item["weight"] for item in applied] == [0.7, 1.2]


def test_detect_lora_arch_recognizes_modern_flux_key_layouts(monkeypatch):
    from models import registry as asset_registry

    class FakeSafeOpen:
        def __init__(self, keys):
            self._keys = keys

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return self._keys

    key_sets = [
        ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"],
        ["transformer.single_transformer_blocks.3.proj_mlp.lora_B.weight"],
        ["lora_unet_double_blocks_0_img_attn_q.lora_down.weight"],
        ["diffusion_model.single_blocks.1.linear1.lora_A.weight"],
    ]

    for keys in key_sets:
        monkeypatch.setattr(asset_registry, "safe_open", lambda *_a, _keys=keys, **_k: FakeSafeOpen(_keys))
        assert asset_registry.detect_lora_arch(Path("adapter.safetensors")) == "flux"


def test_runtime_catalog_lora_endpoint_supports_gguf_models(monkeypatch):
    from server import model_catalog_api

    registry = {
        "flux-2-klein-9b-Q5_K_M": {
            "type": "gguf",
            "arch": "flux",
            "backend": backend_for_architecture("flux"),
            "path": "/models/flux/flux-2-klein-9b-Q5_K_M.gguf",
            "source": "local",
            "detection": {
                "family": "flux2_klein",
                "parameter_size": "9B",
                "quantization": "Q5_K_M",
            },
        }
    }
    monkeypatch.setattr(model_catalog_api, "runtime_registry", lambda: registry)
    monkeypatch.setattr(
        model_catalog_api,
        "LORA_REGISTRY",
        {
            "RedCraft": {"arch": "flux", "description": "FLUX", "weight": 1.0},
            "SDXL": {"arch": "sdxl", "description": "SDXL", "weight": 1.0},
        },
    )

    result = model_catalog_api.list_model_loras("flux-2-klein-9b-Q5_K_M")

    assert result == [{"name": "RedCraft", "description": "FLUX", "weight": 1.0}]


def test_studio_lora_api_uses_runtime_catalog_route():
    source = Path("ui/core/api.js").read_text(encoding="utf-8")

    assert "`/model-catalog/${encodeURIComponent(modelName)}/loras`" in source
