from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.flux_lora import (
    _convert_flux2_diffusers_suffix_lora_to_peft,
    _maybe_convert_diffusers_suffix_lora_to_peft,
    apply_flux_loras,
    inject_lora_trigger,
    lora_trigger_phrase,
    resolve_flux_loras,
)
from models.model_descriptor import backend_for_architecture


def test_convert_diffusers_suffix_lora_to_peft_handles_native_transformer_paths():
    import torch

    state = {
        "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.randn(64, 4096),
        "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.randn(4096, 64),
        "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(64.0),
        "transformer.single_transformer_blocks.0.attn.to_out.lora_down.weight": torch.randn(64, 16384),
        "transformer.single_transformer_blocks.0.attn.to_out.lora_up.weight": torch.randn(4096, 64),
        "transformer.single_transformer_blocks.0.attn.to_out.alpha": torch.tensor(64.0),
        "transformer.single_transformer_blocks.0.attn.to_out.dora_scale": torch.randn(64),
    }

    converted = _convert_flux2_diffusers_suffix_lora_to_peft(state)

    # dora_scale dropped, alpha/leaf suffixes folded into lora_A/lora_B only
    assert "transformer.single_transformer_blocks.0.attn.to_out.dora_scale" not in converted
    assert not any(k.endswith((".alpha", ".dora_scale", ".lora_down.weight", ".lora_up.weight")) for k in converted)

    key = "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
    assert key in converted
    assert converted[key].shape == (64, 4096)
    # alpha(64)/rank(64) == 1.0
    assert torch.allclose(converted[key], state["transformer.transformer_blocks.0.attn.to_q.lora_down.weight"])

    up_key = "transformer.transformer_blocks.0.attn.to_q.lora_B.weight"
    assert up_key in converted
    assert torch.equal(
        converted[up_key],
        state["transformer.transformer_blocks.0.attn.to_q.lora_up.weight"],
    )


def test_maybe_convert_diffusers_suffix_lora_only_for_native_transformer_paths(tmp_path, monkeypatch):
    import torch

    fake_state = {
        "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.randn(4, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.randn(4, 4),
    }
    monkeypatch.setattr(
        "core.backends.flux_lora._load_lora_tensor_dict",
        lambda path: fake_state,
    )

    path = tmp_path / "native.safetensors"
    path.write_bytes(b"x")
    assert _maybe_convert_diffusers_suffix_lora_to_peft(path) is not None

    # PEFT already (no lora_down suffix) -> no conversion
    monkeypatch.setattr(
        "core.backends.flux_lora._load_lora_tensor_dict",
        lambda path: {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 4)},
    )
    assert _maybe_convert_diffusers_suffix_lora_to_peft(path) is None

    # Flat Kohya lora_unet_* (diffusers handles) -> no conversion
    monkeypatch.setattr(
        "core.backends.flux_lora._load_lora_tensor_dict",
        lambda path: {"lora_unet_double_blocks_0_img_attn_q.lora_down.weight": torch.randn(4, 4)},
    )
    assert _maybe_convert_diffusers_suffix_lora_to_peft(path) is None


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
    assert phrase == "redcraft_style"
    assert inject_lora_trigger("portrait", phrase) == "portrait, redcraft_style"


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


def test_resolve_flux_loras_accepts_flux1_and_flux2(tmp_path, monkeypatch):
    from models import registry as asset_registry

    for arch in ("flux", "flux1", "flux2"):
        path = tmp_path / f"{arch}.safetensors"
        path.write_bytes(b"adapter")
        monkeypatch.setattr(
            asset_registry,
            "LORA_REGISTRY",
            {f"LoRA_{arch}": {"path": path, "arch": arch, "weight": 1.0}},
        )
        resolved = resolve_flux_loras([{"name": f"LoRA_{arch}", "weight": 0.5}])
        assert len(resolved) == 1
        assert resolved[0]["name"] == f"LoRA_{arch}"


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
    assert pipe.loads[0][0] == str(tmp_path)
    assert pipe.loads[0][2]["weight_name"] == "first.safetensors"
    assert pipe.loads[0][2]["low_cpu_mem_usage"] is True
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

    # FLUX.1 key patterns (Kohya / Comfy exports)
    flux1_key_sets = [
        ["lora_unet_double_blocks_0_img_attn_q.lora_down.weight"],
        ["diffusion_model.single_blocks.1.linear1.lora_A.weight"],
    ]
    for keys in flux1_key_sets:
        monkeypatch.setattr(asset_registry, "safe_open", lambda *_a, _keys=keys, **_k: FakeSafeOpen(_keys))
        assert asset_registry.detect_lora_arch(Path("adapter.safetensors")) == "flux1"

    # Generic transformer.* namespace without FLUX.2 markers falls back to "flux"
    generic_key_sets = [
        ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"],
        ["transformer.single_transformer_blocks.3.proj_mlp.lora_B.weight"],
    ]
    for keys in generic_key_sets:
        monkeypatch.setattr(asset_registry, "safe_open", lambda *_a, _keys=keys, **_k: FakeSafeOpen(_keys))
        assert asset_registry.detect_lora_arch(Path("adapter.safetensors")) == "flux"


def test_detect_lora_arch_distinguishes_flux2_by_key_patterns(monkeypatch):
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

    # FLUX.2 exclusive keys
    flux2_key_sets = [
        ["guidance_in.in_layer.lora_A.weight", "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight"],
        ["transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"],
    ]
    for keys in flux2_key_sets:
        monkeypatch.setattr(asset_registry, "safe_open", lambda *_a, _keys=keys, **_k: FakeSafeOpen(_keys))
        assert asset_registry.detect_lora_arch(Path("adapter.safetensors")) == "flux2"


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


def test_catalog_lora_endpoint_filters_flux1_from_flux2_checkpoint(monkeypatch):
    from server import model_catalog_api

    registry = {
        "flux-2-klein-9b": {
            "type": "gguf",
            "arch": "flux",
            "backend": backend_for_architecture("flux"),
            "path": "/models/flux/flux-2-klein-9b.gguf",
            "source": "local",
            "detection": {"family": "flux2_klein"},
        }
    }
    monkeypatch.setattr(model_catalog_api, "runtime_registry", lambda: registry)
    monkeypatch.setattr(
        model_catalog_api,
        "LORA_REGISTRY",
        {
            "Flux1_LoRA": {"arch": "flux1", "description": "FLUX.1 style", "weight": 1.0},
            "Flux2_LoRA": {"arch": "flux2", "description": "FLUX.2 style", "weight": 1.0},
            "Generic_Flux": {"arch": "flux", "description": "Generic", "weight": 1.0},
            "SDXL_LoRA": {"arch": "sdxl", "description": "SDXL", "weight": 1.0},
        },
    )

    result = model_catalog_api.list_model_loras("flux-2-klein-9b")
    names = [entry["name"] for entry in result]

    # Generic "flux" checkpoint shows all FLUX-family LoRAs
    assert "Flux1_LoRA" in names
    assert "Flux2_LoRA" in names
    assert "Generic_Flux" in names
    assert "SDXL_LoRA" not in names


def test_studio_lora_api_uses_runtime_catalog_route():
    source = Path("ui/core/api.js").read_text(encoding="utf-8")

    assert "`/model-catalog/${encodeURIComponent(modelName)}/loras`" in source


def test_studio_drops_loras_not_offered_by_new_checkpoint():
    source = Path("ui/modules/LoraManager.js").read_text(encoding="utf-8")

    assert "for (const name of Array.from(this.selectedLoras.keys()))" in source
    assert "if (this.availableLorasMap.has(name)) continue;" in source
    assert "this.selectedLoras.delete(name);" in source
