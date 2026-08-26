from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from core.backends.krea2_weights import map_krea2_source_key
from core.backends.krea2_worker import overlay_single_file_transformer
from models.discovery import discover_local_image_models
from models.model_descriptor import describe_registry_model
from models.quantization import classify_comfy_quant, parse_comfy_quant_payload
from models.single_file_inspection import inspect_single_image_checkpoint


def _quant_tensor(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def _minimal_krea_file(path: Path, *, quant: dict | None = None) -> None:
    tensors = {
        "first.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "blocks.0.attn.wq.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "txtfusion.projector.weight": torch.ones((2, 2), dtype=torch.bfloat16),
    }
    if quant is not None:
        tensors["first.comfy_quant"] = _quant_tensor(quant)
        tensors["first.weight_scale"] = torch.tensor(2.0, dtype=torch.float32)
    save_file(tensors, str(path))


def test_structural_inspection_detects_dark_beast_style_krea_fp8(tmp_path):
    path = tmp_path / "Dark_Beast_3_FP8.safetensors"
    _minimal_krea_file(path, quant={"format": "float8_e4m3fn"})

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "krea2"
    assert detection["method"] == "tensor_keys"
    assert detection["confidence"] == "high"
    assert detection["variant"] == "base"
    assert detection["quantization"] == "fp8_scaled"


def test_generic_discovery_finds_krea_single_file_outside_family_folder(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    path = root / "Dark_Beast_3_FP8.safetensors"
    _minimal_krea_file(path, quant={"format": "float8_e4m3fn"})

    models = discover_local_image_models(root)

    assert models[path.stem]["arch"] == "krea2"
    assert models[path.stem]["detection"]["quantization"] == "fp8_scaled"


def test_convrot_is_recognized_but_not_runnable(tmp_path):
    path = tmp_path / "Dark_Beast_3_INT8_ConvRot.safetensors"
    _minimal_krea_file(
        path,
        quant={
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        },
    )
    architecture, detection = inspect_single_image_checkpoint(path)
    descriptor = describe_registry_model(
        "Dark Beast 3 INT8 ConvRot",
        {
            "path": path,
            "type": "single",
            "source": "local",
            "arch": architecture,
            "detection": detection,
        },
    )

    assert architecture == "krea2"
    assert detection["quantization"] == "convrot"
    assert descriptor.supported is False
    assert descriptor.defaults["steps"] == 28
    assert descriptor.defaults["cfg"] == 4.5


def test_top_level_quant_metadata_also_blocks_convrot(tmp_path):
    path = tmp_path / "custom_name.safetensors"
    tensors = {
        "first.weight": torch.ones((2, 2), dtype=torch.int8),
        "blocks.0.attn.wq.weight": torch.ones((2, 2), dtype=torch.int8),
        "txtfusion.projector.weight": torch.ones((2, 2), dtype=torch.int8),
    }
    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "layers": {
                    "first": {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 256,
                    }
                },
            }
        )
    }
    save_file(tensors, str(path), metadata=metadata)

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "krea2"
    assert detection["quantization"] == "convrot"


def test_turbo_and_base_defaults_are_model_specific(tmp_path):
    turbo = describe_registry_model(
        "Krea-2-Turbo",
        {
            "path": tmp_path,
            "type": "diffusers",
            "source": "local",
            "arch": "krea2",
        },
    )
    base = describe_registry_model(
        "Dark Beast 3",
        {
            "path": tmp_path,
            "type": "single",
            "source": "local",
            "arch": "krea2",
            "detection": {"variant": "base", "quantization": "fp8"},
        },
    )

    assert turbo.defaults == {"width": 1024, "height": 1024, "steps": 8, "cfg": 0.0}
    assert base.defaults == {"width": 1024, "height": 1024, "steps": 28, "cfg": 4.5}
    assert base.supported is True


def test_key_mapping_covers_original_krea_names():
    assert map_krea2_source_key("first.weight") == ("img_in.weight", None)
    assert map_krea2_source_key("blocks.3.attn.wq.weight") == (
        "transformer_blocks.3.attn.to_q.weight",
        None,
    )
    assert map_krea2_source_key("blocks.3.mod.lin") == (
        "transformer_blocks.3.scale_shift_table",
        "six_rows",
    )
    assert map_krea2_source_key("last.modulation.lin") == (
        "final_layer.scale_shift_table",
        "two_rows",
    )


def test_quant_marker_parser_and_classifier():
    marker = _quant_tensor({"format": "float8_e4m3fn"})
    config = parse_comfy_quant_payload(marker)
    assert classify_comfy_quant(config) == "fp8"
    assert classify_comfy_quant({"format": "int8_tensorwise", "convrot": True}) == "convrot"


class _FakeTransformer:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.bfloat16))

    def state_dict(self, *, keep_vars=False):
        return {"img_in.weight": self.weight if keep_vars else self.weight.detach()}


def test_fp8_scaled_overlay_dequantizes_and_replaces_base_weight(tmp_path):
    path = tmp_path / "dark_beast_fp8.safetensors"
    save_file(
        {
            "first.weight": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
            "first.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "first.comfy_quant": _quant_tensor({"format": "float8_e4m3fn"}),
        },
        str(path),
    )
    transformer = _FakeTransformer()

    result = overlay_single_file_transformer(transformer, path)

    assert result["mapped_tensors"] == 1
    assert torch.equal(transformer.weight.detach(), torch.full((2, 2), 6.0, dtype=torch.bfloat16))


def test_overlay_rejects_convrot_before_copying_weights(tmp_path):
    path = tmp_path / "dark_beast_convrot.safetensors"
    save_file(
        {
            "first.weight": torch.ones((2, 2), dtype=torch.int8),
            "first.weight_scale": torch.ones((2, 1), dtype=torch.float32),
            "first.comfy_quant": _quant_tensor(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
            ),
        },
        str(path),
    )
    transformer = _FakeTransformer()

    with pytest.raises(RuntimeError, match="ConvRot"):
        overlay_single_file_transformer(transformer, path)
    assert torch.equal(transformer.weight.detach(), torch.zeros((2, 2), dtype=torch.bfloat16))
