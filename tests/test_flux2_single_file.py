"""Discovery + descriptor coverage for FLUX.2 single-file (bf16/fp8/...) checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from models.discovery import discover_local_image_models
from models.model_descriptor import describe_registry_model
from models.single_file_inspection import _flux_parameter_size, inspect_single_image_checkpoint


def _klein_tensors(dtype=torch.bfloat16):
    """A minimal set of tensors matching the FLUX.2 Klein transformer signature.

    Only the fingerprint keys need to be present; the inspector reads the header
    (names + dtypes) and never materializes weight values.
    """
    return {
        "img_in.weight": torch.ones((16, 16), dtype=dtype),
        "txt_in.weight": torch.ones((16, 16), dtype=dtype),
        "time_in.in_layer.weight": torch.ones((16, 16), dtype=dtype),
        "time_in.out_layer.weight": torch.ones((16, 16), dtype=dtype),
        "final_layer.adaLN_modulation.1.weight": torch.ones((4, 4), dtype=dtype),
        "final_layer.linear.weight": torch.ones((16, 16), dtype=dtype),
        "double_stream_modulation_img.lin.weight": torch.ones((16, 16), dtype=dtype),
        "double_stream_modulation_txt.lin.weight": torch.ones((16, 16), dtype=dtype),
        "single_stream_modulation.lin.weight": torch.ones((16, 16), dtype=dtype),
        "double_blocks.0.img_attn.norm.key_norm.scale": torch.ones((4,), dtype=dtype),
        "double_blocks.0.img_attn.norm.query_norm.scale": torch.ones((4,), dtype=dtype),
        "double_blocks.0.img_attn.proj.weight": torch.ones((16, 16), dtype=dtype),
        "double_blocks.0.img_attn.qkv.weight": torch.ones((48, 16), dtype=dtype),
        "double_blocks.0.img_mlp.0.weight": torch.ones((16, 16), dtype=dtype),
        "double_blocks.0.img_mlp.2.weight": torch.ones((16, 16), dtype=dtype),
        "double_blocks.0.txt_attn.proj.weight": torch.ones((16, 16), dtype=dtype),
        "double_blocks.0.txt_attn.qkv.weight": torch.ones((48, 16), dtype=dtype),
        "single_blocks.0.linear1.weight": torch.ones((16, 16), dtype=dtype),
        "single_blocks.0.linear2.weight": torch.ones((16, 16), dtype=dtype),
        "single_blocks.0.norm.key_norm.scale": torch.ones((4,), dtype=dtype),
        "single_blocks.0.norm.query_norm.scale": torch.ones((4,), dtype=dtype),
    }


def test_detects_flux2_klein_single_file_fp8(tmp_path):
    path = tmp_path / "snofsKlein9bDistilled.safetensors"
    save_file(_klein_tensors(dtype=torch.float8_e4m3fn), str(path))

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "flux"
    assert detection["method"] == "tensor_keys"
    assert detection["confidence"] == "high"
    assert detection["family"] == "flux2"
    assert detection["quantization"] == "fp8"
    assert detection["parameter_size"] == "9B"


def test_detects_flux2_klein_single_file_bf16(tmp_path):
    path = tmp_path / "FluxKlein4b.safetensors"
    save_file(_klein_tensors(dtype=torch.bfloat16), str(path))

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "flux"
    assert detection["quantization"] == "bf16"
    assert detection["parameter_size"] == "4B"


def test_prefixed_flux2_keys_are_normalized(tmp_path):
    # A `transformer.`-prefixed export normalizes to the same signature.
    tensors = {f"transformer.{k}": v for k, v in _klein_tensors().items()}
    path = tmp_path / "klein_9b.safetensors"
    save_file(tensors, str(path))

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "flux"
    assert detection["family"] == "flux2"


def test_unrelated_safetensors_is_not_flux(tmp_path):
    path = tmp_path / "random.safetensors"
    save_file({"foo.weight": torch.ones((2, 2))}, str(path))

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "unknown"


def test_flux_parameter_size_variants():
    assert _flux_parameter_size("snofsKlein9bDistilled") == "9B"
    assert _flux_parameter_size("FLUX2-klein-9b") == "9B"
    assert _flux_parameter_size("pGGUF_v14_4b_x") == "4B"
    assert _flux_parameter_size("v149b") is None  # revision-style, not a size
    assert _flux_parameter_size("bucket") is None


def test_discovery_registers_flux2_single_file(tmp_path):
    root = tmp_path / "checkpoints" / "flux"
    root.mkdir(parents=True)
    path = root / "snofsKlein9bDistilled.safetensors"
    save_file(_klein_tensors(dtype=torch.float8_e4m3fn), str(path))

    models = discover_local_image_models(root)

    assert path.stem in models
    assert models[path.stem]["arch"] == "flux"
    assert models[path.stem]["type"] == "single"
    assert models[path.stem]["detection"]["quantization"] == "fp8"


def test_descriptor_is_supported_and_inpaint_ready(tmp_path):
    path = tmp_path / "snofsKlein9bDistilled.safetensors"
    save_file(_klein_tensors(dtype=torch.float8_e4m3fn), str(path))
    models = discover_local_image_models(tmp_path)
    entry = models[path.stem]

    descriptor = describe_registry_model(path.stem, entry)

    assert descriptor.architecture == "flux"
    assert descriptor.format == "single"
    assert descriptor.backend == "flux_diffusers"
    assert descriptor.supported is True
    caps = descriptor.capabilities.to_dict()
    assert caps["text2img"] is True
    assert caps["img2img"] is True
    assert caps["inpaint"] is True
    assert caps["identity_adapter"] is True
