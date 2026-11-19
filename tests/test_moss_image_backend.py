from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file

from core.backends.moss_image import MossImageDiffusersBackend
from core.backends.moss_image_worker import (
    _configure_memory,
    map_moss_transformer_key,
    overlay_single_file_transformer,
)
from models.discovery import discover_local_image_models
from models.model_descriptor import (
    ModelDescriptor,
    capabilities_for_architecture,
    defaults_for_architecture,
    describe_registry_model,
    detect_diffusers_architecture,
)
from models.single_file_inspection import inspect_single_image_checkpoint


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="Dirty-MossV2",
        path="/models/checkpoints/moss/Dirty-MossV2.safetensors",
        format="single",
        source="local",
        architecture="moss_image",
        backend="moss_image_diffusers",
        capabilities=capabilities_for_architecture("moss_image"),
        defaults=defaults_for_architecture("moss_image"),
        constraints={"dimension_multiple": 16},
        detection={"quantization": "bf16"},
    )


def _minimal_moss_file(path: Path, *, prefix: str = "model.diffusion_model.") -> None:
    tensors = {
        f"{prefix}final_linear_video.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        f"{prefix}video_head_modulation.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        f"{prefix}shared_block_modulation.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        f"{prefix}layers.0.proj.qkv.weight": torch.ones((3, 2), dtype=torch.bfloat16),
        f"{prefix}text_embedder.blocks.0.qkv.weight": torch.ones((3, 2), dtype=torch.bfloat16),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def test_structural_inspection_detects_moss_image_single_file(tmp_path):
    path = tmp_path / "Dirty-MossV2.safetensors"
    _minimal_moss_file(path)

    architecture, detection = inspect_single_image_checkpoint(path)

    assert architecture == "moss_image"
    assert detection["method"] == "tensor_keys"
    assert detection["confidence"] == "high"
    assert detection["quantization"] == "bf16"


def test_generic_discovery_finds_moss_single_file_outside_family_folder(tmp_path):
    root = tmp_path / "checkpoint"
    path = root / "Dirty-MossV2.safetensors"
    _minimal_moss_file(path)

    models = discover_local_image_models(root)
    entry = models[path.stem]

    assert entry["arch"] == "moss_image"
    assert entry["backend"] == "moss_image_diffusers"
    assert entry["detection"]["family"] == "moss_image"

    payload = describe_registry_model(path.stem, entry).to_public_dict()
    assert payload["supported"] is True
    assert payload["defaults"] == {"width": 1024, "height": 1024, "steps": 50, "cfg": 3.0}
    assert payload["constraints"]["dimension_multiple"] == 16


def test_detects_moss_diffusers_pipeline_and_transformer_config(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MOSSImagePipeline"}), encoding="utf-8"
    )
    architecture, detection = detect_diffusers_architecture(tmp_path)
    assert architecture == "moss_image"
    assert detection["confidence"] == "high"

    other = tmp_path / "transformer-only"
    (other / "transformer").mkdir(parents=True)
    (other / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "MOSSImageTransformer"}), encoding="utf-8"
    )
    assert detect_diffusers_architecture(other)[0] == "moss_image"


def test_descriptor_rejects_quantized_moss_single_file(tmp_path):
    path = tmp_path / "Dirty-MossV2-INT8.safetensors"
    _minimal_moss_file(path)

    descriptor = describe_registry_model(
        path.stem,
        {
            "path": path,
            "type": "single",
            "source": "local",
            "arch": "moss_image",
            "detection": {"quantization": "int8"},
        },
    )

    assert descriptor.capabilities.text2img is True
    assert descriptor.supported is False


def test_moss_transformer_key_mapping_splits_only_fused_image_qkv():
    assert map_moss_transformer_key("model.diffusion_model.layers.7.proj.qkv.weight") == [
        "layers.7.proj.q.weight",
        "layers.7.proj.k.weight",
        "layers.7.proj.v.weight",
    ]
    assert map_moss_transformer_key(
        "model.diffusion_model.text_embedder.blocks.0.qkv.weight"
    ) == ["text_embedder.blocks.0.qkv.weight"]
    assert map_moss_transformer_key(
        "model.diffusion_model.shared_block_modulation.weight"
    ) == ["shared_block_modulation.weight"]


class _AttentionProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.v = nn.Linear(4, 4, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = _AttentionProjection()


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


def test_overlay_streams_and_splits_fused_qkv(tmp_path):
    path = tmp_path / "tiny-moss.safetensors"
    fused = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    save_file(
        {"model.diffusion_model.layers.0.proj.qkv.weight": fused},
        str(path),
    )

    transformer = _TinyTransformer()
    report = overlay_single_file_transformer(transformer, path, target_dtype=torch.float32)

    assert report["mapped_tensors"] == 3
    assert report["missing_tensors"] == []
    torch.testing.assert_close(transformer.layers[0].proj.q.weight, fused[0:4])
    torch.testing.assert_close(transformer.layers[0].proj.k.weight, fused[4:8])
    torch.testing.assert_close(transformer.layers[0].proj.v.weight, fused[8:12])


class _FakeProcess:
    def __init__(self, cmd, **_kwargs):
        request_path = Path(cmd[cmd.index("--request") + 1])
        result_path = Path(cmd[cmd.index("--result") + 1])
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        self.payload = json.loads(request_path.read_text(encoding="utf-8"))
        image_path = output_dir / "moss_image_000.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        result_path.write_text(
            json.dumps({"ok": True, "images": [str(image_path)], "seed": 0}),
            encoding="utf-8",
        )
        self.returncode = 0

    def poll(self):
        return self.returncode


def test_moss_backend_serializes_single_file_t2i_request(monkeypatch):
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    backend = MossImageDiffusersBackend()
    images, seed = backend.generate(
        _descriptor(),
        {
            "prompt": "a rubber duck holding a sign that says WebbDuck",
            "negative_prompt": "blurry",
            "width": 1000,
            "height": 1025,
            "steps": 50,
            "cfg": 3.0,
            "num_images": 1,
            "seed": 0,
        },
    )

    assert seed == 0
    assert len(images) == 1
    payload = captured["payload"]
    assert payload["model_path"].endswith("Dirty-MossV2.safetensors")
    assert payload["model_format"] == "single"
    assert payload["component_model"] == "OpenMOSS-Team/MOSS-Image-Base"
    assert payload["width"] == 992
    assert payload["height"] == 1024
    assert payload["guidance_scale"] == 3.0
    assert payload["negative_prompt"] == "blurry"
    assert payload["seed"] == 0


def test_readiness_probes_isolated_runtime_and_component_source(monkeypatch):
    captured = {}

    def fake_probe(python, symbols, **_kwargs):
        captured["python"] = python
        captured["symbols"] = tuple(symbols)
        return {"ready": True, "cuda_available": True}

    monkeypatch.setattr("core.backends.moss_image.probe_python_runtime", fake_probe)
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_PYTHON", "/runtime/moss/python")

    payload = MossImageDiffusersBackend().readiness(_descriptor())

    assert payload["ready"] is True
    assert payload["model_format"] == "single"
    assert payload["component_model"] == "OpenMOSS-Team/MOSS-Image-Base"
    assert captured["python"] == "/runtime/moss/python"
    assert ("diffusers", "AutoencoderKLFlux2") in captured["symbols"]
    assert ("transformers", "Qwen3_5Model") in captured["symbols"]


def test_component_model_env_override_wins(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_COMPONENT_MODEL", "/models/moss-components")
    backend = MossImageDiffusersBackend()
    assert backend.can_handle(_descriptor()) is True

    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    backend.generate(_descriptor(), {"prompt": "duck", "seed": 1})
    assert captured["payload"]["component_model"] == "/models/moss-components"


class _FakePipe:
    def __init__(self):
        self.mode = None

    def enable_sequential_cpu_offload(self, **_kwargs):
        self.mode = "sequential"

    def enable_model_cpu_offload(self, **_kwargs):
        self.mode = "model"

    def to(self, device):
        self.mode = str(device)


def test_moss_worker_uses_sequential_offload_on_16gb_gpu(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 16.0) == "sequential"
    assert pipe.mode == "sequential"


def test_moss_worker_uses_model_offload_when_components_fit(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 32.0) == "model"
    assert pipe.mode == "model"
