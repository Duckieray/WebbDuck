from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file

from core.backends.moss_image import MossImageDiffusersBackend
from core.backends.moss_image_worker import (
    _configure_memory,
    _reference_paths,
    _resolve_component_source,
    _snapshot_is_complete,
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


@pytest.fixture(autouse=True)
def _isolated_lora_registry(monkeypatch):
    """Keep identity tests independent of LoRAs installed on the host."""
    import models.registry as registry

    monkeypatch.setattr(registry, "LORA_REGISTRY", {})
    monkeypatch.setattr("core.backends.moss_identity_face._largest_face", lambda _img: None)


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
    assert payload["capabilities"]["identity_adapter"] is True


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


def test_moss_backend_serializes_identity_reference_payload(monkeypatch, tmp_path):
    reference = tmp_path / "person.png"
    Image.new("RGB", (64, 64), "white").save(reference)
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    settings = {
        "prompt": "a woman standing on a rooftop at sunset",
        "seed": 7,
        "identity_adapter": {
            "enabled": True,
            "type": "moss_image_reference",
            "reference_images": [str(reference)],
        },
    }
    MossImageDiffusersBackend().generate(_descriptor(), settings)

    payload = captured["payload"]
    assert payload["reference_images"] == [str(reference.resolve())]
    assert payload["reference_pixels"] == 640 * 640
    assert payload["isolate_references"] is True
    assert payload["prompt"].startswith("Same person as Picture 1.")
    assert payload["prompt"].endswith("a woman standing on a rooftop at sunset")
    debug = settings["identity_adapter_debug"]
    assert debug["provider"] == "moss_image_reference"
    assert debug["reference_images_resolved"] == 1


def test_moss_backend_identity_prompt_prefix_override(monkeypatch, tmp_path):
    reference = tmp_path / "person.png"
    Image.new("RGB", (64, 64), "white").save(reference)
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    prefix = "Same character and art style as Picture 1."
    MossImageDiffusersBackend().generate(
        _descriptor(),
        {
            "prompt": "in a park",
            "seed": 1,
            "identity_adapter": {
                "enabled": True,
                "type": "moss_image_reference",
                "prompt_prefix": prefix,
                "reference_images": [str(reference)],
            },
        },
    )
    assert captured["payload"]["prompt"] == f"{prefix} in a park"


def test_moss_backend_identity_does_not_duplicate_prompt_prefix(monkeypatch, tmp_path):
    reference = tmp_path / "person.png"
    Image.new("RGB", (64, 64), "white").save(reference)
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    MossImageDiffusersBackend().generate(
        _descriptor(),
        {
            "prompt": "Same person as Picture 1. in a park",
            "seed": 1,
            "identity_adapter": {
                "enabled": True,
                "type": "moss_image_reference",
                "reference_images": [str(reference)],
            },
        },
    )
    assert captured["payload"]["prompt"] == "Same person as Picture 1. in a park"


def test_moss_backend_rejects_multiple_identity_references(tmp_path):
    refs = []
    for index in range(2):
        path = tmp_path / f"person-{index}.png"
        Image.new("RGB", (32, 32), "white").save(path)
        refs.append(str(path))

    with pytest.raises(ValueError, match="at most 1 reference image"):
        MossImageDiffusersBackend().generate(
            _descriptor(),
            {
                "prompt": "portrait",
                "identity_adapter": {
                    "enabled": True,
                    "type": "moss_image_reference",
                    "reference_images": refs,
                },
            },
        )


def test_moss_backend_identity_rejects_missing_reference(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MossImageDiffusersBackend().generate(
            _descriptor(),
            {
                "prompt": "portrait",
                "identity_adapter": {
                    "enabled": True,
                    "type": "moss_image_reference",
                    "reference_images": [str(missing)],
                },
            },
        )


def test_moss_reference_budget_clamps_only_on_low_vram():
    from core.backends.moss_image_worker import _effective_reference_pixels

    assert _effective_reference_pixels(640 * 640, 15.56) == 640 * 640
    assert _effective_reference_pixels(640 * 640, 12.0) == 512 * 512
    assert _effective_reference_pixels(256 * 256, 12.0) == 256 * 256
    assert _effective_reference_pixels(640 * 640, 0.0) == 640 * 640


def test_moss_worker_reference_paths_validates_and_caps(tmp_path):
    path = tmp_path / "person.png"
    Image.new("RGB", (32, 32), "white").save(path)
    assert _reference_paths({"reference_images": [str(path)]}) == [path]
    with pytest.raises(FileNotFoundError):
        _reference_paths({"reference_images": [str(tmp_path / "missing.png")]})
    with pytest.raises(ValueError, match="at most 4"):
        _reference_paths({"reference_images": [str(path)] * 5})


def _identity_request(monkeypatch, tmp_path, *, loras=None, adapter_extra=None):
    import models.registry as registry

    reference = tmp_path / "person.png"
    Image.new("RGB", (64, 64), "white").save(reference)
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)
    adapter = {
        "enabled": True,
        "type": "moss_image_reference",
        "reference_images": [str(reference)],
    }
    adapter.update(adapter_extra or {})
    settings = {
        "prompt": "a woman in a park",
        "seed": 3,
        "loras": loras or [],
        "identity_adapter": adapter,
    }
    MossImageDiffusersBackend().generate(_descriptor(), settings)
    return captured["payload"], settings


def _armv5_registry(monkeypatch, tmp_path):
    import models.registry as registry

    lora_path = tmp_path / "armV5.safetensors"
    lora_path.write_bytes(b"")
    entry = {"path": str(lora_path), "arch": "moss_image", "weight": 1.0, "trigger": None}
    monkeypatch.setattr(registry, "LORA_REGISTRY", {"armV5": entry})


def test_moss_identity_auto_attaches_installed_edit_adapter(monkeypatch, tmp_path):
    _armv5_registry(monkeypatch, tmp_path)
    payload, settings = _identity_request(monkeypatch, tmp_path)

    assert [entry["name"] for entry in payload["loras"]] == ["armV5"]
    assert payload["loras"][0]["weight"] == 1.0
    assert payload["prompt"] == "Keep the same person from Picture 1. a woman in a park"
    debug = settings["identity_adapter_debug"]
    assert debug["edit_adapter"] == "armV5"
    assert debug["edit_adapter_source"] == "auto"


def test_moss_identity_does_not_duplicate_selected_edit_adapter(monkeypatch, tmp_path):
    _armv5_registry(monkeypatch, tmp_path)
    payload, settings = _identity_request(
        monkeypatch, tmp_path, loras=[{"name": "armV5", "weight": 0.8}]
    )

    assert [entry["name"] for entry in payload["loras"]] == ["armV5"]
    assert payload["loras"][0]["weight"] == 0.8
    assert settings["identity_adapter_debug"]["edit_adapter_source"] == "selected"


def test_moss_identity_can_disable_edit_adapter(monkeypatch, tmp_path):
    _armv5_registry(monkeypatch, tmp_path)
    payload, settings = _identity_request(
        monkeypatch, tmp_path, adapter_extra={"edit_adapter": "none"}
    )

    assert payload["loras"] == []
    assert payload["prompt"] == "Same person as Picture 1. a woman in a park"
    assert settings["identity_adapter_debug"]["edit_adapter"] is None


def test_moss_identity_reports_missing_edit_adapter(monkeypatch, tmp_path):
    payload, settings = _identity_request(monkeypatch, tmp_path)

    assert payload["loras"] == []
    assert payload["prompt"] == "Same person as Picture 1. a woman in a park"
    assert settings["identity_adapter_debug"]["edit_adapter"] is None


def test_moss_identity_rejects_unknown_explicit_edit_adapter(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown identity edit adapter"):
        _identity_request(
            monkeypatch, tmp_path, adapter_extra={"edit_adapter": "missing_adapter"}
        )


def test_moss_face_reference_masks_outside_face(monkeypatch, tmp_path):
    import numpy as np

    from core.backends import moss_identity_face as face

    source = tmp_path / "photo.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (800, 600, 3), dtype=np.uint8)).save(source)
    monkeypatch.setenv("WEBBDUCK_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(face, "_largest_face", lambda _img: (200, 250, 200, 260))

    prepared, masked = face.prepare_face_reference(source)
    assert masked is True
    out = np.asarray(Image.open(prepared).convert("RGB"))
    assert out.shape == (1024, 1024, 3)
    # Corners are outside the oval, so they are the neutral gray fill.
    assert np.all(np.abs(out[:20, :20].astype(int) - 128) <= 2)

    again, _ = face.prepare_face_reference(source)
    assert again == prepared


def test_moss_face_reference_falls_back_when_no_face(monkeypatch, tmp_path):
    from core.backends import moss_identity_face as face

    source = tmp_path / "scenery.png"
    Image.new("RGB", (300, 300), "green").save(source)
    monkeypatch.setenv("WEBBDUCK_OUTPUT_DIR", str(tmp_path / "out"))

    path, masked = face.prepare_face_reference(source)
    assert masked is False
    assert path == str(source.resolve())


def test_moss_identity_uses_face_reference_by_default_and_full_on_request(monkeypatch, tmp_path):
    reference = tmp_path / "person.png"
    Image.new("RGB", (400, 400), "white").save(reference)
    monkeypatch.setenv("WEBBDUCK_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(
        "core.backends.moss_identity_face._largest_face", lambda _img: (100, 100, 200, 200)
    )
    captured = {}

    def fake_popen(cmd, **kwargs):
        process = _FakeProcess(cmd, **kwargs)
        captured["payload"] = process.payload
        return process

    monkeypatch.setattr("core.backends.moss_image.subprocess.Popen", fake_popen)

    def run(extra):
        adapter = {"enabled": True, "type": "moss_image_reference", "reference_images": [str(reference)]}
        adapter.update(extra)
        MossImageDiffusersBackend().generate(
            _descriptor(), {"prompt": "a woman", "seed": 2, "identity_adapter": adapter}
        )
        return captured["payload"]["reference_images"][0], adapter

    face_path, face_adapter = run({})
    assert face_path.endswith(".png") and ".moss_identity_cache" in face_path
    assert face_adapter["reference_face_masked"] is True

    full_path, _ = run({"reference_crop": "full"})
    assert full_path == str(reference.resolve())


def test_moss_identity_rejects_unknown_reference_crop(tmp_path):
    reference = tmp_path / "person.png"
    Image.new("RGB", (64, 64), "white").save(reference)
    with pytest.raises(ValueError, match="reference_crop"):
        MossImageDiffusersBackend().generate(
            _descriptor(),
            {
                "prompt": "portrait",
                "identity_adapter": {
                    "enabled": True,
                    "type": "moss_image_reference",
                    "reference_images": [str(reference)],
                    "reference_crop": "torso",
                },
            },
        )


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


def test_encode_prompt_texts_pads_rows_and_keeps_true_length_masks():
    from types import SimpleNamespace

    from core.backends.moss_image_worker import _encode_prompt_texts

    class _Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def __call__(self, text, **_kwargs):
            return {"input_ids": torch.tensor([[1 if text == "short" else 3]])}

    class _TextEncoder(nn.Module):
        def forward(self, input_ids, **_kwargs):
            length = int(input_ids[0, 0].item())
            return SimpleNamespace(
                last_hidden_state=torch.ones(1, length, 4)
            )

    pipe = SimpleNamespace(tokenizer=_Tokenizer(), text_encoder=_TextEncoder())
    embeds, mask = _encode_prompt_texts(pipe, ["long", "short"], 4096)

    assert embeds.shape == (2, 3, 4)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert torch.all(embeds[1, 1:] == 0)


def test_moss_worker_uses_split_placement_on_16gb_gpu(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 16.0) == "split"
    assert pipe.mode is None


def test_moss_worker_falls_back_to_sequential_on_small_gpu(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 10.0) == "sequential"
    assert pipe.mode == "sequential"


def test_moss_worker_accepts_explicit_sequential_override(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "sequential")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 16.0) == "sequential"
    assert pipe.mode == "sequential"


def _write_component_snapshot(root: Path, *, shards: bool = True) -> None:
    (root / "text_encoder").mkdir(parents=True)
    (root / "transformer").mkdir()
    (root / "tokenizer").mkdir()
    (root / "vae").mkdir()
    (root / "pipeline_moss_image.py").write_text("", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    weight_map = {"a": "model-00001-of-00004.safetensors"}
    if shards:
        weight_map["b"] = "model-00002-of-00004.safetensors"
    (root / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    for shard in set(weight_map.values()):
        (root / "text_encoder" / shard).write_bytes(b"")


def test_component_resolution_skips_incomplete_cache_for_complete_one(
    tmp_path, monkeypatch
):
    primary = tmp_path / "external-hub"
    incomplete = (
        primary
        / "models--OpenMOSS-Team--MOSS-Image-Base"
        / "snapshots"
        / "incomplete"
    )
    incomplete.mkdir(parents=True)
    (incomplete / "pipeline_moss_image.py").write_text("", encoding="utf-8")

    home = tmp_path / "home"
    complete = (
        home
        / ".cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/complete"
    )
    _write_component_snapshot(complete)

    monkeypatch.setenv("HF_HUB_CACHE", str(primary))
    monkeypatch.setenv("HOME", str(home))
    for env in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        monkeypatch.delenv(env, raising=False)

    assert _snapshot_is_complete(complete) is True
    assert _snapshot_is_complete(incomplete) is False
    assert _resolve_component_source("OpenMOSS-Team/MOSS-Image-Base") == str(complete)
    assert _resolve_component_source(str(complete)) == str(complete)


def test_component_snapshot_requires_every_text_encoder_shard(tmp_path):
    snapshot = tmp_path / "snapshots" / "broken"
    _write_component_snapshot(snapshot)
    (snapshot / "text_encoder" / "model-00002-of-00004.safetensors").unlink()

    assert _snapshot_is_complete(snapshot) is False


def test_moss_worker_uses_model_offload_when_components_fit(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_MOSS_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    assert _configure_memory(pipe, "cuda", 32.0) == "model"
    assert pipe.mode == "model"


class _TinyProj(nn.Module):
    def __init__(self, dim: int = 8, hidden: int = 12):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ffn_down = nn.Linear(dim, hidden, bias=False)
        self.ffn_up_gate = nn.Linear(hidden, hidden, bias=False)


class _LoraTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].proj = _TinyProj()


def _lora_delta(down: torch.Tensor, up: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (up.float() @ down.float()) * scale


def test_apply_moss_loras_merges_scaled_delta(tmp_path):
    from core.backends.moss_image_worker import apply_moss_loras

    torch.manual_seed(0)
    transformer = _LoraTransformer()
    before = transformer.layers[0].proj.q.weight.detach().clone()
    down = torch.randn(2, 8)
    up = torch.randn(8, 2)
    path = tmp_path / "moss.safetensors"
    save_file(
        {
            "diffusion_model.layers.0.proj.q.lora_A.weight": down,
            "diffusion_model.layers.0.proj.q.lora_B.weight": up,
        },
        str(path),
    )

    applied = apply_moss_loras(
        transformer, [{"name": "tiny", "path": str(path), "weight": 0.5}]
    )

    expected = before + _lora_delta(down, up, 0.5)
    assert torch.allclose(transformer.layers[0].proj.q.weight, expected, atol=1e-6)
    assert applied[0]["modules"] == 1
    assert applied[0]["weight"] == 0.5


def test_apply_moss_loras_honours_kohya_alpha(tmp_path):
    from core.backends.moss_image_worker import apply_moss_loras

    torch.manual_seed(1)
    transformer = _LoraTransformer()
    before = transformer.layers[0].proj.o.weight.detach().clone()
    down = torch.randn(4, 8)
    up = torch.randn(8, 4)
    path = tmp_path / "kohya.safetensors"
    save_file(
        {
            "diffusion_model.layers.0.proj.o.lora_down.weight": down,
            "diffusion_model.layers.0.proj.o.lora_up.weight": up,
            "diffusion_model.layers.0.proj.o.alpha": torch.tensor(2.0),
        },
        str(path),
    )

    apply_moss_loras(transformer, [{"name": "kohya", "path": str(path), "weight": 1.0}])

    expected = before + _lora_delta(down, up, 2.0 / 4.0)
    assert torch.allclose(transformer.layers[0].proj.o.weight, expected, atol=1e-6)


def test_apply_moss_loras_splits_fused_qkv_adapters(tmp_path):
    from core.backends.moss_image_worker import apply_moss_loras

    torch.manual_seed(2)
    transformer = _LoraTransformer()
    proj = transformer.layers[0].proj
    before = {
        leaf: getattr(proj, leaf).weight.detach().clone() for leaf in ("q", "k", "v")
    }
    down = torch.randn(2, 8)
    up = torch.randn(24, 2)
    path = tmp_path / "fused.safetensors"
    save_file(
        {
            "diffusion_model.layers.0.proj.qkv.lora_A.weight": down,
            "diffusion_model.layers.0.proj.qkv.lora_B.weight": up,
        },
        str(path),
    )

    applied = apply_moss_loras(
        transformer, [{"name": "fused", "path": str(path), "weight": 1.0}]
    )

    delta = _lora_delta(down, up)
    for offset, leaf in enumerate(("q", "k", "v")):
        expected = before[leaf] + delta.chunk(3, dim=0)[offset]
        assert torch.allclose(getattr(proj, leaf).weight, expected, atol=1e-6)
    assert applied[0]["modules"] == 3
    assert applied[0]["fused_qkv_modules"] == 1


def test_apply_moss_loras_rejects_unrelated_layout(tmp_path):
    import pytest

    from core.backends.moss_image_worker import apply_moss_loras

    path = tmp_path / "sdxl.safetensors"
    save_file(
        {
            "lora_unet_down_blocks_0_attn1_to_q.lora_down.weight": torch.randn(4, 8),
            "lora_unet_down_blocks_0_attn1_to_q.lora_up.weight": torch.randn(8, 4),
        },
        str(path),
    )

    with pytest.raises(RuntimeError, match="does not match this checkpoint"):
        apply_moss_loras(
            _LoraTransformer(), [{"name": "sdxl", "path": str(path), "weight": 1.0}]
        )


def test_detect_lora_arch_reads_moss_metadata(tmp_path):
    from models.registry import detect_lora_arch

    path = tmp_path / "moss-meta.safetensors"
    save_file(
        {"diffusion_model.layers.0.proj.q.lora_A.weight": torch.zeros(2, 8)},
        str(path),
        metadata={"ss_base_model_version": "moss_image"},
    )
    assert detect_lora_arch(path) == "moss_image"


def test_detect_lora_arch_recognizes_moss_key_layout(tmp_path):
    from models.registry import detect_lora_arch

    path = tmp_path / "moss-keys.safetensors"
    save_file(
        {"diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight": torch.zeros(2, 12)},
        str(path),
    )
    assert detect_lora_arch(path) == "moss_image"


def test_moss_descriptor_advertises_lora_capability():
    caps = capabilities_for_architecture("moss_image")
    assert caps.text2img is True
    assert caps.lora is True


def test_moss_backend_rejects_foreign_lora_architecture(tmp_path, monkeypatch):
    import pytest

    import models.registry as registry
    from core.backends.moss_image import _resolve_loras

    lora = tmp_path / "foreign.safetensors"
    lora.write_bytes(b"")
    monkeypatch.setattr(
        registry,
        "LORA_REGISTRY",
        {"sdxl-lora": {"path": str(lora), "arch": "sdxl"}},
    )

    with pytest.raises(ValueError, match="not MOSS-Image"):
        _resolve_loras([{"name": "sdxl-lora", "weight": 1.0}])


def test_moss_backend_resolves_lora_and_injects_trigger(tmp_path, monkeypatch):
    import models.registry as registry
    from core.backends.moss_image import _inject_lora_triggers, _resolve_loras

    lora = tmp_path / "moss.safetensors"
    lora.write_bytes(b"")
    monkeypatch.setattr(
        registry,
        "LORA_REGISTRY",
        {
            "moss-lora": {
                "path": str(lora),
                "arch": "moss_image",
                "trigger": "mossref",
            }
        },
    )

    resolved = _resolve_loras([{"name": "moss-lora", "weight": 0.7}])
    assert resolved[0]["path"] == str(lora)
    assert resolved[0]["weight"] == 0.7
    assert _inject_lora_triggers("a red fox", resolved) == "a red fox, mossref"
    assert _inject_lora_triggers("a mossref fox", resolved) == "a mossref fox"
