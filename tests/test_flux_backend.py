from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.backends.flux import FluxDiffusersBackend, _worker_environment
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture, defaults_for_architecture


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="black-forest-labs/FLUX.2-klein-4B",
        path="/cache/flux",
        format="diffusers",
        source="hf_cache",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=capabilities_for_architecture("flux"),
        defaults=defaults_for_architecture("flux"),
        constraints={"dimension_multiple": 8},
    )


def _gguf_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="flux-2-klein-9b-Q5_K_M",
        path="/models/flux/flux-2-klein-9b-Q5_K_M.gguf",
        format="gguf",
        source="local",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=capabilities_for_architecture("flux"),
        defaults=defaults_for_architecture("flux"),
        constraints={"dimension_multiple": 8},
        detection={
            "method": "gguf_filename",
            "family": "flux2_klein",
            "variant": "distilled",
            "parameter_size": "9B",
            "quantization": "Q5_K_M",
            "component_model": "black-forest-labs/FLUX.2-klein-9B",
        },
    )


def test_flux_descriptor_exposes_current_workflows_and_defaults():
    descriptor = _descriptor()
    assert descriptor.supported is True
    assert descriptor.capabilities.text2img is True
    assert descriptor.capabilities.img2img is True
    assert descriptor.capabilities.inpaint is False
    assert descriptor.capabilities.lora is False
    assert descriptor.defaults["steps"] == 4
    assert descriptor.defaults["cfg"] == 1.0
    assert descriptor.constraints["dimension_multiple"] == 8


def test_flux_backend_serializes_request_and_reads_images(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "flux_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 123}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.flux.subprocess.Popen", FakeProcess)
    backend = FluxDiffusersBackend()
    images, seed = backend.generate(
        _descriptor(),
        {
            "prompt": "a rubber duck astronaut",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg": 1.0,
            "num_images": 1,
            "seed": 123,
        },
    )

    assert seed == 123
    assert len(images) == 1
    assert images[0].size == (16, 16)
    assert captured["payload"]["model_path"] == "/cache/flux"
    assert captured["payload"]["model_format"] == "diffusers"
    assert captured["payload"]["steps"] == 4
    assert captured["payload"]["guidance"] == 1.0


def test_flux_backend_serializes_gguf_component_contract(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            captured["env"] = dict(kwargs.get("env") or {})
            image_path = output_dir / "flux_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "images": [str(image_path)],
                        "seed": 0,
                        "runtime": {
                            "model_format": "gguf",
                            "quantization": "Q5_K_M",
                            "offload_mode": "model_cpu",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.flux._preflight_component_access", lambda _model: None)
    monkeypatch.setattr(
        "core.backends.flux.resolve_provider_token",
        lambda _provider: ("hf_test_token", "settings"),
    )
    monkeypatch.setattr("core.backends.flux.subprocess.Popen", FakeProcess)
    backend = FluxDiffusersBackend()
    settings = {
        "prompt": "a rubber duck astronaut",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 1.0,
        "num_images": 1,
        "seed": 0,
    }
    images, seed = backend.generate(_gguf_descriptor(), settings)

    assert seed == 0
    assert len(images) == 1
    assert captured["payload"]["model_format"] == "gguf"
    assert captured["payload"]["component_model"] == "black-forest-labs/FLUX.2-klein-9B"
    assert captured["payload"]["detection"]["quantization"] == "Q5_K_M"
    assert captured["env"]["HF_TOKEN"] == "hf_test_token"
    assert captured["env"]["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
    assert settings["flux_runtime"]["offload_mode"] == "model_cpu"


def test_flux_worker_environment_propagates_settings_token(monkeypatch):
    monkeypatch.setattr(
        "core.backends.flux.resolve_provider_token",
        lambda _provider: ("hf_settings_token", "settings"),
    )

    env = _worker_environment()

    assert env["HF_TOKEN"] == "hf_settings_token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_settings_token"


def test_flux_worker_uses_explicit_klein_config_for_gguf():
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert "Flux2Transformer2DModel.from_single_file" in worker_source
    assert "GGUFQuantizationConfig" in worker_source
    assert 'config=component_model' in worker_source
    assert 'subfolder="transformer"' in worker_source
