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
    assert descriptor.capabilities.inpaint is True
    assert descriptor.capabilities.outpaint is True
    assert descriptor.capabilities.identity_adapter is True
    assert descriptor.capabilities.lora is True
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
    assert captured["payload"]["loras"] == []


def test_flux_backend_prepares_native_outpaint_and_locks_source(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            # Simulate FLUX's internal source resize / OOM demotion result.
            Image.new("RGB", (16, 8), "red").save(output_dir / "flux_000.png")
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(output_dir / "flux_000.png")], "seed": 4}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "blue").save(source)
    monkeypatch.setattr("core.backends.flux.subprocess.Popen", FakeProcess)

    images, _seed = FluxDiffusersBackend().generate(
        _descriptor(),
        {
            "prompt": "extend a blue square",
            "image": str(source),
            "smart_extend": True,
            "smart_extend_anchor": "left",
            "width": 32,
            "height": 16,
            "steps": 4,
            "cfg": 1.0,
            "seed": 4,
        },
    )

    assert captured["payload"]["input_image"].endswith("input.png")
    assert captured["payload"]["mask_image"].endswith("mask.png")
    assert captured["payload"]["width"] == 32
    assert captured["payload"]["height"] == 16
    assert images[0].size == (16, 8)
    assert images[0].getpixel((0, 4)) == (0, 0, 255)
    assert images[0].getpixel((15, 4)) == (255, 0, 0)


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
    assert captured["payload"]["loras"] == []
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


def test_flux_worker_normalizes_comfy_flux2_single_file_prefix():
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert 'prefix = "model.diffusion_model."' in worker_source
    assert "key.removeprefix(prefix): value" in worker_source
    assert "Flux2Transformer2DModel.from_single_file(\n            checkpoint," in worker_source


def test_flux_worker_phases_qwen_prompt_encoding_on_constrained_9b_gpus():
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert "_encode_prompt_on_cpu" in worker_source
    assert "total_vram_gb < 20.0" in worker_source
    assert "free_vram_gb < 19.0" in worker_source
    assert 'kwargs["prompt_embeds"] = prompt_embeds_cpu.to(device=device, dtype=dtype)' in worker_source
    assert 'kwargs["prompt"] = None' in worker_source
    assert '"phased_prompt_encoding": phased_prompt_encoding' in worker_source


def test_flux_worker_uses_native_mask_blur_for_inpaint():
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert 'request.get("mask_blur")' in worker_source
    assert "pipe.mask_processor.blur(mask_image, blur_factor=mask_blur)" in worker_source


def test_flux_worker_adapts_qwen_padding_to_real_prompt_length():
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert "WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH" in worker_source
    assert "token_count + 16" in worker_source
    assert "for bucket in (64, 96, 128, 192, 256, 384, 512)" in worker_source


def test_flux_lora_helper_uses_live_weighted_adapters_without_fusion():
    helper_source = Path("core/backends/flux_lora.py").read_text(encoding="utf-8")
    worker_source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert "load_lora_weights" in helper_source
    assert "set_adapters" in helper_source
    assert "adapter_weights=adapter_weights" in helper_source
    assert "fuse_lora(" not in helper_source
    assert "apply_flux_loras" in worker_source
    assert '"lora_count": len(applied_loras)' in worker_source
