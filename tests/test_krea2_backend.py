from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from core.backends.krea2 import (
    Krea2DiffusersBackend,
    _component_access_error,
    _forward_worker_progress,
    _preflight_component_access,
    _runtime_python,
    _worker_error,
)
from core.backends.krea2_worker import (
    ScaledFP8Linear,
    _auto_execution_mode,
    _configure_execution,
    _encode_prompt_phase,
    _install_scaled_fp8_linear,
    _write_progress,
)
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture, defaults_for_model


def _descriptor() -> ModelDescriptor:
    detection = {"variant": "turbo"}
    return ModelDescriptor(
        name="krea/Krea-2-Turbo",
        path="/cache/krea2",
        format="diffusers",
        source="hf_cache",
        architecture="krea2",
        backend="krea2_diffusers",
        capabilities=capabilities_for_architecture("krea2"),
        defaults=defaults_for_model("krea2", "krea/Krea-2-Turbo", detection),
        constraints={"dimension_multiple": 16},
        detection=detection,
    )


def test_krea2_descriptor_matches_turbo_contract():
    descriptor = _descriptor()
    assert descriptor.supported is True
    assert descriptor.capabilities.text2img is True
    assert descriptor.capabilities.img2img is False
    assert descriptor.capabilities.lora is False
    assert descriptor.defaults["width"] == 1024
    assert descriptor.defaults["height"] == 1024
    assert descriptor.defaults["steps"] == 8
    assert descriptor.defaults["cfg"] == 0.0
    assert descriptor.constraints["dimension_multiple"] == 16


def test_krea2_default_runtime_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.delenv("WEBBDUCK_KREA2_PYTHON", raising=False)
    monkeypatch.setenv("WEBBDUCK_RUNTIME_HOME", str(tmp_path))
    expected = tmp_path / "krea2" / "bin" / "python"
    assert Path(_runtime_python()) == expected


def test_krea2_gated_component_error_is_actionable_without_traceback_noise():
    error = _worker_error(
        {
            "error": (
                "krea/Krea-2-Raw is not a local folder and is not a valid model identifier; "
                "make sure to pass a token"
            ),
            "traceback": "Traceback (most recent call last): very long gated stack",
        },
        [],
        1,
    )
    message = str(error)
    assert "Krea 2 support components are gated on Hugging Face" in message
    assert "Community License" in message
    assert "WebbDuck Settings" in message
    assert "very long gated stack" not in message


def test_krea2_component_access_error_distinguishes_authorized_token_failure():
    error = _component_access_error(
        "krea/Krea-2-Raw",
        RuntimeError("403 Client Error: gated repo restricted"),
        token_configured=True,
    )
    message = str(error)
    assert "configured token is not authorized" in message
    assert "krea/Krea-2-Raw" in message
    assert "does not need to be re-downloaded" in message


def test_krea2_component_preflight_accepts_complete_local_override(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    _preflight_component_access(str(tmp_path))


def test_krea2_component_preflight_rejects_incomplete_local_override(tmp_path):
    (tmp_path / "transformer").mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        _preflight_component_access(str(tmp_path))
    assert "component directory is incomplete" in str(excinfo.value)
    assert "model_index.json" in str(excinfo.value)


def test_krea2_progress_snapshot_round_trips_to_host_callback(tmp_path):
    progress_path = tmp_path / "progress.json"
    _write_progress(
        progress_path,
        "Denoising with Krea 2",
        0.75,
        step=6,
        total_steps=8,
    )
    received = []
    raw = _forward_worker_progress(
        progress_path,
        lambda stage, progress, step, total: received.append((stage, progress, step, total)),
        None,
    )
    assert raw is not None
    assert received == [("Denoising with Krea 2", 0.75, 6, 8)]
    assert _forward_worker_progress(progress_path, lambda *_args: received.append("duplicate"), raw) == raw
    assert "duplicate" not in received


def test_scaled_fp8_linear_keeps_quantized_storage_but_matches_dequantized_math():
    qweight = torch.tensor([[2.0, -4.0], [6.0, 8.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor(0.25, dtype=torch.float32)
    layer = ScaledFP8Linear(2, 2, qweight=qweight, scale=scale)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    actual = layer(x)
    expected = F.linear(x, qweight.to(torch.bfloat16) * scale.to(torch.bfloat16))
    assert layer.qweight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual, expected)


def test_scaled_fp8_overlay_replaces_linear_without_expanding_full_weight():
    model = nn.Sequential(nn.Linear(2, 3, bias=True))
    original_bias = model[0].bias.detach().clone()
    qweight = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float8_e4m3fn,
    )
    assert _install_scaled_fp8_linear(
        model,
        "0.weight",
        qweight,
        torch.tensor(0.5),
        None,
    )
    assert isinstance(model[0], ScaledFP8Linear)
    assert model[0].qweight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(model[0].bias, original_bias)


def test_krea2_16gb_scaled_fp8_auto_path_is_resident():
    assert _auto_execution_mode(preserved_fp8_linears=100, total_vram_gb=16.0) == "resident-fp8"
    assert _auto_execution_mode(preserved_fp8_linears=0, total_vram_gb=16.0) == "transformer-block"


def test_krea2_resident_fp8_moves_only_vae_and_transformer_to_gpu(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "auto")
    moved = []

    class FakeModule:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            moved.append((self.name, str(device)))
            return self

    class FakePipe:
        vae = FakeModule("vae")
        transformer = FakeModule("transformer")

    mode = _configure_execution(
        FakePipe(),
        device="cuda",
        total_vram_gb=16.0,
        load_info={"preserved_fp8_linears": 100},
    )
    assert mode == "resident-fp8"
    assert ("vae", "cuda") in moved
    assert ("transformer", "cuda") in moved


def test_krea2_dense_16gb_auto_uses_block_group_offload(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "auto")
    monkeypatch.setenv("WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM", "0")
    captured = {}

    class FakeVAE:
        def to(self, device):
            captured["vae_device"] = str(device)
            return self

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            captured.update(kwargs)

    class FakePipe:
        vae = FakeVAE()
        transformer = FakeTransformer()

    mode = _configure_execution(
        FakePipe(),
        device="cuda",
        total_vram_gb=16.0,
        load_info={"preserved_fp8_linears": 0},
    )
    assert mode == "transformer-block-stream-2"
    assert captured["vae_device"] == "cuda"
    assert captured["offload_type"] == "block_level"
    assert captured["num_blocks_per_group"] == 2
    assert captured["use_stream"] is True
    assert captured["record_stream"] is True
    assert captured["low_cpu_mem_usage"] is False


def test_krea2_block_offload_falls_back_to_transformer_leaf(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "block")
    calls = []

    class FakeVAE:
        def to(self, _device):
            return self

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            calls.append(kwargs)
            if kwargs["offload_type"] == "block_level":
                raise RuntimeError("unsupported block grouping")

    class FakePipe:
        vae = FakeVAE()
        transformer = FakeTransformer()

    mode = _configure_execution(
        FakePipe(),
        device="cuda",
        total_vram_gb=16.0,
        load_info={"preserved_fp8_linears": 0},
    )
    assert mode == "transformer-leaf-stream-fallback"
    assert [call["offload_type"] for call in calls] == ["block_level", "leaf_level"]


def test_krea2_prompt_encoding_is_one_gpu_phase_then_encoder_returns_to_cpu(monkeypatch):
    moved = []
    encoded = []

    class FakeEncoder:
        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

        def to(self, device):
            moved.append(str(device))
            return self

    class FakePipe:
        text_encoder = FakeEncoder()

        def encode_prompt(self, *, prompt, device, **_kwargs):
            encoded.append((prompt, str(device)))
            return torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    positive, mask, negative, negative_mask = _encode_prompt_phase(
        FakePipe(),
        prompt="duck",
        guidance=4.5,
        device="cuda",
    )
    assert moved == ["cuda", "cpu"]
    assert encoded == [("duck", "cuda"), ("", "cuda")]
    assert positive.shape == (1, 2, 3)
    assert mask.shape == (1, 2)
    assert negative is not None
    assert negative_mask is not None


def test_krea2_backend_serializes_request_reads_progress_and_runtime(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            progress_path = Path(cmd[cmd.index("--progress") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            progress_path.write_text(
                json.dumps(
                    {
                        "stage": "Denoising with Krea 2",
                        "progress": 0.75,
                        "step": 6,
                        "total_steps": 8,
                    }
                ),
                encoding="utf-8",
            )
            image_path = output_dir / "krea2_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "images": [str(image_path)],
                        "seed": 321,
                        "timing": {
                            "pipeline_load_seconds": 10.0,
                            "prompt_encode_seconds": 2.0,
                            "denoise_decode_seconds": 18.0,
                            "inference_seconds": 20.0,
                        },
                        "runtime": {
                            "offload": "resident-fp8",
                            "device": "cuda",
                            "execution_quantization": "scaled-fp8",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.krea2.subprocess.Popen", FakeProcess)
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    backend = Krea2DiffusersBackend()
    settings = {
        "prompt": "a cinematic rubber duck",
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "cfg": 0.0,
        "num_images": 1,
        "seed": 321,
    }
    progress_events = []
    images, seed = backend.generate(
        _descriptor(),
        settings,
        progress_callback=lambda stage, progress, step, total: progress_events.append(
            (stage, progress, step, total)
        ),
    )

    assert seed == 321
    assert len(images) == 1
    assert images[0].size == (16, 16)
    assert captured["cmd"][0] == "/runtime/krea/python"
    assert "--progress" in captured["cmd"]
    assert captured["payload"]["model_path"] == "/cache/krea2"
    assert captured["payload"]["model_format"] == "diffusers"
    assert captured["payload"]["variant"] == "turbo"
    assert captured["payload"]["component_source"] == "krea/Krea-2-Turbo"
    assert captured["payload"]["steps"] == 8
    assert captured["payload"]["guidance"] == 0.0
    assert ("Denoising with Krea 2", 0.75, 6, 8) in progress_events
    assert settings["performance_timing"]["krea_pipeline_load_seconds"] == 10.0
    assert settings["performance_timing"]["krea_prompt_encode_seconds"] == 2.0
    assert settings["performance_timing"]["krea_denoise_decode_seconds"] == 18.0
    assert settings["performance_timing"]["krea_inference_seconds"] == 20.0
    assert settings["krea_runtime"]["offload"] == "resident-fp8"
    assert settings["krea_runtime"]["execution_quantization"] == "scaled-fp8"
