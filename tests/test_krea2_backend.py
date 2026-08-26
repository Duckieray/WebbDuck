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
    _activation_reserve_gb,
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


def _cuda_hardware(*, free: float = 15.0, total: float = 16.0, fp8: bool = True) -> dict:
    return {
        "accelerator": "cuda",
        "vendor": "nvidia",
        "name": "Test GPU",
        "free_vram_gb": free,
        "total_vram_gb": total,
        "bf16": True,
        "fp8_storage": fp8,
        "stream_prefetch": True,
        "memory_tier": "constrained",
    }


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
    _write_progress(progress_path, "Denoising with Krea 2", 0.75, step=6, total_steps=8)
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
    assert _install_scaled_fp8_linear(model, "0.weight", qweight, torch.tensor(0.5), None)
    assert isinstance(model[0], ScaledFP8Linear)
    assert model[0].qweight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(model[0].bias, original_bias)


def test_krea_activation_reserve_scales_with_resolution():
    reserve_1k = _activation_reserve_gb(16.0, 1024, 1024)
    reserve_2k = _activation_reserve_gb(16.0, 2048, 2048)
    assert reserve_1k >= 2.0
    assert reserve_2k > reserve_1k


def test_krea_auto_profile_uses_live_free_vram_not_only_card_capacity():
    resident = _auto_execution_mode(
        preserved_fp8_linears=100,
        hardware=_cuda_hardware(free=15.0),
        transformer_storage_gb=12.0,
        reserve_gb=2.0,
    )
    constrained = _auto_execution_mode(
        preserved_fp8_linears=100,
        hardware=_cuda_hardware(free=13.0),
        transformer_storage_gb=12.0,
        reserve_gb=2.0,
    )
    assert resident == "resident-fp8"
    assert constrained == "transformer-block"


def test_krea_auto_profile_does_not_use_fp8_resident_when_device_lacks_fp8_storage():
    mode = _auto_execution_mode(
        preserved_fp8_linears=100,
        hardware=_cuda_hardware(free=15.0, fp8=False),
        transformer_storage_gb=12.0,
        reserve_gb=2.0,
    )
    assert mode == "transformer-block"


def test_krea_dense_model_can_be_resident_on_large_free_vram():
    mode = _auto_execution_mode(
        preserved_fp8_linears=0,
        hardware=_cuda_hardware(free=34.0, total=40.0),
        transformer_storage_gb=26.0,
        reserve_gb=4.0,
    )
    assert mode == "resident"


def test_krea_cuda_block_profile_uses_streamed_single_block_groups(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "auto")
    captured = {}

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            captured.update(kwargs)

        def to(self, _device):
            return self

    class FakePipe:
        transformer = FakeTransformer()

    mode = _configure_execution(
        FakePipe(),
        hardware=_cuda_hardware(free=10.0),
        load_info={"preserved_fp8_linears": 100},
        transformer_storage_gb=12.0,
        reserve_gb=2.0,
    )
    assert mode == "transformer-block-stream-1"
    assert captured["offload_type"] == "block_level"
    assert captured["num_blocks_per_group"] == 1
    assert captured["use_stream"] is True
    assert captured["record_stream"] is True


def test_krea_rocm_block_profile_uses_conservative_sync_offload(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "auto")
    captured = {}

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            captured.update(kwargs)

        def to(self, _device):
            return self

    class FakePipe:
        transformer = FakeTransformer()

    hardware = _cuda_hardware(free=10.0)
    hardware.update({"accelerator": "rocm", "vendor": "amd", "stream_prefetch": False})
    mode = _configure_execution(
        FakePipe(),
        hardware=hardware,
        load_info={"preserved_fp8_linears": 0},
        transformer_storage_gb=22.0,
        reserve_gb=2.0,
    )
    assert mode == "transformer-block-sync-2"
    assert captured["offload_type"] == "block_level"
    assert captured["num_blocks_per_group"] == 2
    assert captured["use_stream"] is False


def test_krea_block_offload_falls_back_to_transformer_leaf(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "block")
    calls = []

    class FakeTransformer:
        def enable_group_offload(self, **kwargs):
            calls.append(kwargs)
            if kwargs["offload_type"] == "block_level":
                raise RuntimeError("unsupported block grouping")

        def to(self, _device):
            return self

    class FakePipe:
        transformer = FakeTransformer()

    mode = _configure_execution(
        FakePipe(),
        hardware=_cuda_hardware(free=10.0),
        load_info={"preserved_fp8_linears": 0},
        transformer_storage_gb=26.0,
        reserve_gb=2.0,
    )
    assert mode == "transformer-leaf-stream-fallback"
    assert [call["offload_type"] for call in calls] == ["block_level", "leaf_level"]


def test_krea_prompt_encoding_is_one_gpu_phase_and_detaches_encoder(monkeypatch):
    moved = []
    encoded = []

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

        def to(self, device):
            moved.append(str(device))
            return self

    class FakePipe:
        def __init__(self):
            self.text_encoder = FakeEncoder()

        def encode_prompt(self, *, prompt, device, **_kwargs):
            encoded.append((prompt, str(device)))
            return torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    pipe = FakePipe()
    positive, mask, negative, negative_mask, mode = _encode_prompt_phase(
        pipe,
        prompt="duck",
        guidance=4.5,
        hardware=_cuda_hardware(free=15.0),
    )
    assert mode == "resident"
    assert moved == ["cuda"]
    assert encoded == [("duck", "cuda"), ("", "cuda")]
    assert pipe.text_encoder is None
    assert positive.device.type == "cpu"
    assert mask.device.type == "cpu"
    assert negative is not None and negative.device.type == "cpu"
    assert negative_mask is not None and negative_mask.device.type == "cpu"


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
                            "denoise_seconds": 16.0,
                            "decode_seconds": 2.0,
                            "denoise_decode_seconds": 18.0,
                            "inference_seconds": 20.0,
                        },
                        "runtime": {
                            "offload": "transformer-block-stream-1",
                            "initial_offload": "resident-fp8",
                            "fallback_reason": "resident_oom",
                            "device": "cuda",
                            "execution_quantization": "scaled-fp8",
                            "hardware": {
                                "name": "Test GPU",
                                "total_vram_gb": 16.0,
                                "free_vram_gb": 12.0,
                            },
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
    assert settings["performance_timing"]["krea_denoise_seconds"] == 16.0
    assert settings["performance_timing"]["krea_decode_seconds"] == 2.0
    assert settings["performance_timing"]["krea_inference_seconds"] == 20.0
    assert settings["krea_runtime"]["offload"] == "transformer-block-stream-1"
    assert settings["krea_runtime"]["fallback_reason"] == "resident_oom"
