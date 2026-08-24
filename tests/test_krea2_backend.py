from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from core.backends.krea2 import (
    Krea2DiffusersBackend,
    _component_access_error,
    _forward_worker_progress,
    _preflight_component_access,
    _runtime_python,
    _worker_error,
)
from core.backends.krea2_worker import _configure_offload, _write_progress
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


def test_krea2_16gb_auto_offload_prefers_streamed_leaf_groups(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "auto")
    monkeypatch.setenv("WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM", "1")
    captured = {}

    class FakePipe:
        def enable_group_offload(self, **kwargs):
            captured.update(kwargs)

        def enable_sequential_cpu_offload(self, **_kwargs):
            raise AssertionError("group offload should be preferred on the 16 GB auto path")

    mode = _configure_offload(FakePipe(), "cuda", 16.0)
    assert mode == "group-leaf-stream"
    assert captured["offload_type"] == "leaf_level"
    assert captured["use_stream"] is True
    assert captured["record_stream"] is True
    assert captured["low_cpu_mem_usage"] is True


def test_krea2_group_offload_falls_back_to_sequential(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_KREA2_OFFLOAD", "group")
    captured = {"sequential": False}

    class FakePipe:
        def enable_group_offload(self, **_kwargs):
            raise RuntimeError("unsupported component")

        def enable_sequential_cpu_offload(self, **_kwargs):
            captured["sequential"] = True

    mode = _configure_offload(FakePipe(), "cuda", 16.0)
    assert mode == "sequential-fallback"
    assert captured["sequential"] is True


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
                            "inference_seconds": 20.0,
                        },
                        "runtime": {
                            "offload": "group-leaf-stream",
                            "device": "cuda",
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
    assert settings["performance_timing"]["krea_inference_seconds"] == 20.0
    assert settings["krea_runtime"]["offload"] == "group-leaf-stream"
