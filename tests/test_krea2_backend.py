from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from core.backends.krea2 import (
    Krea2DiffusersBackend,
    _component_access_error,
    _preflight_component_access,
    _runtime_python,
    _worker_error,
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


def test_krea2_backend_serializes_request_and_reads_images(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "krea2_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 321}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.krea2.subprocess.Popen", FakeProcess)
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    backend = Krea2DiffusersBackend()
    images, seed = backend.generate(
        _descriptor(),
        {
            "prompt": "a cinematic rubber duck",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "cfg": 0.0,
            "num_images": 1,
            "seed": 321,
        },
    )

    assert seed == 321
    assert len(images) == 1
    assert images[0].size == (16, 16)
    assert captured["cmd"][0] == "/runtime/krea/python"
    assert captured["payload"]["model_path"] == "/cache/krea2"
    assert captured["payload"]["model_format"] == "diffusers"
    assert captured["payload"]["variant"] == "turbo"
    assert captured["payload"]["component_source"] == "krea/Krea-2-Turbo"
    assert captured["payload"]["steps"] == 8
    assert captured["payload"]["guidance"] == 0.0
