from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.backends.flux import FluxDiffusersBackend
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
    assert captured["payload"]["steps"] == 4
    assert captured["payload"]["guidance"] == 1.0
