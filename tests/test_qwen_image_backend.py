from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.backends.qwen_image import QwenImageDiffusersBackend
from core.backends.qwen_image_worker import _configure_memory
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture, defaults_for_architecture


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen/Qwen-Image-2512",
        path="/cache/qwen-image-2512",
        format="diffusers",
        source="hf_cache",
        architecture="qwen_image",
        backend="qwen_image_diffusers",
        capabilities=capabilities_for_architecture("qwen_image"),
        defaults=defaults_for_architecture("qwen_image"),
        constraints={"dimension_multiple": 16},
    )


def test_qwen_2512_descriptor_advertises_official_t2i_workflow_only():
    descriptor = _descriptor()
    assert descriptor.supported is True
    assert descriptor.capabilities.text2img is True
    assert descriptor.capabilities.img2img is False
    assert descriptor.capabilities.inpaint is False
    assert descriptor.capabilities.outpaint is False
    assert descriptor.capabilities.negative_prompt is True
    assert descriptor.defaults["width"] == 1328
    assert descriptor.defaults["height"] == 1328
    assert descriptor.defaults["steps"] == 50
    assert descriptor.defaults["cfg"] == 4.0


def test_qwen_backend_serializes_t2i_request_and_preserves_zero_seed(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "qwen_image_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 0}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.qwen_image.subprocess.Popen", FakeProcess)
    backend = QwenImageDiffusersBackend()
    images, seed = backend.generate(
        _descriptor(),
        {
            "prompt": "a rubber duck holding a sign that says WebbDuck",
            "width": 1328,
            "height": 1328,
            "steps": 50,
            "cfg": 4.0,
            "num_images": 1,
            "seed": 0,
        },
    )

    assert seed == 0
    assert len(images) == 1
    assert captured["payload"]["model_path"] == "/cache/qwen-image-2512"
    assert captured["payload"]["seed"] == 0
    assert captured["payload"]["true_cfg_scale"] == 4.0
    assert captured["payload"]["width"] == 1328
    assert "input_image" not in captured["payload"]
    assert "mask_image" not in captured["payload"]


class _FakePipe:
    def __init__(self):
        self.mode = None

    def enable_sequential_cpu_offload(self, **_kwargs):
        self.mode = "sequential"

    def enable_model_cpu_offload(self, **_kwargs):
        self.mode = "model"

    def to(self, device):
        self.mode = str(device)


def test_qwen_worker_uses_sequential_offload_on_16gb_gpu(monkeypatch):
    monkeypatch.setenv("WEBBDUCK_QWEN_IMAGE_OFFLOAD", "auto")
    pipe = _FakePipe()
    mode = _configure_memory(pipe, "cuda", 16.0)
    assert mode == "sequential"
    assert pipe.mode == "sequential"
