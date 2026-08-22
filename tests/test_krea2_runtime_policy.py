from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core.backends.krea2 import Krea2DiffusersBackend
from core.backends.krea2_worker import _component_source, _configure_offload
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture


class _FakePipe:
    def __init__(self):
        self.calls = []

    def enable_sequential_cpu_offload(self, **kwargs):
        self.calls.append(("sequential", kwargs))

    def enable_model_cpu_offload(self, **kwargs):
        self.calls.append(("model", kwargs))

    def to(self, device):
        self.calls.append(("to", device))
        return self


def test_auto_offload_uses_sequential_policy_on_16gb(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_OFFLOAD", raising=False)
    pipe = _FakePipe()

    policy = _configure_offload(pipe, "cuda", 16.0)

    assert policy == "sequential"
    assert pipe.calls == [("sequential", {"device": "cuda"})]


def test_component_source_matches_checkpoint_variant(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_KREA2_COMPONENT_MODEL", raising=False)
    monkeypatch.delenv("WEBBDUCK_KREA2_TURBO_COMPONENT_MODEL", raising=False)
    monkeypatch.delenv("WEBBDUCK_KREA2_BASE_COMPONENT_MODEL", raising=False)

    assert _component_source("turbo") == "krea/Krea-2-Turbo"
    assert _component_source("base") == "krea/Krea-2-Raw"


def test_backend_preserves_explicit_zero_seed(monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "krea2_000.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 0}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    descriptor = ModelDescriptor(
        name="krea/Krea-2-Turbo",
        path="/cache/krea2",
        format="diffusers",
        source="hf_cache",
        architecture="krea2",
        backend="krea2_diffusers",
        capabilities=capabilities_for_architecture("krea2"),
        defaults={"width": 1024, "height": 1024, "steps": 8, "cfg": 0.0},
        constraints={"dimension_multiple": 16},
        detection={"variant": "turbo"},
    )
    monkeypatch.setattr("core.backends.krea2.subprocess.Popen", FakeProcess)

    _images, seed = Krea2DiffusersBackend().generate(
        descriptor,
        {"prompt": "duck", "seed": 0},
    )

    assert seed == 0
    assert captured["payload"]["seed"] == 0
