from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from core.backends.base import GenerationBackend
from core.backends.flux import FluxDiffusersBackend
from core.backends.krea2 import Krea2DiffusersBackend
from core.backends.qwen_image import QwenImageDiffusersBackend
from core.backends import runtime_probe
from core.backends.runtime_probe import probe_python_runtime
from core.backends.sdxl import SDXLDiffusersBackend
from models.model_descriptor import ModelDescriptor, ModelCapabilities


def _descriptor(name: str, architecture: str, backend: str) -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        path="/tmp/model",
        format="diffusers",
        source="test",
        architecture=architecture,
        backend=backend,
        capabilities=ModelCapabilities(text2img=True),
    )


def test_missing_runtime_python_fails_without_loading_weights(tmp_path):
    result = probe_python_runtime(str(tmp_path / "missing-python"), (("diffusers", "Anything"),))
    assert result["ready"] is False
    assert "does not exist" in result["reason"]


def test_isolated_backends_probe_required_pipeline_classes(monkeypatch):
    captured = []

    def fake_probe(python, symbols, **_kwargs):
        captured.append((python, tuple(symbols)))
        return {"ready": True, "cuda_available": True}

    monkeypatch.setattr("core.backends.sdxl.probe_python_runtime", fake_probe)
    monkeypatch.setattr("core.backends.flux.probe_python_runtime", fake_probe)
    monkeypatch.setattr("core.backends.krea2.probe_python_runtime", fake_probe)
    monkeypatch.setattr("core.backends.qwen_image.probe_python_runtime", fake_probe)
    monkeypatch.setenv("WEBBDUCK_SDXL_PYTHON", "/runtime/sdxl/python")
    monkeypatch.setenv("WEBBDUCK_FLUX_PYTHON", "/runtime/flux/python")
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    monkeypatch.setenv("WEBBDUCK_QWEN_IMAGE_PYTHON", "/runtime/qwen/python")

    assert SDXLDiffusersBackend().readiness(_descriptor("sdxl", "sdxl", "sdxl_diffusers"))["ready"]
    assert FluxDiffusersBackend().readiness(_descriptor("flux", "flux", "flux_diffusers"))["ready"]
    assert Krea2DiffusersBackend().readiness(_descriptor("krea", "krea2", "krea2_diffusers"))["ready"]
    assert QwenImageDiffusersBackend().readiness(_descriptor("qwen", "qwen_image", "qwen_image_diffusers"))["ready"]

    assert captured[0][0] == "/runtime/sdxl/python"
    assert ("diffusers", "StableDiffusionXLPipeline") in captured[0][1]
    assert ("diffusers", "StableDiffusionXLImg2ImgPipeline") in captured[0][1]
    assert ("diffusers", "StableDiffusionXLInpaintPipeline") in captured[0][1]
    assert captured[1] == ("/runtime/flux/python", (("diffusers", "Flux2KleinPipeline"),))
    assert ("diffusers", "Krea2Pipeline") in captured[2][1]
    assert ("diffusers", "Krea2Transformer2DModel") in captured[2][1]
    assert captured[3] == ("/runtime/qwen/python", (("diffusers", "QwenImagePipeline"),))


def test_generation_refuses_unready_runtime_before_backend_worker(monkeypatch):
    import core.model_runtime as model_runtime

    descriptor = _descriptor("known-sdxl", "sdxl", "sdxl_diffusers")
    called = {"generate": False}

    class FakeBackend(GenerationBackend):
        backend_id = "sdxl_diffusers"

        def can_handle(self, candidate):
            return candidate is descriptor

        def readiness(self, candidate):
            assert candidate is descriptor
            return {
                "ready": False,
                "python": "/runtime/sdxl/bin/python",
                "reason": "One or more required runtime imports are unavailable.",
                "missing": [
                    {
                        "module": "safetensors",
                        "symbol": "safe_open",
                        "error": "No module named 'safetensors'",
                    }
                ],
                "repair_hint": (
                    "Repair/update this isolated runtime with "
                    "`python tools/prepare_model_runtimes.py sdxl`, then restart WebbDuck."
                ),
            }

        def generate(self, candidate, settings, **kwargs):
            called["generate"] = True
            raise AssertionError("unready backend must never launch generation")

    backend = FakeBackend()
    monkeypatch.setattr(model_runtime, "descriptor_for_model", lambda _name: descriptor)
    monkeypatch.setattr(model_runtime, "register_installed_backends", lambda: None)
    monkeypatch.setattr(model_runtime.backend_resolver, "resolve", lambda _descriptor: backend)

    with pytest.raises(RuntimeError) as excinfo:
        model_runtime.run_selected_model({"base_model": descriptor.name, "prompt": "duck"})

    message = str(excinfo.value)
    assert "runtime is not ready" in message
    assert "safetensors.safe_open" in message
    assert "prepare_model_runtimes.py sdxl" in message
    assert called["generate"] is False


def test_sdxl_failed_readiness_includes_runtime_sync_hint(monkeypatch):
    monkeypatch.setattr(
        "core.backends.sdxl.probe_python_runtime",
        lambda *_args, **_kwargs: {
            "ready": False,
            "reason": "One or more required runtime imports are unavailable.",
            "missing": [{"module": "safetensors", "symbol": "safe_open"}],
        },
    )
    payload = SDXLDiffusersBackend().readiness(
        _descriptor("sdxl", "sdxl", "sdxl_diffusers")
    )
    assert payload["ready"] is False
    assert "prepare_model_runtimes.py sdxl" in payload["repair_hint"]


def test_runtime_probe_always_checks_safetensors_before_reporting_ready():
    source = inspect.getsource(runtime_probe)
    assert '_SHARED_REQUIRED_SYMBOLS = (("safetensors", "safe_open"),)' in source
    assert "_SHARED_REQUIRED_SYMBOLS" in inspect.getsource(probe_python_runtime)


def test_image_isolated_requirements_are_pinned():
    root = Path(__file__).resolve().parents[1] / "runtime_requirements"

    sdxl = (root / "sdxl.txt").read_text(encoding="utf-8")
    assert "diffusers==0.36.0" in sdxl
    assert "safetensors==0.5.3" in sdxl
    assert "git+https://github.com/huggingface/diffusers" not in sdxl

    for filename in ("flux.txt", "krea2.txt", "qwen_image.txt"):
        text = (root / filename).read_text(encoding="utf-8")
        assert "diffusers==0.39.0" in text
        assert "safetensors" in text
        assert "git+https://github.com/huggingface/diffusers" not in text
