from __future__ import annotations

import inspect
from pathlib import Path

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
