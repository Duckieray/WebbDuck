from __future__ import annotations

from pathlib import Path

from core.backends.flux import FluxDiffusersBackend
from core.backends.krea2 import Krea2DiffusersBackend
from core.backends.qwen_image import QwenImageDiffusersBackend
from core.backends.runtime_probe import probe_python_runtime
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

    monkeypatch.setattr("core.backends.flux.probe_python_runtime", fake_probe)
    monkeypatch.setattr("core.backends.krea2.probe_python_runtime", fake_probe)
    monkeypatch.setattr("core.backends.qwen_image.probe_python_runtime", fake_probe)
    monkeypatch.setenv("WEBBDUCK_FLUX_PYTHON", "/runtime/flux/python")
    monkeypatch.setenv("WEBBDUCK_KREA2_PYTHON", "/runtime/krea/python")
    monkeypatch.setenv("WEBBDUCK_QWEN_IMAGE_PYTHON", "/runtime/qwen/python")

    assert FluxDiffusersBackend().readiness(_descriptor("flux", "flux", "flux_diffusers"))["ready"]
    assert Krea2DiffusersBackend().readiness(_descriptor("krea", "krea2", "krea2_diffusers"))["ready"]
    assert QwenImageDiffusersBackend().readiness(_descriptor("qwen", "qwen_image", "qwen_image_diffusers"))["ready"]

    assert captured[0] == ("/runtime/flux/python", (("diffusers", "Flux2KleinPipeline"),))
    assert ("diffusers", "Krea2Pipeline") in captured[1][1]
    assert ("diffusers", "Krea2Transformer2DModel") in captured[1][1]
    assert captured[2] == ("/runtime/qwen/python", (("diffusers", "QwenImagePipeline"),))


def test_image_isolated_requirements_use_stable_diffusers():
    root = Path(__file__).resolve().parents[1] / "runtime_requirements"
    for filename in ("flux.txt", "krea2.txt", "qwen_image.txt"):
        text = (root / filename).read_text(encoding="utf-8")
        assert "diffusers==0.39.0" in text
        assert "git+https://github.com/huggingface/diffusers" not in text
