from __future__ import annotations

import inspect
from pathlib import Path

import core.backends.sdxl as sdxl_backend


def test_sdxl_backend_routes_through_isolated_worker():
    source = inspect.getsource(sdxl_backend.SDXLDiffusersBackend)
    assert "WEBBDUCK_SDXL_PYTHON" in inspect.getsource(sdxl_backend._runtime_python)
    assert "sdxl_worker.py" in source
    assert "subprocess.Popen" in source
    assert "from core.backends.sdxl_runtime import" not in source
    assert "from core.backends.sdxl_pipeline import" not in source


def test_host_pipeline_module_is_lightweight_worker_gate():
    root = Path(__file__).resolve().parents[1]
    generation = (root / "core" / "generation.py").read_text(encoding="utf-8")
    pipeline = (root / "core" / "pipeline.py").read_text(encoding="utf-8")

    assert "from core.backends.sdxl_runtime import *" in generation
    assert "WEBBDUCK_SDXL_WORKER" in pipeline
    assert "if str(os.getenv" in pipeline
    assert "from core.backends.sdxl_pipeline import *" in pipeline
    assert "class _IsolatedPipelineManager" in pipeline


def test_backend_owned_sdxl_implementations_are_present():
    root = Path(__file__).resolve().parents[1] / "core" / "backends"
    assert (root / "sdxl_runtime.py").stat().st_size > 10_000
    assert (root / "sdxl_pipeline.py").stat().st_size > 30_000
    worker = (root / "sdxl_worker.py").read_text(encoding="utf-8")
    assert "WEBBDUCK_SDXL_WORKER" in worker
    assert "metadata_updates" in worker
    assert "progress" in worker


def test_sdxl_execution_registry_is_hydrated_from_canonical_catalog(monkeypatch, tmp_path):
    import models.catalog as catalog
    import models.registry as registry

    sdxl_path = tmp_path / "new-sdxl"
    flux_path = tmp_path / "flux"
    canonical = {
        "New SDXL": {
            "type": "diffusers",
            "arch": "sdxl",
            "path": sdxl_path,
            "source": "local",
            "defaults": {"steps": 31},
        },
        "FLUX": {
            "type": "diffusers",
            "arch": "flux",
            "path": flux_path,
            "source": "local",
        },
    }
    monkeypatch.setattr(catalog, "runtime_registry", lambda: canonical)
    monkeypatch.setattr(
        registry,
        "MODEL_REGISTRY",
        {
            "Stale SDXL": {"arch": "sdxl", "path": tmp_path / "stale"},
            "Internal Other": {"arch": "other", "path": tmp_path / "other"},
        },
    )

    sdxl_backend._hydrate_execution_registry()

    assert "Stale SDXL" not in registry.MODEL_REGISTRY
    assert registry.MODEL_REGISTRY["New SDXL"]["path"] == sdxl_path
    assert registry.MODEL_REGISTRY["New SDXL"]["defaults"]["steps"] == 31
    assert "FLUX" not in registry.MODEL_REGISTRY
    assert "Internal Other" in registry.MODEL_REGISTRY
