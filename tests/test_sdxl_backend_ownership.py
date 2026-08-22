from __future__ import annotations

import inspect
from pathlib import Path

import core.backends.sdxl as sdxl_backend


def test_sdxl_backend_does_not_route_through_old_core_modules():
    source = inspect.getsource(sdxl_backend.SDXLDiffusersBackend)
    assert "from core.generation" not in source
    assert "from core.pipeline" not in source
    assert "core.backends.sdxl_runtime" in source
    assert "core.backends.sdxl_pipeline" in source


def test_old_sdxl_module_names_are_forwarding_seams_only():
    root = Path(__file__).resolve().parents[1]
    generation = (root / "core" / "generation.py").read_text(encoding="utf-8")
    pipeline = (root / "core" / "pipeline.py").read_text(encoding="utf-8")
    faceid = (root / "core" / "run_official.py").read_text(encoding="utf-8")

    assert "from core.backends.sdxl_runtime import *" in generation
    assert "from core.backends.sdxl_pipeline import *" in pipeline
    assert "from core.backends.sdxl_faceid import *" in faceid

    assert generation.count("\n") < 12
    assert pipeline.count("\n") < 12
    assert faceid.count("\n") < 12


def test_backend_owned_sdxl_implementations_are_present():
    root = Path(__file__).resolve().parents[1] / "core" / "backends"
    assert (root / "sdxl_runtime.py").stat().st_size > 10_000
    assert (root / "sdxl_pipeline.py").stat().st_size > 30_000
    assert (root / "sdxl_faceid.py").stat().st_size > 2_000


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
