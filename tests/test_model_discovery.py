import json
from pathlib import Path

from models.discovery import (
    discover_hf_image_models,
    discover_local_image_models,
    public_model_catalog,
    resolve_checkpoint_root,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_checkpoint_root_prefers_generic_parent(tmp_path):
    (tmp_path / "checkpoint" / "sdxl").mkdir(parents=True)
    assert resolve_checkpoint_root(tmp_path) == tmp_path / "checkpoint"


def test_local_discovery_finds_sdxl_flux_and_krea_diffusers(tmp_path):
    root = tmp_path / "checkpoint"

    sdxl = root / "sdxl" / "RealVisXL"
    _write_json(sdxl / "unet" / "config.json", {})
    (sdxl / "text_encoder_2").mkdir(parents=True)

    flux = root / "flux" / "FLUX.1-dev"
    _write_json(flux / "model_index.json", {"_class_name": "FluxPipeline"})

    krea = root / "krea2" / "Krea-2-Turbo"
    _write_json(krea / "model_index.json", {"_class_name": "Krea2Pipeline"})

    models = discover_local_image_models(root)

    assert models["RealVisXL"]["arch"] == "sdxl"
    assert models["FLUX.1-dev"]["arch"] == "flux"
    assert models["Krea-2-Turbo"]["arch"] == "krea2"


def test_single_file_uses_explicit_family_folder_hint_only(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "sdxl").mkdir(parents=True)
    (root / "sdxl" / "legacy.safetensors").write_bytes(b"")
    (root / "misc").mkdir(parents=True)
    (root / "misc" / "unknown.safetensors").write_bytes(b"")

    models = discover_local_image_models(root)

    assert models["legacy"]["arch"] == "sdxl"
    assert "unknown" not in models


def test_hf_cache_discovers_transformer_image_models_without_unet(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--black-forest-labs--FLUX.1-dev" / "snapshots" / "abc123"
    _write_json(snapshot / "model_index.json", {"_class_name": "FluxPipeline"})

    models = discover_hf_image_models(cache)

    assert "black-forest-labs/FLUX.1-dev" in models
    assert models["black-forest-labs/FLUX.1-dev"]["arch"] == "flux"
    assert models["black-forest-labs/FLUX.1-dev"]["source"] == "hf_cache"


def test_hf_cache_does_not_put_video_model_in_image_catalog(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "abc123"
    _write_json(snapshot / "model_index.json", {"_class_name": "LTX2Pipeline"})

    assert discover_hf_image_models(cache) == {}


def test_public_catalog_hides_architecture_and_marks_future_backend_unavailable(tmp_path):
    flux = tmp_path / "flux"
    _write_json(flux / "model_index.json", {"_class_name": "FluxPipeline"})
    registry = {
        "FLUX.1-dev": {
            "type": "diffusers",
            "arch": "flux",
            "path": flux,
            "source": "local",
            "defaults": {"steps": 28},
        }
    }

    payload = public_model_catalog(registry)[0]

    assert payload["name"] == "FLUX.1-dev"
    assert payload["capabilities"]["text2img"] is True
    assert payload["supported"] is False
    assert "architecture" not in payload
    assert "backend" not in payload
