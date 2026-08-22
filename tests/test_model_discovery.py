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


def test_local_discovery_finds_all_installed_diffusers_image_families(tmp_path):
    root = tmp_path / "checkpoint"

    sdxl = root / "sdxl" / "RealVisXL"
    _write_json(sdxl / "unet" / "config.json", {})
    (sdxl / "text_encoder_2").mkdir(parents=True)

    flux = root / "flux" / "FLUX.2-klein-4B"
    _write_json(flux / "model_index.json", {"_class_name": "Flux2KleinPipeline"})

    krea = root / "krea2" / "Krea-2-Turbo"
    _write_json(krea / "model_index.json", {"_class_name": "Krea2Pipeline"})

    qwen = root / "qwen" / "Qwen-Image-2512"
    _write_json(qwen / "model_index.json", {"_class_name": "QwenImagePipeline"})

    models = discover_local_image_models(root)

    assert models["RealVisXL"]["arch"] == "sdxl"
    assert models["FLUX.2-klein-4B"]["arch"] == "flux"
    assert models["Krea-2-Turbo"]["arch"] == "krea2"
    assert models["Qwen-Image-2512"]["arch"] == "qwen_image"


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
    snapshot = cache / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc123"
    _write_json(snapshot / "model_index.json", {"_class_name": "Flux2KleinPipeline"})

    models = discover_hf_image_models(cache)

    assert "black-forest-labs/FLUX.2-klein-4B" in models
    assert models["black-forest-labs/FLUX.2-klein-4B"]["arch"] == "flux"
    assert models["black-forest-labs/FLUX.2-klein-4B"]["source"] == "hf_cache"


def test_hf_cache_discovers_qwen_image_2512_as_runnable(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen-Image-2512" / "snapshots" / "deadbeef"
    _write_json(snapshot / "model_index.json", {"_class_name": "QwenImagePipeline"})

    models = discover_hf_image_models(cache)
    payload = public_model_catalog(models)[0]

    assert models["Qwen/Qwen-Image-2512"]["arch"] == "qwen_image"
    assert payload["name"] == "Qwen/Qwen-Image-2512"
    assert payload["supported"] is True
    assert payload["capabilities"]["text2img"] is True
    assert payload["capabilities"]["img2img"] is False
    assert payload["capabilities"]["inpaint"] is False
    assert payload["defaults"]["width"] == 1328
    assert payload["defaults"]["steps"] == 50
    assert payload["defaults"]["cfg"] == 4.0
    assert "architecture" not in payload
    assert "backend" not in payload


def test_hf_cache_does_not_put_video_model_in_image_catalog(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "abc123"
    _write_json(snapshot / "model_index.json", {"_class_name": "LTX2Pipeline"})

    assert discover_hf_image_models(cache) == {}


def test_public_catalog_marks_live_flux_runtime_available(tmp_path):
    flux = tmp_path / "flux"
    _write_json(flux / "model_index.json", {"_class_name": "Flux2KleinPipeline"})
    registry = {
        "FLUX.2-klein-4B": {
            "type": "diffusers",
            "arch": "flux",
            "path": flux,
            "source": "local",
        }
    }

    payload = public_model_catalog(registry)[0]

    assert payload["name"] == "FLUX.2-klein-4B"
    assert payload["capabilities"]["text2img"] is True
    assert payload["capabilities"]["img2img"] is True
    assert payload["capabilities"]["tokenize"] is False
    assert payload["defaults"]["steps"] == 4
    assert payload["supported"] is True
    assert "architecture" not in payload
    assert "backend" not in payload
