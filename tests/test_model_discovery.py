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


def _write_weight(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


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


def test_local_discovery_finds_flux2_klein_9b_gguf(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = root / "flux" / "flux-2-klein-9b-Q5_K_M.gguf"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    models = discover_local_image_models(root)
    entry = models["flux-2-klein-9b-Q5_K_M"]

    assert entry["arch"] == "flux"
    assert entry["type"] == "gguf"
    assert entry["path"] == checkpoint
    assert entry["detection"]["family"] == "flux2_klein"
    assert entry["detection"]["variant"] == "distilled"
    assert entry["detection"]["parameter_size"] == "9B"
    assert entry["detection"]["quantization"] == "Q5_K_M"
    assert entry["detection"]["component_model"] == "black-forest-labs/FLUX.2-klein-9B"

    payload = public_model_catalog(models)[0]
    assert payload["supported"] is True
    assert payload["defaults"]["steps"] == 4
    assert payload["defaults"]["cfg"] == 1.0


def test_local_discovery_ignores_unknown_gguf(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = root / "misc" / "something-Q5_K_M.gguf"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    assert discover_local_image_models(root) == {}


def test_single_file_uses_explicit_family_folder_hint_only(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "sdxl").mkdir(parents=True)
    (root / "sdxl" / "legacy.safetensors").write_bytes(b"")
    (root / "misc").mkdir(parents=True)
    (root / "misc" / "unknown.safetensors").write_bytes(b"")

    models = discover_local_image_models(root)

    assert models["legacy"]["arch"] == "sdxl"
    assert "unknown" not in models


def test_hf_cache_discovers_complete_transformer_image_model_without_unet(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc123"
    _write_json(snapshot / "model_index.json", {"_class_name": "Flux2KleinPipeline"})
    _write_json(snapshot / "transformer" / "config.json", {"_class_name": "Flux2Transformer2DModel"})
    _write_weight(snapshot / "transformer" / "diffusion_pytorch_model.safetensors")

    models = discover_hf_image_models(cache)

    assert "black-forest-labs/FLUX.2-klein-4B" in models
    assert models["black-forest-labs/FLUX.2-klein-4B"]["arch"] == "flux"
    assert models["black-forest-labs/FLUX.2-klein-4B"]["source"] == "hf_cache"


def test_hf_cache_ignores_partial_flux_support_component_snapshot(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--black-forest-labs--FLUX.2-klein-9B" / "snapshots" / "supportonly"
    _write_json(snapshot / "model_index.json", {"_class_name": "Flux2KleinPipeline"})
    _write_json(snapshot / "transformer" / "config.json", {"_class_name": "Flux2Transformer2DModel"})
    _write_weight(snapshot / "text_encoder" / "model.safetensors")
    _write_weight(snapshot / "vae" / "diffusion_pytorch_model.safetensors")

    models = discover_hf_image_models(cache)

    assert "black-forest-labs/FLUX.2-klein-9B" not in models


def test_hf_cache_prefers_complete_older_snapshot_over_newer_partial_snapshot(tmp_path):
    cache = tmp_path / "hub"
    repo = cache / "models--black-forest-labs--FLUX.2-klein-9B" / "snapshots"

    complete = repo / "complete"
    _write_json(complete / "model_index.json", {"_class_name": "Flux2KleinPipeline"})
    _write_json(complete / "transformer" / "config.json", {"_class_name": "Flux2Transformer2DModel"})
    _write_weight(complete / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors")

    partial = repo / "partial"
    _write_json(partial / "model_index.json", {"_class_name": "Flux2KleinPipeline"})
    _write_json(partial / "transformer" / "config.json", {"_class_name": "Flux2Transformer2DModel"})

    complete.touch()
    partial.touch()
    partial_mtime = complete.stat().st_mtime + 10
    import os
    os.utime(partial, (partial_mtime, partial_mtime))

    models = discover_hf_image_models(cache)

    assert models["black-forest-labs/FLUX.2-klein-9B"]["path"] == complete


def test_hf_cache_discovers_qwen_image_2512_as_runnable(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen-Image-2512" / "snapshots" / "deadbeef"
    _write_json(snapshot / "model_index.json", {"_class_name": "QwenImagePipeline"})
    _write_json(snapshot / "transformer" / "config.json", {"_class_name": "QwenImageTransformer2DModel"})
    _write_weight(snapshot / "transformer" / "diffusion_pytorch_model.safetensors")

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
