from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from core.backends.flux import FluxDiffusersBackend, _resolve_flux_identity
from models.model_descriptor import ModelDescriptor, capabilities_for_architecture, defaults_for_architecture


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="black-forest-labs/FLUX.2-klein-9B",
        path="/cache/flux",
        format="diffusers",
        source="hf_cache",
        architecture="flux",
        backend="flux_diffusers",
        capabilities=capabilities_for_architecture("flux"),
        defaults=defaults_for_architecture("flux"),
        constraints={"dimension_multiple": 8},
    )


def _reference(path: Path, color: str) -> Path:
    Image.new("RGB", (32, 32), color).save(path)
    return path


def test_flux_capabilities_expose_identity_personas():
    capabilities = capabilities_for_architecture("flux")
    assert capabilities.identity_adapter is True
    assert capabilities.text2img is True
    assert capabilities.img2img is True


def test_flux_identity_resolves_persona_reference_files(tmp_path):
    front = _reference(tmp_path / "front.png", "white")
    three_quarter = _reference(tmp_path / "three_quarter.png", "gray")
    settings = {
        "identity_adapter": {
            "enabled": True,
            "preset_name": "Della",
            "reference_images": [str(front), str(three_quarter)],
        }
    }

    refs, persona_name = _resolve_flux_identity(settings)

    assert refs == [str(front.resolve()), str(three_quarter.resolve())]
    assert persona_name == "Della"
    assert settings["identity_adapter"]["type"] == "flux2_native"
    assert settings["identity_adapter"]["provider"] == "flux2_native"
    assert settings["identity_adapter_debug"] == {
        "provider": "flux2_native",
        "persona_name": "Della",
        "reference_images_requested": 2,
        "reference_images_resolved": 2,
        "text_identity_guidance": True,
    }


def test_flux_identity_rejects_more_than_five_references(tmp_path):
    refs = []
    for index in range(6):
        refs.append(str(_reference(tmp_path / f"ref_{index}.png", "white")))

    settings = {
        "identity_adapter": {
            "enabled": True,
            "reference_images": refs,
        }
    }

    with pytest.raises(ValueError, match="at most 5 reference images"):
        _resolve_flux_identity(settings)


def test_flux_backend_serializes_native_persona_references(monkeypatch, tmp_path):
    front = _reference(tmp_path / "front.png", "white")
    profile = _reference(tmp_path / "profile.png", "gray")
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            image_path = output_dir / "flux_000.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            result_path.write_text(
                json.dumps({"ok": True, "images": [str(image_path)], "seed": 42}),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("core.backends.flux.subprocess.Popen", FakeProcess)
    backend = FluxDiffusersBackend()
    settings = {
        "prompt": "Della debugging code at a workstation late at night",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 1.0,
        "num_images": 1,
        "seed": 42,
        "identity_adapter": {
            "enabled": True,
            "preset_name": "Della",
            "text_identity_guidance": False,
            # Deliberately keep an old SDXL preset type to prove the generic
            # saved-persona reference set is normalized by the FLUX backend.
            "type": "faceid_sdxl",
            "reference_images": [str(front), str(profile)],
        },
    }

    images, seed = backend.generate(_descriptor(), settings)

    assert seed == 42
    assert len(images) == 1
    payload = captured["payload"]
    assert payload["reference_images"] == [str(front.resolve()), str(profile.resolve())]
    assert payload["identity_provider"] == "flux2_native"
    assert payload["identity_name"] == "Della"
    assert payload["input_image"] is None
    assert payload["prompt"] == "Della debugging code at a workstation late at night"


def test_flux_worker_combines_input_and_identity_conditioning_without_replacing_img2img():
    source = Path("core/backends/flux_worker.py").read_text(encoding="utf-8")

    assert 'request.get("reference_images")' in source
    assert "conditioning_images = []" in source
    assert "conditioning_images.append(input_image)" in source
    assert "conditioning_images.extend(reference_images)" in source
    assert "if len(conditioning_images) > 1" in source
    assert '"identity_reference_count": len(reference_images)' in source
    assert '"conditioning_image_count": len(conditioning_images)' in source


def test_persona_ui_reuses_existing_reference_and_preset_storage():
    source = Path("ui/modules/PersonaIdentityUI.js").read_text(encoding="utf-8")
    app_source = Path("ui/app.js").read_text(encoding="utf-8")

    assert "Identity / Persona" in source
    assert "Saved Personas" in source
    assert "FLUX.2 Native References" in source
    assert "Up to 5 references are supported" in source
    assert "./modules/PersonaIdentityUI.js" in app_source
