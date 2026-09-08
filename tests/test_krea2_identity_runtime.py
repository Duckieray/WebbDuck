"""Request-layer wiring tests for the Krea identity adapter.

Covers Phase 2 integration: the backend threads the normalized Krea identity
payload to the worker (fail-fast before the runtime spawns), the isolated
worker guards identity requests until the GPU identity path lands (no silent
rendering of non-identity output), and server-side persona presets round-trip
the Krea tuning keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.backends import krea2 as krea2_backend
from core.backends.krea2_identity import KreaIdentityError
from core.backends.krea2_worker_adaptive import _guard_identity_staged


# --------------------------------------------------------------------------------------
# Backend identity payload
# --------------------------------------------------------------------------------------

def _settings_with_ref(ref: Path) -> dict:
    return {
        "prompt": "put this person at a night market",
        "identity_adapter": {
            "type": "krea2_identity_edit",
            "reference_images": [str(ref)],
            "ref_boost": 5.0,
        },
    }


def test_identity_payload_none_when_disabled(tmp_path):
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    assert krea2_backend._identity_worker_payload({}) is None
    assert krea2_backend._identity_worker_payload({"prompt": "x"}) is None
    assert krea2_backend._identity_worker_payload(
        {"identity_adapter": {"type": "faceid_sdxl", "reference_images": [str(ref)]}}
    ) is None
    assert krea2_backend._identity_worker_payload(
        {"identity_adapter": {"enabled": False, "reference_images": [str(ref)]}}
    ) is None


def test_identity_payload_env_override(tmp_path, monkeypatch):
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    weight = tmp_path / "identity.safetensors"
    weight.write_bytes(b"weight-data")
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_WEIGHT", str(weight))

    payload = krea2_backend._identity_worker_payload(_settings_with_ref(ref))

    assert payload["provider"] == "krea2_identity_edit"
    assert payload["weight_path"] == str(weight.resolve())
    assert payload["weight_source"] == "env-override"
    assert payload["reference_image"] == str(ref.resolve())
    assert payload["reference_count_used"] == 1
    assert payload["ref_boost"] == 5.0
    assert payload["grounding_px"] == 768
    assert payload["fit_mode"] == "fit"
    assert payload["lora_scale"] == 1.0
    assert payload["max_megapixels"] == 1.0


def test_identity_payload_malformed_fails_fast(tmp_path):
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    bad = _settings_with_ref(ref)
    bad["identity_adapter"]["ref_boost"] = 99.0
    with pytest.raises(KreaIdentityError, match="ref_boost"):
        krea2_backend._identity_worker_payload(bad)


def test_identity_payload_missing_weight_file(tmp_path, monkeypatch):
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    monkeypatch.setenv(
        "WEBBDUCK_KREA2_IDENTITY_WEIGHT", str(tmp_path / "missing.safetensors")
    )
    with pytest.raises(KreaIdentityError, match="does not exist|missing file"):
        krea2_backend._identity_worker_payload(_settings_with_ref(ref))


def test_generate_fails_before_spawning_worker(tmp_path, monkeypatch):
    """Malformed identity must raise in-process, never spawning the runtime."""
    ref = tmp_path / "refs" / "face.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"x")
    weight = tmp_path / "identity.safetensors"
    weight.write_bytes(b"weight-data")
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_WEIGHT", str(weight))

    spawned = {"called": False}

    def fake_popen(*args, **kwargs):
        spawned["called"] = True
        raise AssertionError("worker should not spawn")

    monkeypatch.setattr(krea2_backend.subprocess, "Popen", fake_popen)

    from core.backends.krea2 import Krea2DiffusersBackend
    from models.model_descriptor import (
        ModelDescriptor,
        capabilities_for_architecture,
    )

    descriptor = ModelDescriptor(
        name="krea-2-turbo",
        path="/cache/krea2",
        format="diffusers",
        source="hf_cache",
        architecture="krea2",
        backend="krea2_diffusers",
        capabilities=capabilities_for_architecture("krea2"),
        detection={"variant": "turbo"},
    )

    settings = _settings_with_ref(ref)
    settings["identity_adapter"]["ref_boost"] = 99.0
    with pytest.raises(KreaIdentityError, match="ref_boost"):
        Krea2DiffusersBackend().generate(descriptor, settings)
    assert not spawned["called"]


# --------------------------------------------------------------------------------------
# Worker staged guard
# --------------------------------------------------------------------------------------

def test_worker_guard_passes_when_identity_absent():
    _guard_identity_staged({})
    _guard_identity_staged({"width": 832, "height": 1216})


def test_worker_guard_rejects_malformed_identity():
    with pytest.raises(ValueError, match="weight_path"):
        _guard_identity_staged({"identity": {}})
    with pytest.raises(ValueError, match="weight_path"):
        _guard_identity_staged({"identity": "not-a-dict"})


def test_worker_guard_fails_fast_not_silent():
    with pytest.raises(NotImplementedError, match="phased GPU identity runtime"):
        _guard_identity_staged(
            {"identity": {"weight_path": "/cache/identity.safetensors"}}
        )


# --------------------------------------------------------------------------------------
# Server-side persona preset round-trip
# --------------------------------------------------------------------------------------

def _call_resolve(monkeypatch, tmp_path, preset_name):
    import server.app as server_app

    presets_file = tmp_path / ".faceid_presets.json"
    presets_file.write_text(
        json.dumps(
            {
                preset_name: {
                    "type": "krea2_identity_edit",
                    "refs": ["/outputs/refs/face.png"],
                    "ref_boost": 6.5,
                    "grounding_px": 1024,
                    "fit_mode": "fit",
                    "lora_scale": 1.1,
                    "lora_rank": "r64",
                }
            }
        )
    )
    monkeypatch.setattr(server_app, "PRESETS_FILE", presets_file)
    cfg = {"preset_name": preset_name, "ref_boost": 3.0}
    return server_app._resolve_identity_adapter_preset(cfg)


def test_preset_resolve_merges_krea_keys(monkeypatch, tmp_path):
    merged = _call_resolve(monkeypatch, tmp_path, "della")
    assert merged["type"] == "krea2_identity_edit"
    assert merged["reference_images"] == ["/outputs/refs/face.png"]
    assert merged["ref_boost"] == 3.0      # explicit request wins
    assert merged["grounding_px"] == 1024  # preset default used
    assert merged["fit_mode"] == "fit"
    assert merged["lora_scale"] == 1.1
    assert merged["lora_rank"] == "r64"


def test_preset_save_persists_krea_keys(tmp_path, monkeypatch):
    import asyncio

    import server.app as server_app

    presets_file = tmp_path / ".faceid_presets.json"
    monkeypatch.setattr(server_app, "PRESETS_FILE", presets_file)

    data = server_app.PresetData(
        name="della",
        type="krea2_identity_edit",
        refs=["/outputs/refs/face.png"],
        ref_boost=7.0,
        grounding_px=896,
        fit_mode="fit",
        lora_scale=1.0,
        lora_rank="r128",
    )
    asyncio.run(server_app.save_preset(data))
    saved = json.loads(presets_file.read_text())["della"]
    assert saved["ref_boost"] == 7.0
    assert saved["grounding_px"] == 896
    assert saved["fit_mode"] == "fit"
    assert saved["lora_rank"] == "r128"

    cfg = server_app._resolve_identity_adapter_preset(
        {"preset_name": "della", "lora_scale": 0.9}
    )
    assert cfg["ref_boost"] == 7.0
    assert cfg["grounding_px"] == 896
    assert cfg["lora_scale"] == 0.9  # explicit wins
    assert cfg["lora_rank"] == "r128"