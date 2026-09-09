"""Request-layer wiring tests for the Krea identity adapter.

Covers Phase 2 + Phase 4 integration: the backend threads the normalized Krea
identity payload to the worker (fail-fast before the runtime spawns), the
isolated worker dispatches identity requests to the phased GPU identity run
(grounded encode, reference latent, edit forward; never silent non-identity
output), and server-side persona presets round-trip the Krea tuning keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.backends import krea2 as krea2_backend
from core.backends.krea2_identity import KreaIdentityError
from core.backends.krea2_worker import _identity_resident_blocks


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
    assert payload["max_megapixels"] == 2.0


# --------------------------------------------------------------------------------------
# Resident-block VRAM budgeting
# --------------------------------------------------------------------------------------

def test_identity_resident_blocks_thresholds():
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28) == 20
    assert _identity_resident_blocks(12.2, 3.04, 0.419, 28) == 18
    assert _identity_resident_blocks(4.0, 2.0, 0.419, 28) == 1
    assert _identity_resident_blocks(60.0, 2.0, 0.419, 28) == 28
    assert _identity_resident_blocks(12.2, 2.0, 0.0, 28) == 0


def test_identity_resident_blocks_override(monkeypatch):
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28, override=5) == 5
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28, override=0) == 0
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28, override=1000) == 28
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS", "7")
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28) == 7
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS", "0")
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28) == 0
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS", "bogus")
    assert _identity_resident_blocks(12.2, 2.0, 0.419, 28) == 20


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
# Worker identity dispatch (Phase 4: GPU identity path is installed)
# --------------------------------------------------------------------------------------

def test_base_run_dispatches_identity(tmp_path, monkeypatch):
    from core.backends import krea2_worker as base_worker

    calls: dict = {}

    def fake_identity(request, output_dir, progress_path=None):
        calls["request"] = request
        return {"ok": True, "images": [], "runtime": {}}

    monkeypatch.setattr(base_worker, "_run_identity", fake_identity)
    result = base_worker._run(
        {"identity": {"weight_path": "/cache/id.safetensors"}, "prompt": "p"},
        tmp_path,
    )
    assert calls["request"]["identity"]["weight_path"] == "/cache/id.safetensors"
    assert result["ok"] is True


def test_base_run_keeps_text2img_path_when_no_identity(tmp_path, monkeypatch):
    from core.backends import krea2_worker as base_worker

    def boom(*args, **kwargs):
        raise AssertionError("identity run should not be dispatched")

    monkeypatch.setattr(base_worker, "_run_identity", boom)

    class ReachedLoad(Exception):
        pass

    def bad_load(*args, **kwargs):
        raise ReachedLoad("normal load path reached")

    monkeypatch.setattr(base_worker, "_load_pipeline", bad_load)
    with pytest.raises(ReachedLoad):
        base_worker._run({"prompt": "x", "width": 512, "height": 512}, tmp_path)


def test_safe_run_dispatches_identity(tmp_path, monkeypatch):
    from core.backends import krea2_worker as base_worker
    from core.backends import krea2_worker_safe as safe_worker

    calls = {"n": 0}

    def fake_identity(request, output_dir, progress_path=None):
        calls["n"] += 1
        return {"ok": True, "runtime": {}}

    monkeypatch.setattr(base_worker, "_run_identity", fake_identity)
    result = safe_worker._run(
        {"identity": {"weight_path": "/cache/id.safetensors"}, "prompt": "p"},
        tmp_path,
    )
    assert calls["n"] == 1
    assert result["ok"] is True


def test_run_identity_rejects_malformed_payload(tmp_path):
    from core.backends import krea2_worker as base_worker

    with pytest.raises(RuntimeError, match="malformed identity payload"):
        base_worker._run_identity({"identity": {"ref_boost": 4.0}}, tmp_path)
    with pytest.raises(RuntimeError, match="malformed identity payload"):
        base_worker._run_identity({"prompt": "p", "identity": None}, tmp_path)


def test_run_identity_fails_fast_on_missing_reference(tmp_path, monkeypatch):
    from core.backends import krea2_worker as base_worker

    def bad_load(*args, **kwargs):
        raise AssertionError("pipeline should not load when the reference is missing")

    monkeypatch.setattr(base_worker, "_load_pipeline", bad_load)
    with pytest.raises(KreaIdentityError, match="reference image does not exist"):
        base_worker._run_identity(
            {
                "prompt": "p",
                "identity": {
                    "weight_path": "/cache/id.safetensors",
                    "reference_image": str(tmp_path / "missing.png"),
                },
            },
            tmp_path,
        )


def test_adaptive_request_preserves_identity_geometry():
    from core.backends.krea2_worker_adaptive import _adaptive_request

    hardware = {"accelerator": "cuda", "total_vram_gb": 12.0, "free_vram_gb": 8.0}
    request = {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "variant": "base",
        "prompt": "p",
        "identity": {"weight_path": "/cache/id.safetensors"},
    }
    tuned, plan = _adaptive_request(request, hardware)
    assert tuned["width"] == 1024
    assert tuned["height"] == 1024
    assert tuned["steps"] == 28
    assert plan["identity_enabled"] is True
    assert plan["resolution_scaled"] is False
    assert plan["steps_tuned"] is False


def test_adaptive_request_still_tunes_text2img():
    from core.backends.krea2_worker_adaptive import _adaptive_request

    hardware = {"accelerator": "cuda", "total_vram_gb": 12.0, "free_vram_gb": 8.0}
    tuned, plan = _adaptive_request(
        {"width": 1024, "height": 1024, "steps": 28, "variant": "base", "prompt": "p"},
        hardware,
    )
    assert plan["identity_enabled"] is False
    assert plan["resolution_scaled"] is True
    assert plan["steps_tuned"] is True
    assert tuned["steps"] < 28


def test_configure_krea_identity_transformer_offloads_and_reconfigures(monkeypatch):
    from core.backends import krea2_worker as base_worker

    calls = {"to_cpu": 0, "configured": []}

    class FakeTransformer:
        def to(self, *args, **kwargs):
            if args and str(args[0]) == "cpu":
                calls["to_cpu"] += 1

    class FakePipe:
        def __init__(self) -> None:
            self.transformer = FakeTransformer()

    def fake_configure(pipe, **kwargs):
        calls["configured"].append(kwargs)
        return kwargs.get("mode_override") or "resident"

    monkeypatch.setattr(
        base_worker,
        "detect_torch_hardware",
        lambda torch_mod: {
            "accelerator": "cuda",
            "total_vram_gb": 15.51,
            "free_vram_gb": 13.2,
        },
    )
    monkeypatch.setattr(base_worker, "_configure_execution", fake_configure)

    pipe = FakePipe()
    mode = base_worker.configure_krea_identity_transformer(
        pipe,
        load_info={},
        transformer_storage_gb=11.0,
        reserve_gb=3.0,
    )
    assert mode == "resident"
    assert calls["to_cpu"] == 1
    assert calls["configured"][0]["mode_override"] is None
    assert calls["configured"][0]["reserve_gb"] == 3.0

    # Simulated OOM retry reconfiguration forces a safer profile.
    retry_mode = base_worker.configure_krea_identity_transformer(
        pipe,
        load_info={},
        transformer_storage_gb=11.0,
        reserve_gb=3.0,
        mode_override="transformer-block",
    )
    assert retry_mode == "transformer-block"
    assert calls["configured"][1]["mode_override"] == "transformer-block"


def test_run_identity_retry_ladder_uses_reusable_reconfigure():
    import inspect

    from core.backends import krea2_worker as base_worker

    source = inspect.getsource(base_worker._run_identity)
    ladder_tail = source.split("configure_krea_identity_transformer(", 1)[1]
    assert 'mode_override="transformer-block"' in ladder_tail
    assert "torch.cuda.empty_cache()" not in ladder_tail
    assert "pipe.transformer.to(" not in ladder_tail


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


# --------------------------------------------------------------------------------------
# Identity artifact upscale (effective res -> requested size)
# --------------------------------------------------------------------------------------

def test_identity_artifact_infer_upscale_step():
    from core.backends.krea2_worker import _infer_upscale_step

    assert _infer_upscale_step((528, 784), (832, 1216)) == 2
    assert _infer_upscale_step((512, 768), (1024, 1536)) == 2
    assert _infer_upscale_step((256, 384), (832, 1216)) == 4
    assert _infer_upscale_step((0, 0), (832, 1216)) == 2


def test_identity_artifact_no_upscale_when_native(monkeypatch):
    from PIL import Image

    from core.backends.krea2_worker import _maybe_upscale_identity_artifact

    monkeypatch.delenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE", raising=False)
    img = Image.new("RGB", (832, 1216), (10, 20, 30))
    out, note = _maybe_upscale_identity_artifact(
        img, requested=(832, 1216), effective=(832, 1216)
    )
    assert note is None
    assert out is img


def test_identity_artifact_upscale_disabled_env(monkeypatch):
    from PIL import Image

    from core.backends.krea2_worker import _maybe_upscale_identity_artifact

    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE", "0")
    img = Image.new("RGB", (528, 784), (10, 20, 30))
    out, note = _maybe_upscale_identity_artifact(
        img, requested=(832, 1216), effective=(528, 784)
    )
    assert note is None
    assert out is img


def test_identity_artifact_upscale_realesrgan_path(monkeypatch):
    import sys
    import types

    import numpy as np
    from PIL import Image

    from core.backends.krea2_worker import _maybe_upscale_identity_artifact

    class FakeUpsampler:
        def enhance(self, bgr, outscale=2):
            h, w = bgr.shape[:2]
            return (
                np.repeat(bgr, outscale, axis=0).repeat(outscale, axis=1),
                None,
            )

    fake = types.ModuleType("models.upscaler")
    fake.get_upsampler = lambda scale: FakeUpsampler()
    monkeypatch.setitem(sys.modules, "models.upscaler", fake)
    monkeypatch.delenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE", raising=False)

    img = Image.new("RGB", (528, 784), (120, 30, 200))
    out, note = _maybe_upscale_identity_artifact(
        img, requested=(832, 1216), effective=(528, 784)
    )
    assert note["upscaler"] == "realesrgan-x2"
    assert note["from"] == [528, 784]
    assert note["to"] == [832, 1216]
    assert out.size == (832, 1216)


def test_identity_artifact_upscale_lanczos_fallback(monkeypatch):
    import sys
    import types

    from PIL import Image

    from core.backends.krea2_worker import _maybe_upscale_identity_artifact

    def boom(*_args, **_kwargs):
        raise RuntimeError("no upscaler here")

    fake = types.ModuleType("models.upscaler")
    fake.get_upsampler = boom
    monkeypatch.setitem(sys.modules, "models.upscaler", fake)
    monkeypatch.delenv("WEBBDUCK_KREA2_IDENTITY_UPSCALE", raising=False)

    img = Image.new("RGB", (528, 784), (120, 30, 200))
    out, note = _maybe_upscale_identity_artifact(
        img, requested=(832, 1216), effective=(528, 784)
    )
    assert note["upscaler"] == "lanczos"
    assert "RuntimeError" in note["upscale_error"]
    assert out.size == (832, 1216)