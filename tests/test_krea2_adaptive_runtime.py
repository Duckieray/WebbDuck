from __future__ import annotations

from pathlib import Path

from core.backends import krea2
from core.backends import krea2_worker_adaptive as adaptive


def _cuda_hardware(total: float, free: float) -> dict:
    return {
        "accelerator": "cuda",
        "total_vram_gb": total,
        "free_vram_gb": free,
    }


def test_16gb_raw_portrait_is_scaled_to_safe_token_budget():
    request = {
        "variant": "base",
        "width": 896,
        "height": 1344,
        "steps": 28,
    }
    tuned, plan = adaptive._adaptive_request(
        request,
        _cuda_hardware(15.51, 13.2),
    )

    assert plan["requested_image_tokens"] == 4704
    assert plan["token_budget"] == 3584
    assert tuned["width"] == 784
    assert tuned["height"] == 1168
    assert plan["effective_image_tokens"] <= 3584
    assert plan["resolution_scaled"] is True
    assert tuned["steps"] == 14
    assert plan["steps_tuned"] is True


def test_custom_step_count_is_preserved_when_resolution_is_scaled():
    request = {
        "variant": "base",
        "width": 896,
        "height": 1344,
        "steps": 12,
    }
    tuned, plan = adaptive._adaptive_request(
        request,
        _cuda_hardware(15.51, 13.2),
    )

    assert plan["resolution_scaled"] is True
    assert tuned["steps"] == 12
    assert plan["steps_tuned"] is False


def test_live_vram_pressure_reduces_token_budget():
    normal = adaptive._token_budget(
        _cuda_hardware(15.51, 13.5),
        "base",
    )
    pressured = adaptive._token_budget(
        _cuda_hardware(15.51, 9.5),
        "base",
    )

    assert normal == 3584
    assert pressured == 3072


def test_non_cuda_runtime_does_not_apply_cuda_token_budget():
    hardware = {
        "accelerator": "cpu",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
    }
    request = {
        "variant": "base",
        "width": 896,
        "height": 1344,
        "steps": 28,
    }
    tuned, plan = adaptive._adaptive_request(request, hardware)

    assert plan["token_budget"] is None
    assert tuned["width"] == 896
    assert tuned["height"] == 1344
    assert tuned["steps"] == 28


def test_turbo_keeps_its_step_contract():
    request = {
        "variant": "turbo",
        "width": 1024,
        "height": 1024,
        "steps": 8,
    }
    tuned, plan = adaptive._adaptive_request(
        request,
        _cuda_hardware(15.51, 13.2),
    )

    assert tuned["steps"] == 8
    assert plan["steps_tuned"] is False


def test_effective_request_replaces_saved_generation_dimensions():
    settings = {
        "width": 896,
        "height": 1344,
        "steps": 28,
    }
    runtime = {
        "adaptive_request": {
            "requested_width": 896,
            "requested_height": 1344,
            "requested_steps": 28,
            "effective_width": 784,
            "effective_height": 1168,
            "effective_steps": 14,
            "resolution_scaled": True,
            "steps_tuned": True,
        }
    }

    krea2._apply_effective_request_settings(settings, runtime)

    assert settings["requested_width"] == 896
    assert settings["requested_height"] == 1344
    assert settings["requested_steps"] == 28
    assert settings["width"] == 784
    assert settings["height"] == 1168
    assert settings["steps"] == 14
    assert settings["krea_request_adapted"] is True


def test_backend_launches_adaptive_worker():
    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "backends"
        / "krea2.py"
    ).read_text(encoding="utf-8")
    assert 'with_name("krea2_worker_adaptive.py")' in source
