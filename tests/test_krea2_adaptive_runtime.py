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


def _rocm_hardware(total: float, free: float) -> dict:
    return {
        "accelerator": "rocm",
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
    assert tuned["width"] == 768
    assert tuned["height"] == 1152
    assert plan["effective_image_tokens"] <= 3584
    assert plan["resolution_scaled"] is True
    assert tuned["steps"] == 14
    assert plan["steps_tuned"] is True


def test_square_request_stays_square_when_scaled():
    width, height = adaptive._fit_token_budget(1024, 1024, 3584)
    assert width == height
    assert adaptive._image_tokens(width, height) <= 3584


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


def test_rocm_uses_its_own_conservative_budget():
    cuda_budget = adaptive._token_budget(
        _cuda_hardware(15.51, 13.5),
        "base",
    )
    rocm_budget = adaptive._token_budget(
        _rocm_hardware(15.51, 13.5),
        "base",
    )

    assert cuda_budget == 3584
    assert rocm_budget == 3072


def test_non_accelerator_runtime_does_not_apply_gpu_token_budget():
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
            "effective_width": 768,
            "effective_height": 1152,
            "effective_steps": 14,
            "resolution_scaled": True,
            "steps_tuned": True,
        }
    }

    krea2._apply_effective_request_settings(settings, runtime)

    assert settings["requested_width"] == 896
    assert settings["requested_height"] == 1344
    assert settings["requested_steps"] == 28
    assert settings["width"] == 768
    assert settings["height"] == 1152
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


# --------------------------------------------------------------------------------------
# Phase 5: identity-aware adaptive planning
# --------------------------------------------------------------------------------------

def _make_reference(tmp_path: Path, width: int = 1024, height: int = 1024) -> Path:
    from PIL import Image

    ref = tmp_path / "refs.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (120, 80, 200)).save(ref)
    return ref


def test_identity_token_budget_halves_text2img_budget():
    hw = _cuda_hardware(15.51, 13.2)
    text_budget = adaptive._token_budget(hw, "base")
    identity_budget = adaptive._identity_token_budget(hw, "base")
    assert text_budget == 3584
    assert identity_budget == 1792

    # Unconstrained cards stay unconstrained for identity too.
    big = _cuda_hardware(40.0, 35.0)
    assert adaptive._identity_token_budget(big, "base") is None


def test_identity_token_budget_override(monkeypatch):
    hw = _cuda_hardware(15.51, 13.2)
    assert adaptive._identity_token_budget(hw, "base") == 1792

    # Request-level override (identity.token_budget) short-circuits the env.
    assert adaptive._identity_token_budget(hw, "base", override=2688) == 2688
    assert adaptive._identity_token_budget(hw, "base", override=2048) == 2048
    assert adaptive._identity_token_budget(hw, "base", override=0) == 16
    assert adaptive._identity_token_budget(hw, "base", override=7777) == 7776

    # Env override (no explicit override) snaps to the 16-token grid.
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET", "2048")
    assert adaptive._identity_token_budget(hw, "base") == 2048
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET", "bogus")
    assert adaptive._identity_token_budget(hw, "base") == 1792
    monkeypatch.setenv("WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET", "auto")
    assert adaptive._identity_token_budget(hw, "base") == 1792


def test_identity_request_token_budget_plans_bigger_box(tmp_path):
    hw = _cuda_hardware(15.51, 13.2)
    ref = _make_reference(tmp_path, 768, 1024)
    request = {
        "width": 1024,
        "height": 1408,
        "identity": {"reference_image": str(ref), "fit_mode": "fit"},
        "steps": 28,
    }
    tuned_default, _ = adaptive._adaptive_request(request, hw)
    request["identity"]["token_budget"] = 2688
    tuned_override, plan = adaptive._adaptive_request(request, hw)
    assert plan["identity"]["token_budget"] == 2688
    scaled_tokens = (tuned_override["width"] // 16) * (tuned_override["height"] // 16)
    default_tokens = (tuned_default["width"] // 16) * (tuned_default["height"] // 16)
    assert scaled_tokens > default_tokens
    assert plan["identity"]["target_tokens"] <= 2688


def test_identity_token_plan_counts_combined_tokens(tmp_path):
    ref = _make_reference(tmp_path, 768, 1024)
    identity = {"reference_image": str(ref), "fit_mode": "fit"}
    account, box = adaptive._identity_token_plan(
        identity, (768, 1024), (1024, 1024), budget=None
    )

    # Contained within the 1024x1024 box, source AR preserved, snapped to 16px.
    assert account["effective_size"] == [768, 1024]
    assert account["reference_tokens"] == 48 * 64
    assert account["target_tokens"] == 48 * 64
    assert account["combined_image_tokens"] == 2 * 48 * 64
    assert account["reference_readable"] is True
    assert account["resolution_scaled"] is False
    assert box == (1024, 1024)


def test_identity_token_plan_shrinks_box_to_budget(tmp_path):
    ref = _make_reference(tmp_path, 1024, 1024)
    identity = {"reference_image": str(ref), "fit_mode": "fit"}
    account, box = adaptive._identity_token_plan(
        identity, (1024, 1024), (1024, 1024), budget=1792
    )

    assert account["target_tokens"] <= 1792
    assert account["combined_image_tokens"] >= account["target_tokens"]
    assert account["resolution_scaled"] is True
    assert box[0] % 16 == 0 and box[1] % 16 == 0
    w, h = box
    assert (w // 16) * (h // 16) <= 1792


def test_identity_token_plan_does_not_act_without_reference():
    account, box = adaptive._identity_token_plan(
        {"fit_mode": "fit"}, (0, 0), (1024, 1024), budget=1792
    )
    assert account["reference_readable"] is False
    assert account["resolution_scaled"] is False
    assert box == (1024, 1024)


def test_adaptive_request_scales_identity_geometry_on_constrained_gpu(tmp_path):
    ref = _make_reference(tmp_path, 1024, 1024)
    request = {
        "variant": "base",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "prompt": "p",
        "identity": {
            "weight_path": "/cache/id.safetensors",
            "reference_image": str(ref),
            "fit_mode": "fit",
        },
    }
    tuned, plan = adaptive._adaptive_request(
        request,
        _cuda_hardware(15.51, 13.2),
    )

    assert plan["identity_enabled"] is True
    assert plan["resolution_scaled"] is True
    assert plan["steps_tuned"] is False
    assert tuned["steps"] == 28

    w, h = tuned["width"], tuned["height"]
    assert w % 16 == 0 and h % 16 == 0
    assert (w // 16) * (h // 16) <= 1792
    assert plan["identity"]["reference_tokens"] == (w // 16) * (h // 16)
    assert plan["identity"]["combined_image_tokens"] == 2 * plan["identity"]["target_tokens"]


def test_adaptive_request_keeps_identity_geometry_on_large_gpu(tmp_path):
    ref = _make_reference(tmp_path, 768, 1024)
    request = {
        "variant": "base",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "prompt": "p",
        "identity": {
            "weight_path": "/cache/id.safetensors",
            "reference_image": str(ref),
            "fit_mode": "fit",
        },
    }
    tuned, plan = adaptive._adaptive_request(
        request,
        _cuda_hardware(40.0, 35.0),
    )

    assert tuned["width"] == 1024
    assert tuned["height"] == 1024
    assert plan["resolution_scaled"] is False
    assert plan["identity"]["token_budget"] is None
    assert plan["identity"]["combined_image_tokens"] == 2 * 48 * 64
