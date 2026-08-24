from __future__ import annotations

from pathlib import Path

import torch

from core.backends import krea2_worker_safe as safe


def test_conditioning_is_forced_to_selected_inference_dtype():
    prompt = torch.randn(1, 8, 2, 16, dtype=torch.float32)
    negative = torch.randn(1, 8, 2, 16, dtype=torch.float32)

    cast_prompt, cast_negative = safe._coerce_conditioning_dtype(
        prompt,
        negative,
        torch.bfloat16,
    )

    assert cast_prompt.device.type == "cpu"
    assert cast_prompt.dtype == torch.bfloat16
    assert cast_negative is not None
    assert cast_negative.dtype == torch.bfloat16


def test_16gb_cuda_profile_keeps_four_gib_transient_reserve():
    hardware = {
        "accelerator": "cuda",
        "total_vram_gb": 15.51,
    }
    reserve = safe._memory_reserve_gb(hardware, 1024, 1024)
    assert reserve >= 4.0


def test_larger_cuda_profile_uses_resolution_aware_base_reserve():
    hardware = {
        "accelerator": "cuda",
        "total_vram_gb": 24.0,
    }
    reserve_1024 = safe._memory_reserve_gb(hardware, 1024, 1024)
    reserve_2048 = safe._memory_reserve_gb(hardware, 2048, 2048)
    assert reserve_2048 > reserve_1024


def test_streamed_block_oom_downgrades_to_synchronous_profiles():
    retries = safe._retry_profiles(
        "transformer-block-stream-1",
        {"accelerator": "cuda"},
    )
    assert retries[0] == ("transformer-block", False)
    assert ("group", False) in retries
    assert ("sequential", False) in retries


def test_leaf_oom_does_not_retry_the_same_leaf_profile():
    retries = safe._retry_profiles(
        "transformer-leaf-sync",
        {"accelerator": "cuda"},
    )
    assert ("group", False) not in retries
    assert retries == [("sequential", False)]


def test_resident_oom_never_retries_resident():
    retries = safe._retry_profiles(
        "resident-fp8",
        {"accelerator": "cuda"},
    )
    assert retries
    assert all(not mode.startswith("resident") for mode, _stream in retries)


def test_backend_launches_hardened_worker():
    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "backends"
        / "krea2.py"
    ).read_text(encoding="utf-8")
    assert 'with_name("krea2_worker_safe.py")' in source
