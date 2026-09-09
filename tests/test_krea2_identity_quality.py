from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from core.backends import krea2
from core.backends.krea2_identity_quality import (
    denormalize_krea_latents,
    edit_position_ids_v124,
    edit_target_size_v124,
    fit_reference_pixels_v124,
    normalize_krea_latents,
)


class _FakeVAE:
    config = SimpleNamespace(
        z_dim=2,
        latents_mean=[0.25, -0.5],
        latents_std=[2.0, 0.5],
    )


def test_krea_latent_normalization_round_trip():
    vae = _FakeVAE()
    z = torch.tensor([[[[[2.25]]], [[[-0.25]]]]], dtype=torch.float32)
    normalized = normalize_krea_latents(z, vae)
    assert normalized.flatten().tolist() == pytest.approx([1.0, 0.5])
    restored = denormalize_krea_latents(normalized, vae)
    assert torch.allclose(restored, z)


def test_fit_keeps_requested_target_aspect_ratio():
    h, w = edit_target_size_v124((1024, 1536), (832, 1216), max_megapixels=2.0)
    assert (h, w) == (1216, 832)


def test_fit_reference_uses_smaller_centered_grid_for_real_ar_mismatch():
    source = Image.new('RGB', (640, 640), 'white')
    fitted, geometry = fit_reference_pixels_v124(
        source,
        target_width=848,
        target_height=1216,
        fit_mode='fit',
    )
    assert fitted.size == tuple(geometry['reference_pixels'])
    assert geometry['reference_grid'][0] <= geometry['target_grid'][0]
    assert geometry['reference_grid'][1] <= geometry['target_grid'][1]
    assert geometry['offset'][0] == pytest.approx(11.5)


def test_position_ids_apply_fractional_source_offset_only_to_reference():
    ids = edit_position_ids_v124(
        2,
        target_grid_h=4,
        target_grid_w=3,
        source_grid_h=2,
        source_grid_w=3,
        source_offset_h=1.0,
        source_offset_w=0.5,
        device='cpu',
    )
    src = ids[2:8]
    tgt = ids[8:]
    assert torch.all(src[:, 0] == 1)
    assert torch.all(tgt[:, 0] == 0)
    assert float(src[:, 2].min()) == pytest.approx(0.5)
    assert float(tgt[:, 2].min()) == pytest.approx(0.0)


def test_turbo_identity_recipe_biases_toward_face_detail():
    assert krea2._identity_recipe_defaults('turbo') == (12, 0.0)
    assert krea2._identity_recipe_defaults('base') == (20, 3.0)


def test_identity_token_budget_uses_live_vram_pressure():
    # Healthy 16 GB-class card: use the hardware-validated 2688 face-detail tier.
    assert krea2._recommended_identity_token_budget(15.51, 13.2) == 2688
    # Moderate pressure: step down without returning to the tiny default immediately.
    assert krea2._recommended_identity_token_budget(15.51, 10.2) == 2048
    # Heavy pressure: retain the original conservative escape hatch.
    assert krea2._recommended_identity_token_budget(15.51, 8.5) == 1792
    # Missing live-free telemetry stays conservative rather than assuming headroom.
    assert krea2._recommended_identity_token_budget(15.51, None) == 2048
