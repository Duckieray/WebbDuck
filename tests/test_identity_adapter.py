"""Tests for IP-Adapter FaceID identity adapter.

These tests verify that:
1. The adapter is loaded correctly for both faceid_sdxl and faceid_plusv2_sdxl modes
2. Debug metadata is populated correctly
3. Errors are thrown loudly when conditions aren't met
4. The adapter kwargs contain the expected keys

Requires: pytest, GPU, a reference face image, and the webbduck conda env.
"""

import pytest
import json
from pathlib import Path
import torch


# ── helpers ──────────────────────────────────────────────────────

def _make_adapter_cfg(enabled=True, **overrides):
    cfg = {
        "enabled": enabled,
        "type": "faceid_sdxl",
        "repo": "h94/IP-Adapter-FaceID",
        "adapter_weight": "ip-adapter-faceid_sdxl.bin",
        "embedder": "buffalo_l",
        "adapter_scale": 1.0,
        "reference_images": [],
        "reference_mode": "primary_only",
    }
    cfg.update(overrides)
    return cfg


# ── tests ────────────────────────────────────────────────────────

@pytest.mark.slow
class TestIdentityAdapterDebugMetadata:
    """Debug metadata must be populated in all code paths."""

    def test_disabled_returns_empty_debug(self, pipeline_manager):
        pm = pipeline_manager
        kwargs = pm.apply_identity_adapter({"enabled": False})
        assert kwargs == {}
        debug = pm._identity_debug
        assert debug is not None
        assert debug["identity_adapter_applied"] is False

    def test_none_config_returns_empty_debug(self, pipeline_manager):
        pm = pipeline_manager
        kwargs = pm.apply_identity_adapter(None)
        assert kwargs == {}
        assert pm._identity_debug["identity_adapter_applied"] is False

    def test_empty_config_returns_empty_debug(self, pipeline_manager):
        pm = pipeline_manager
        kwargs = pm.apply_identity_adapter({})
        assert kwargs == {}
        assert pm._identity_debug["identity_adapter_applied"] is False


@pytest.mark.slow
class TestIdentityAdapterFailureModes:
    """Adapter must fail loudly when it cannot condition."""

    def test_fails_if_no_refs(self, pipeline_manager):
        pm = pipeline_manager
        with pytest.raises(RuntimeError, match="reference_images is empty"):
            pm.apply_identity_adapter(_make_adapter_cfg(enabled=True))


@pytest.mark.slow
class TestIdentityAdapterFaceIDSDXL:
    """FaceID SDXL mode — the primary mode for identity conditioning."""

    @pytest.mark.skipif(not Path("refs/test_face.png").exists(), reason="No test face image")
    def test_faceid_sdxl_kwargs(self, pipeline_manager):
        from core.pipeline import resolve_web_path
        pm = pipeline_manager
        ref = str(resolve_web_path("refs/test_face.png"))
        cfg = _make_adapter_cfg(enabled=True, reference_images=[ref])
        kwargs = pm.apply_identity_adapter(cfg)
        debug = pm._identity_debug

        assert "ip_adapter_image_embeds" in kwargs, (
            f"Expected ip_adapter_image_embeds in kwargs, got {list(kwargs.keys())}"
        )
        assert debug["identity_adapter_applied"] is True
        assert debug["identity_adapter_type"] == "faceid_sdxl"
        assert debug["reference_faces_detected"] >= 1
        assert debug["ip_adapter_loaded"] is True
        assert "ip_adapter_image_embeds" in debug["faceid_kwargs_keys"]

        # Check tensor shapes
        embeds = kwargs["ip_adapter_image_embeds"]
        assert isinstance(embeds, list)
        assert len(embeds) >= 1
        t = embeds[0]
        assert isinstance(t, torch.Tensor), f"Expected tensor, got {type(t)}"
        assert t.ndim == 4, f"Expected 4D tensor, got shape {t.shape}"
        assert t.shape[0] == 2, f"Expected batch=2 (CFG pair), got {t.shape[0]}"
        assert t.shape[2] == 1, f"Expected dim 2 = 1, got {t.shape}"
        assert t.shape[3] == 512, f"Expected 512-dim embedding, got {t.shape}"

    @pytest.mark.skipif(not Path("refs/test_face.png").exists(), reason="No test face image")
    def test_faceid_sdxl_multiple_refs_average_mode(self, pipeline_manager):
        from core.pipeline import resolve_web_path
        pm = pipeline_manager
        refs = [
            str(resolve_web_path("refs/test_face.png")),
            str(resolve_web_path("refs/test_face.png")),
        ]
        cfg = _make_adapter_cfg(enabled=True, reference_images=refs, reference_mode="average")
        kwargs = pm.apply_identity_adapter(cfg)
        debug = pm._identity_debug

        assert "ip_adapter_image_embeds" in kwargs
        assert debug["reference_faces_detected"] >= 1
        assert debug["identity_adapter_applied"] is True


@pytest.mark.slow
class TestIdentityAdapterPlusV2:
    """FaceID PlusV2 mode — identity + face-structure conditioning."""

    @pytest.mark.skipif(not Path("refs/test_face.png").exists(), reason="No test face image")
    def test_plusv2_with_clip_encoder(self, pipeline_manager):
        from core.pipeline import resolve_web_path
        pm = pipeline_manager
        ref = str(resolve_web_path("refs/test_face.png"))
        cfg = _make_adapter_cfg(
            enabled=True,
            type="faceid_plusv2_sdxl",
            adapter_weight="ip-adapter-faceid-plusv2_sdxl.bin",
            lora_weight="ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
            image_encoder="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            reference_images=[ref],
        )
        kwargs = pm.apply_identity_adapter(cfg)
        debug = pm._identity_debug

        assert "ip_adapter_image_embeds" in kwargs
        assert "ip_adapter_image" not in kwargs, (
            f"ip_adapter_image is unused when ip_adapter_image_embeds is provided; kwargs={list(kwargs.keys())}"
        )
        assert debug["identity_adapter_applied"] is True
        assert debug["identity_adapter_type"] == "faceid_plusv2_sdxl"
