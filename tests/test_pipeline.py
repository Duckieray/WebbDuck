"""Tests for the backend-owned SDXL pipeline manager."""

import pytest


class TestPipelineManager:
    """Test mature SDXL pipeline manager functionality."""

    def test_pipeline_manager_import(self):
        """Verify backend-owned pipeline manager can be imported."""
        from core.backends.sdxl_pipeline import pipeline_manager
        assert pipeline_manager is not None

    def test_pipeline_manager_has_get(self):
        """Pipeline manager should have get method."""
        from core.backends.sdxl_pipeline import pipeline_manager
        assert hasattr(pipeline_manager, "get")
        assert callable(pipeline_manager.get)

    def test_get_returns_runtime_components(self, first_available_model):
        """get() returns SDXL pipelines plus trigger/identity metadata."""
        from core.backends.sdxl_pipeline import pipeline_manager

        result = pipeline_manager.get(
            base_model=first_available_model,
            second_pass_model=None,
            loras=[],
        )

        assert len(result) == 6
        pipe, img2img, base_img2img, base_inpaint, trigger, faceid_kwargs = result
        assert pipe is not None
        assert img2img is None
        assert base_img2img is not None
        assert base_inpaint is not None
        assert isinstance(trigger, str)
        assert isinstance(faceid_kwargs, dict)

    def test_set_active_unet(self, first_available_model):
        """Test UNet swapping functionality."""
        from core.backends.sdxl_pipeline import pipeline_manager

        pipeline_manager.get(
            base_model=first_available_model,
            second_pass_model=None,
            loras=[],
        )

        pipeline_manager.set_active_unet("base")

        with pytest.raises(AssertionError):
            pipeline_manager.set_active_unet("second_pass")


class TestSchedulers:
    """Test scheduler functionality."""

    def test_schedulers_import(self):
        """Verify scheduler names are host-safe and construction remains callable."""
        from core.schedulers import SCHEDULERS, create_scheduler
        assert SCHEDULERS is not None
        assert callable(create_scheduler)

    def test_scheduler_list_not_empty(self):
        from core.schedulers import SCHEDULERS
        assert len(SCHEDULERS) > 0

    def test_unipc_scheduler_as_default(self):
        from core.schedulers import get_scheduler_class
        from diffusers import UniPCMultistepScheduler
        assert get_scheduler_class("UniPC") == UniPCMultistepScheduler
        assert get_scheduler_class("unknown") == UniPCMultistepScheduler
