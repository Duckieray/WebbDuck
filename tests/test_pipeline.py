"""Tests for the pipeline manager."""

import pytest
from unittest.mock import patch, MagicMock


class TestPipelineManager:
    """Test pipeline manager functionality."""

    def test_pipeline_manager_import(self):
        """Verify pipeline manager can be imported."""
        from webbduck.core.pipeline import pipeline_manager
        assert pipeline_manager is not None

    def test_pipeline_manager_has_get(self):
        """Pipeline manager should have get method."""
        from webbduck.core.pipeline import pipeline_manager
        assert hasattr(pipeline_manager, "get")
        assert callable(pipeline_manager.get)

    def test_get_returns_5_values(self, first_available_model):
        """get() should return 5 values (pipe, img2img, base_img2img, base_inpaint, trigger)."""
        from webbduck.core.pipeline import pipeline_manager
        
        result = pipeline_manager.get(
            base_model=first_available_model,
            second_pass_model=None,
            loras=[],
        )
        
        assert len(result) == 5
        pipe, img2img, base_img2img, base_inpaint, trigger = result
        
        # pipe should always be set
        assert pipe is not None
        # img2img is None when no second_pass_model
        assert img2img is None
        # base_img2img should be set
        assert base_img2img is not None
        # base_inpaint should be set  
        assert base_inpaint is not None
        # trigger is a string
        assert isinstance(trigger, str)

    def test_set_active_unet(self, first_available_model):
        """Test UNet swapping functionality."""
        from webbduck.core.pipeline import pipeline_manager
        
        # First load a model
        pipeline_manager.get(
            base_model=first_available_model,
            second_pass_model=None,
            loras=[],
        )
        
        # Should not raise
        pipeline_manager.set_active_unet("base")
        
        # Without second pass, this should raise
        with pytest.raises(AssertionError):
            pipeline_manager.set_active_unet("second_pass")


class TestSchedulers:
    """Test scheduler functionality."""

    def test_schedulers_import(self):
        """Verify schedulers module can be imported."""
        from webbduck.core.schedulers import SCHEDULERS, create_scheduler
        assert SCHEDULERS is not None
        assert callable(create_scheduler)

    def test_scheduler_list_not_empty(self):
        """Should have at least one scheduler available."""
        from webbduck.core.schedulers import SCHEDULERS
        assert len(SCHEDULERS) > 0

    def test_unipc_scheduler_as_default(self):
        """UniPC should be available as default/fallback scheduler."""
        from webbduck.core.schedulers import get_scheduler_class
        from diffusers import UniPCMultistepScheduler
        # UniPC is the default fallback when scheduler name not found
        assert get_scheduler_class("UniPC") == UniPCMultistepScheduler
        assert get_scheduler_class("unknown") == UniPCMultistepScheduler
