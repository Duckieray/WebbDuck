"""Tests for the generation mode selection system."""

import pytest
from unittest.mock import Mock, MagicMock


class TestModeSelection:
    """Test mode selection logic without GPU."""

    def test_mode_classes_can_be_imported(self):
        """Verify all mode classes can be imported directly."""
        from webbduck.modes.text2img import Text2ImgMode
        from webbduck.modes.img2img import Img2ImgMode
        from webbduck.modes.two_pass import TwoPassMode
        from webbduck.modes.inpaint import InpaintMode
        
        assert all([Text2ImgMode, Img2ImgMode, TwoPassMode, InpaintMode])

    def test_select_mode_can_be_imported(self):
        """Verify select_mode can be imported."""
        from webbduck.modes import select_mode
        assert callable(select_mode)

    def test_text2img_can_run_no_image(self, basic_settings):
        """Text2Img should run when no input_image is set."""
        from webbduck.modes.text2img import Text2ImgMode
        mode = Text2ImgMode()
        
        settings = basic_settings.copy()
        settings["input_image"] = None
        
        assert mode.can_run(settings, None, None, None, None) is True

    def test_text2img_cannot_run_with_image(self, basic_settings):
        """Text2Img should not run when input_image is present."""
        from webbduck.modes.text2img import Text2ImgMode
        mode = Text2ImgMode()
        
        settings = basic_settings.copy()
        settings["input_image"] = Mock()  # Simulate an image
        
        assert mode.can_run(settings, None, None, None, None) is False

    def test_img2img_can_run_with_image(self, basic_settings):
        """Img2Img should run when input_image is present."""
        from webbduck.modes.img2img import Img2ImgMode
        mode = Img2ImgMode()
        
        settings = basic_settings.copy()
        settings["input_image"] = Mock()
        
        assert mode.can_run(settings, None, None, None, None) is True

    def test_img2img_cannot_run_without_image(self, basic_settings):
        """Img2Img should not run when no input_image."""
        from webbduck.modes.img2img import Img2ImgMode
        mode = Img2ImgMode()
        
        settings = basic_settings.copy()
        settings["input_image"] = None
        
        assert mode.can_run(settings, None, None, None, None) is False

    def test_inpaint_requires_mask_and_image(self, basic_settings):
        """Inpaint mode should require both mask and input image."""
        from webbduck.modes.inpaint import InpaintMode
        mode = InpaintMode()
        
        # Neither mask nor image
        settings = basic_settings.copy()
        settings["input_image"] = None
        settings["mask_image"] = None
        assert mode.can_run(settings, None, None, None, None) is False
        
        # Only image
        settings["input_image"] = Mock()
        assert mode.can_run(settings, None, None, None, None) is False
        
        # Only mask (no image)
        settings["input_image"] = None
        settings["mask_image"] = Mock()
        assert mode.can_run(settings, None, None, None, None) is False
        
        # Both
        settings["input_image"] = Mock()
        settings["mask_image"] = Mock()
        assert mode.can_run(settings, None, None, None, None) is True

    def test_two_pass_requires_second_pass_model(self, basic_settings):
        """TwoPass mode should require a second_pass_model."""
        from webbduck.modes.two_pass import TwoPassMode
        mode = TwoPassMode()
        
        settings = basic_settings.copy()
        mock_img2img = Mock()
        
        # No second pass model
        settings["second_pass_model"] = None
        assert mode.can_run(settings, None, mock_img2img, None, None) is False
        
        # Empty string
        settings["second_pass_model"] = ""
        assert mode.can_run(settings, None, mock_img2img, None, None) is False
        
        # "None" string
        settings["second_pass_model"] = "None"
        assert mode.can_run(settings, None, mock_img2img, None, None) is False
        
        # Valid model name (but no img2img pipeline)
        settings["second_pass_model"] = "some-refiner"
        assert mode.can_run(settings, None, None, None, None) is False
        
        # Valid model name with img2img pipeline
        assert mode.can_run(settings, None, mock_img2img, None, None) is True

    def test_mode_priority_inpaint_over_img2img(self, basic_settings):
        """Inpaint should be selected over Img2Img when mask is present."""
        from webbduck.modes import select_mode
        from webbduck.modes.inpaint import InpaintMode
        
        settings = basic_settings.copy()
        settings["input_image"] = Mock()
        settings["mask_image"] = Mock()
        
        mode = select_mode(settings, None, None, None, Mock())
        assert isinstance(mode, InpaintMode)

    def test_mode_priority_img2img_without_mask(self, basic_settings):
        """Img2Img should be selected when image but no mask."""
        from webbduck.modes import select_mode
        from webbduck.modes.img2img import Img2ImgMode
        
        settings = basic_settings.copy()
        settings["input_image"] = Mock()
        settings["mask_image"] = None
        
        mode = select_mode(settings, None, None, None, None)
        assert isinstance(mode, Img2ImgMode)

    def test_mode_priority_text2img_fallback(self, basic_settings):
        """Text2Img should be fallback when no image."""
        from webbduck.modes import select_mode
        from webbduck.modes.text2img import Text2ImgMode
        
        settings = basic_settings.copy()
        settings["input_image"] = None
        
        mode = select_mode(settings, None, None, None, None)
        assert isinstance(mode, Text2ImgMode)


class TestModeSignatures:
    """Test that mode method signatures are consistent."""

    def test_all_modes_have_can_run(self):
        """All modes should have can_run method."""
        from webbduck.modes.text2img import Text2ImgMode
        from webbduck.modes.img2img import Img2ImgMode
        from webbduck.modes.two_pass import TwoPassMode
        from webbduck.modes.inpaint import InpaintMode
        
        for cls in [Text2ImgMode, Img2ImgMode, TwoPassMode, InpaintMode]:
            mode = cls()
            assert hasattr(mode, "can_run")
            assert callable(mode.can_run)

    def test_all_modes_have_run(self):
        """All modes should have run method."""
        from webbduck.modes.text2img import Text2ImgMode
        from webbduck.modes.img2img import Img2ImgMode
        from webbduck.modes.two_pass import TwoPassMode
        from webbduck.modes.inpaint import InpaintMode
        
        for cls in [Text2ImgMode, Img2ImgMode, TwoPassMode, InpaintMode]:
            mode = cls()
            assert hasattr(mode, "run")
            assert callable(mode.run)

    def test_can_run_accepts_base_inpaint(self):
        """All can_run methods should accept base_inpaint parameter."""
        from webbduck.modes.text2img import Text2ImgMode
        from webbduck.modes.img2img import Img2ImgMode
        from webbduck.modes.two_pass import TwoPassMode
        from webbduck.modes.inpaint import InpaintMode
        import inspect
        
        for cls in [Text2ImgMode, Img2ImgMode, TwoPassMode, InpaintMode]:
            mode = cls()
            sig = inspect.signature(mode.can_run)
            params = list(sig.parameters.keys())
            assert "base_inpaint" in params, f"{cls.__name__}.can_run missing base_inpaint"

    def test_run_accepts_base_inpaint(self):
        """All run methods should accept base_inpaint parameter."""
        from webbduck.modes.text2img import Text2ImgMode
        from webbduck.modes.img2img import Img2ImgMode
        from webbduck.modes.two_pass import TwoPassMode
        from webbduck.modes.inpaint import InpaintMode
        import inspect
        
        for cls in [Text2ImgMode, Img2ImgMode, TwoPassMode, InpaintMode]:
            mode = cls()
            sig = inspect.signature(mode.run)
            params = list(sig.parameters.keys())
            assert "base_inpaint" in params, f"{cls.__name__}.run missing base_inpaint"
