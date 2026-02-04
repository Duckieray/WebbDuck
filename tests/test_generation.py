"""Integration tests for image generation (require GPU and models)."""

import pytest
from pathlib import Path
from PIL import Image


@pytest.mark.slow
class TestText2ImgGeneration:
    """Text-to-image generation tests."""

    def test_text2img_basic(self, basic_settings, first_available_model):
        """Test basic text2img generation."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["input_image"] = None
        
        images, seed = run_generation(settings)
        
        assert images is not None
        assert len(images) == settings["num_images"]
        assert all(isinstance(img, Image.Image) for img in images)
        assert seed is not None

    def test_text2img_multiple_images(self, basic_settings, first_available_model):
        """Test generating multiple images."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["num_images"] = 2
        
        images, seed = run_generation(settings)
        
        assert len(images) == 2

    def test_text2img_custom_size(self, basic_settings, first_available_model):
        """Test custom output dimensions."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["width"] = 768
        settings["height"] = 512
        
        images, seed = run_generation(settings)
        
        assert images[0].size == (768, 512)

    def test_text2img_seed_reproducibility(self, basic_settings, first_available_model):
        """Test that same seed produces same result."""
        from webbduck.core.generation import run_generation
        import numpy as np
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["seed"] = 12345
        
        images1, seed1 = run_generation(settings)
        images2, seed2 = run_generation(settings)
        
        assert seed1 == seed2
        # Compare images (should be identical with same seed)
        arr1 = np.array(images1[0])
        arr2 = np.array(images2[0])
        assert np.allclose(arr1, arr2, atol=1)


@pytest.mark.slow
class TestImg2ImgGeneration:
    """Image-to-image generation tests."""

    def test_img2img_basic(self, basic_settings, first_available_model, test_image):
        """Test basic img2img generation."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["input_image"] = test_image
        settings["strength"] = 0.75
        
        images, seed = run_generation(settings)
        
        assert images is not None
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)

    def test_img2img_preserves_size(self, basic_settings, first_available_model, test_image):
        """Img2img should use the specified dimensions."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["input_image"] = test_image
        settings["width"] = 768
        settings["height"] = 768
        settings["strength"] = 0.5
        
        images, seed = run_generation(settings)
        
        # Should match requested dimensions, not input image size
        assert images[0].size == (768, 768)

    def test_img2img_low_strength(self, basic_settings, first_available_model, test_image):
        """Low strength should preserve more of original."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["input_image"] = test_image
        settings["strength"] = 0.1  # Very low - should be close to original
        
        images, seed = run_generation(settings)
        assert images is not None

    def test_img2img_high_strength(self, basic_settings, first_available_model, test_image):
        """High strength should allow more changes."""
        from webbduck.core.generation import run_generation
        
        settings = basic_settings.copy()
        settings["base_model"] = first_available_model
        settings["input_image"] = test_image
        settings["strength"] = 0.95
        
        images, seed = run_generation(settings)
        assert images is not None


@pytest.mark.slow
class TestInpaintGeneration:
    """Inpainting generation tests."""

    def test_inpaint_basic(self, inpaint_settings, first_available_model):
        """Test basic inpainting."""
        from webbduck.core.generation import run_generation
        
        settings = inpaint_settings.copy()
        settings["base_model"] = first_available_model
        
        images, seed = run_generation(settings)
        
        assert images is not None
        assert len(images) == 1

    def test_inpaint_replace_mode(self, inpaint_settings, first_available_model):
        """Test inpainting with replace mode."""
        from webbduck.core.generation import run_generation
        
        settings = inpaint_settings.copy()
        settings["base_model"] = first_available_model
        settings["inpainting_fill"] = "replace"
        
        images, seed = run_generation(settings)
        assert images is not None

    def test_inpaint_keep_mode(self, inpaint_settings, first_available_model):
        """Test inpainting with keep mode (inverted mask)."""
        from webbduck.core.generation import run_generation
        
        settings = inpaint_settings.copy()
        settings["base_model"] = first_available_model
        settings["inpainting_fill"] = "keep"
        
        images, seed = run_generation(settings)
        assert images is not None

    def test_inpaint_with_mask_blur(self, inpaint_settings, first_available_model):
        """Test inpainting with mask blur."""
        from webbduck.core.generation import run_generation
        
        settings = inpaint_settings.copy()
        settings["base_model"] = first_available_model
        settings["mask_blur"] = 16
        
        images, seed = run_generation(settings)
        assert images is not None


@pytest.mark.slow
class TestTwoPassGeneration:
    """Two-pass (refiner) generation tests."""

    @pytest.fixture
    def second_pass_model(self, available_models):
        """Find a second pass model if available."""
        for name in available_models.keys():
            if "refiner" in name.lower():
                return name
        # If no refiner, use first available as fallback
        if len(available_models) > 1:
            models = list(available_models.keys())
            return models[1]
        pytest.skip("No second pass model available")

    def test_two_pass_basic(
        self, two_pass_settings, first_available_model, second_pass_model
    ):
        """Test two-pass generation."""
        from webbduck.core.generation import run_generation
        
        settings = two_pass_settings.copy()
        settings["base_model"] = first_available_model
        settings["second_pass_model"] = second_pass_model
        
        images, seed = run_generation(settings)
        
        assert images is not None
        assert len(images) == 1

    def test_two_pass_auto_mode(
        self, two_pass_settings, first_available_model, second_pass_model
    ):
        """Test two-pass with auto mode selection."""
        from webbduck.core.generation import run_generation
        
        settings = two_pass_settings.copy()
        settings["base_model"] = first_available_model
        settings["second_pass_model"] = second_pass_model
        settings["second_pass_mode"] = "auto"
        
        images, seed = run_generation(settings)
        assert images is not None

    def test_two_pass_img2img_mode(
        self, two_pass_settings, first_available_model, second_pass_model
    ):
        """Test two-pass with explicit img2img mode."""
        from webbduck.core.generation import run_generation
        
        settings = two_pass_settings.copy()
        settings["base_model"] = first_available_model
        settings["second_pass_model"] = second_pass_model
        settings["second_pass_mode"] = "img2img"
        
        images, seed = run_generation(settings)
        assert images is not None
