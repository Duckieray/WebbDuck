"""Pytest configuration and shared fixtures for webbduck tests."""

import pytest
from pathlib import Path
from PIL import Image

# Test assets directory
TESTS_DIR = Path(__file__).parent
TEST_IMAGE_PATH = TESTS_DIR / "test.jpg"


@pytest.fixture(scope="session")
def test_image():
    """Load test image for img2img/inpainting tests."""
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")
    return Image.open(TEST_IMAGE_PATH).convert("RGB")


@pytest.fixture(scope="session")
def test_image_path():
    """Return path to test image."""
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH


@pytest.fixture
def basic_settings():
    """Minimal settings dict for text2img generation."""
    return {
        "prompt": "a photo of a cat",
        "prompt_2": "",
        "negative_prompt": "blurry, low quality",
        "steps": 20,
        "cfg": 7.0,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 42,
        "experimental_compress": False,
        "scheduler": "UniPC",
    }


@pytest.fixture
def img2img_settings(basic_settings, test_image):
    """Settings for img2img generation."""
    settings = basic_settings.copy()
    settings["input_image"] = test_image
    settings["strength"] = 0.75
    return settings


@pytest.fixture
def inpaint_settings(img2img_settings, test_image):
    """Settings for inpainting generation."""
    settings = img2img_settings.copy()
    # Create a simple mask (white center, black edges)
    w, h = test_image.size
    mask = Image.new("L", (w, h), 0)  # Black background
    # Draw white rectangle in center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
    
    settings["mask_image"] = mask
    settings["inpainting_fill"] = "replace"
    settings["mask_blur"] = 4
    return settings


@pytest.fixture
def two_pass_settings(basic_settings):
    """Settings for two-pass generation."""
    settings = basic_settings.copy()
    settings["second_pass_model"] = None  # Will be set in test if available
    settings["second_pass_mode"] = "auto"
    settings["refinement_strength"] = 0.3
    return settings


@pytest.fixture(scope="session")
def available_models():
    """Get list of available models from registry."""
    try:
        from webbduck.models.registry import MODEL_REGISTRY
        return MODEL_REGISTRY
    except Exception:
        return {}


@pytest.fixture(scope="session")
def first_available_model(available_models):
    """Get first available model name for testing."""
    if not available_models:
        pytest.skip("No models available in registry")
    return next(iter(available_models.keys()))


@pytest.fixture
def client():
    """Create test client for FastAPI app (shared fixture)."""
    from fastapi.testclient import TestClient
    from webbduck.server.app import app
    return TestClient(app)
