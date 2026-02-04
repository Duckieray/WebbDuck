# Webbduck Test Suite

Pytest-based test suite for the webbduck image generation system.

## Quick Start

```bash
# Activate environment
# Activate environment (e.g. source .venv/bin/activate)

# Run fast tests only (no GPU required)
pytest tests/test_modes.py -v

# Run all non-GPU tests
pytest -v -m "not slow"

# Run full test suite (requires GPU)
pytest -v
```

## Test Modules

| Module | Description | GPU? |
|--------|-------------|------|
| `test_modes.py` | Mode selection logic and signatures | No |
| `test_pipeline.py` | Pipeline manager and schedulers | Mixed |
| `test_server.py` | FastAPI endpoints | No |
| `test_generation.py` | Full image generation | Yes |

## Test Categories

### Unit Tests (Fast, No GPU)

**Mode Selection** (`test_modes.py`)
- Verifies `Text2ImgMode`, `Img2ImgMode`, `InpaintMode`, `TwoPassMode` selection logic
- Tests priority ordering (inpaint > img2img > text2img)
- Validates method signatures include `base_inpaint` parameter

**Server Endpoints** (`test_server.py`)
- Health check (`/health`)
- Model listing (`/models`, `/second_pass_models`)
- Scheduler listing (`/schedulers`)
- UI serving (`/`)
- Gallery (`/gallery`)

### Integration Tests (Slow, GPU Required)

**Generation Tests** (`test_generation.py`)
- Text-to-image: basic, multiple images, custom size, seed reproducibility
- Image-to-image: basic, size preservation, strength variations
- Inpainting: replace mode, keep mode, mask blur
- Two-pass: refiner integration

## Fixtures

Shared fixtures are defined in `conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_image` | session | PIL Image from `test.jpg` |
| `basic_settings` | function | Minimal text2img settings dict |
| `img2img_settings` | function | Settings with input image |
| `inpaint_settings` | function | Settings with image and mask |
| `first_available_model` | session | First model from registry |

## Test Image

The `test.jpg` file in this folder is used for:
- Img2Img generation tests
- Inpainting tests (mask is auto-generated)

## Markers

```bash
# Skip GPU-intensive tests
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"
```

## Adding New Tests

1. **Mode logic tests**: Add to `test_modes.py` (use mocks, no GPU)
2. **API tests**: Add to `test_server.py` (use TestClient)
3. **Generation tests**: Add to `test_generation.py` with `@pytest.mark.slow`
4. **New fixtures**: Add to `conftest.py`
