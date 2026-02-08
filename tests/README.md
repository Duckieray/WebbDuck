# WebbDuck Test Suite

Pytest-based test suite for WebbDuck.

## Quick Start

```bash
# Activate your environment first
pytest tests/test_modes.py -v          # fast logic checks
pytest -v -m "not slow"               # non-GPU tests
pytest -v                              # full suite
```

## Test Modules

- `tests/test_modes.py`: mode selection and signatures.
- `tests/test_pipeline.py`: pipeline manager and scheduler behavior.
- `tests/test_server.py`: API endpoint behavior.
- `tests/test_generation.py`: integration generation tests (GPU).

## Markers

- `slow`: GPU-heavy/integration tests.

Examples:

```bash
pytest -m "not slow"
pytest -m "slow"
```

## Adding Tests

1. Mode logic: `tests/test_modes.py`
2. API routes: `tests/test_server.py`
3. Generation behavior: `tests/test_generation.py`
4. Shared fixtures: `tests/conftest.py`
