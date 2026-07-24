# WebbDuck Test Guide

WebbDuck uses pytest for server, runtime, mode, UI-sanity, and regression coverage.

## Quick Commands

```bash
pytest tests/test_server.py -v
pytest tests/test_prompt_conditioning.py -v
pytest tests/test_ui_sanity.py -v
pytest -v -m "not slow"
pytest -v
```

If you use the recommended conda environment in WSL/Linux:

```bash
conda run -n webbduck pytest -v -m "not slow"
```

## Test File Map

| File | Covers |
| --- | --- |
| `tests/test_server.py` | API route behavior and request validation |
| `tests/test_server_captioning.py` | caption-related server endpoints |
| `tests/test_modes.py` | mode selection and mode-specific request handling |
| `tests/test_generation.py` | heavier generation integration coverage |
| `tests/test_pipeline.py` | pipeline manager and scheduler behavior |
| `tests/test_prompt_conditioning.py` | SDXL prompt conditioning and long-prompt logic |
| `tests/test_embedding_loading.py` | embedding discovery and loading behavior |
| `tests/test_captioning.py` | captioning integration helpers |
| `tests/test_runtime.py` | runtime/device profile behavior |
| `tests/test_thumbnails.py` | thumbnail generation and serving |
| `tests/test_ui_sanity.py` | UI contract and markup sanity checks |
| `tests/test_outpaint_pyramid_regression.py` | smart-extend/outpaint regression coverage |
| `tests/test_metastore.py` | metastore tests |
| `tests/test_identity_adapter.py` | identity adapter tests |

Shared fixtures and helpers live in `tests/conftest.py`.

`tests/test_ui_sanity.py` is a browser check, not a pure unit test. It expects a running WebbDuck server and Playwright/browser dependencies.

## Choosing The Right Tests

### API or request payload change

Run:

```bash
pytest tests/test_server.py -v
```

### Prompt or conditioning change

Run:

```bash
pytest tests/test_prompt_conditioning.py -v
pytest tests/test_modes.py -v
```

### Pipeline, runtime, or device behavior change

Run:

```bash
pytest tests/test_pipeline.py -v
pytest tests/test_runtime.py -v
```

### Gallery or thumbnail change

Run:

```bash
pytest tests/test_thumbnails.py -v
pytest tests/test_server.py -v
```

### Frontend contract change

Run:

```bash
pytest tests/test_ui_sanity.py -v
pytest tests/test_server.py -v
```

## Markers

- `slow`: heavier integration or GPU-oriented checks
- `unit`: fast unit tests

Examples:

```bash
pytest -m "not slow"
pytest -m "slow"
```

## Adding Tests

- Add server coverage to `tests/test_server.py` unless the feature has a clearer dedicated test file.
- Add UI contract checks to `tests/test_ui_sanity.py` when markup or payload expectations change.
- Add prompt logic tests to `tests/test_prompt_conditioning.py`.
- Add runtime or pipeline checks to the focused runtime/pipeline files instead of growing unrelated modules.

Keep tests close to the subsystem they validate so future contributors can find them quickly.
