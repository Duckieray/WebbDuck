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

Tests are grouped by subsystem. Shared fixtures and helpers live in `tests/conftest.py`.

### Server and API

| File | Covers |
| --- | --- |
| `tests/test_server.py` | FastAPI routes, request validation, and queue entrypoints |
| `tests/test_server_captioning.py` | caption-related server endpoints |
| `tests/test_captioning.py` | captioning integration helpers |
| `tests/test_model_catalog_api.py` | model catalog API responses (capabilities, defaults, constraints) |
| `tests/test_provider_credentials.py` | host-side provider credential storage |
| `tests/test_provider_credentials_api.py` | provider-credentials HTTP endpoints |

### Generation, modes, and pipeline

| File | Covers |
| --- | --- |
| `tests/test_modes.py` | mode selection and mode-specific request handling |
| `tests/test_generation.py` | heavier generation integration coverage (GPU + models) |
| `tests/test_pipeline.py` | backend-owned pipeline manager and scheduler behavior |
| `tests/test_prompt_conditioning.py` | SDXL prompt conditioning and long-prompt logic |
| `tests/test_embedding_loading.py` | embedding discovery and loading behavior |
| `tests/test_identity_adapter.py` | IP-Adapter FaceID identity adapter |
| `tests/test_identity_provider_ownership.py` | identity provider ownership and routing rules |

### Runtime, hardware, and backends

| File | Covers |
| --- | --- |
| `tests/test_runtime.py` | runtime/device profile behavior |
| `tests/test_runtime_hardware.py` | hardware-aware runtime variants |
| `tests/test_runtime_readiness_contract.py` | runtime readiness API contract |
| `tests/test_state_vram.py` | state/VRAM accounting helpers |
| `tests/test_hardware_smoke_runner.py` | `tools/run_hardware_smoke.py` runner coverage |
| `tests/test_sdxl_backend_ownership.py` | SDXL backend ownership contract |
| `tests/test_sdxl_isolated_runtime.py` | SDXL isolated-runtime wiring |
| `tests/test_model_runtime_bridge.py` | model descriptor → runtime resolution |
| `tests/test_qwen_image_backend.py` | Qwen image backend payloads and worker environment |

### FLUX backends

| File | Covers |
| --- | --- |
| `tests/test_flux_backend.py` | FLUX backend payloads and worker environment |
| `tests/test_flux_lora.py` | FLUX LoRA resolution, adapter loading, and FLUX.1 vs FLUX.2 arch detection |
| `tests/test_flux_memory_metadata.py` | FLUX.2 9B detection and OOM demotion tiers |
| `tests/test_flux_identity_personas.py` | FLUX.2 native identity/reference resolution |
| `tests/test_flux_identity_prep.py` | FLUX.2 persona preprocessing: face crop, anchor weighting, prompt |
| `tests/test_flux_identity_reference_policy.py` | FLUX identity reference-count policy |
| `tests/test_flux2_identity_architecture.py` | FLUX.2 identity capability contract |
| `tests/test_flux2_single_file.py` | FLUX.2 single-file checkpoint discovery and descriptor coverage |

### Krea 2 backends

| File | Covers |
| --- | --- |
| `tests/test_krea2_backend.py` | Krea 2 backend payloads and runtime launch |
| `tests/test_krea2_identity.py` | Krea identity adapter contract, markup controls, paired-block + resident-prefix forwards, perf-env overrides, 2.0-MP cap, token_budget validation |
| `tests/test_krea2_identity_runtime.py` | Krea identity worker payload/dispatch, resident-block VRAM budgeting, persona presets, and effective→requested artifact upscale |
| `tests/test_krea2_adaptive_runtime.py` | Krea identity-aware adaptive planning, token accounting, and token_budget override |
| `tests/test_krea2_identity_face.py` | Krea identity face-crop / face-focus reference handling |
| `tests/test_krea2_identity_multiref.py` | Krea multi-reference identity validation |
| `tests/test_krea2_identity_quality.py` | Krea identity quality gates and artifact/upscale handling |
| `tests/test_krea2_identity_stability.py` | Krea identity stability and edit geometry validation |
| `tests/test_krea2_lora.py` | Krea LoRA loading and scaling |
| `tests/test_krea2_meta_overlay.py` | Krea metadata overlay for single-file/GGUF weights |
| `tests/test_krea2_native_fp8.py` | Krea native FP8 GEMM + LoRA residual overlay |
| `tests/test_krea2_runtime_policy.py` | Krea runtime policy rules |
| `tests/test_krea2_safe_runtime.py` | Krea safe-runtime dispatch wrapper |
| `tests/test_krea2_single_file.py` | Krea single-file checkpoint discovery |

### Model discovery and metadata

| File | Covers |
| --- | --- |
| `tests/test_model_descriptor.py` | normalized model descriptor contract |
| `tests/test_model_discovery.py` | checkpoint/model discovery |
| `tests/test_metastore.py` | canonical metadata store (`models/metastore.py`) |
| `tests/test_lora_namespace.py` | LoRA/embedding namespace rules |
| `tests/test_lora_trigger_detection.py` | best-effort LoRA trigger auto-detection |

### Gallery, UI, and regressions

| File | Covers |
| --- | --- |
| `tests/test_thumbnails.py` | thumbnail generation and race safety |
| `tests/test_outpaint_pyramid_regression.py` | smart-extend/outpaint pyramid regression coverage |
| `tests/test_ui_sanity.py` | UI contract and markup sanity checks (running server + browser) |
| `tests/test_capability_driven_ui.py` | capability-driven frontend contract (no server needed) |
| `tests/test_ui_model_switch_contract.py` | model-switch UI/state contract |

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

### Identity / persona or backend adapter change

Run:

```bash
pytest tests/test_identity_adapter.py -v
pytest tests/test_flux_identity_personas.py -v
pytest tests/test_krea2_identity.py -v
pytest tests/test_krea2_identity_runtime.py -v
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
pytest tests/test_capability_driven_ui.py -v
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
- Add backend adapter tests in a dedicated `tests/test_<backend>_*.py` file (see the FLUX/Krea 2/Qwen groups above).
- Add identity/persona coverage in the matching `tests/test_*identity*` or `tests/test_*persona*` module.

Keep tests close to the subsystem they validate so future contributors can find them quickly.
