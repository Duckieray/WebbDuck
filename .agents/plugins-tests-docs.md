# Plugins, Tests, And Docs Reference

## Plugins

WebbDuck supports two optional plugin types:

- Captioners in `plugins/captioners/`
- Web-app plugins in `plugins/webapps/`

Search order for plugin roots:

1. `WEBBDUCK_PLUGINS_DIR`
2. `<repo>/plugins`
3. `~/.webbduck/plugins`

Key files:

- `core/captioning_config.py`: captioner discovery
- `core/captioner.py`: caption execution and unload flow
- `core/web_plugins.py`: local/remote web plugin loading and routing
- `docs/PLUGINS.md`: source-of-truth contract doc
- `plugins/README.md`: local plugin layout doc

## Tests

Primary focused suites:

- `tests/test_server.py`: API behavior and request validation
- `tests/test_server_captioning.py`: caption endpoints
- `tests/test_captioning.py`: captioning helpers
- `tests/test_modes.py`: mode selection and mode-specific request handling
- `tests/test_generation.py`: generation integration (GPU + models)
- `tests/test_pipeline.py`: pipeline and scheduler behavior
- `tests/test_prompt_conditioning.py`: long-prompt conditioning
- `tests/test_runtime.py`: runtime profile and fallback logic
- `tests/test_runtime_hardware.py`: hardware-aware runtime variants
- `tests/test_runtime_readiness_contract.py`: runtime readiness API contract
- `tests/test_state_vram.py`: state/VRAM accounting
- `tests/test_embedding_loading.py`: embedding loading behavior
- `tests/test_thumbnails.py`: thumbnail race safety
- `tests/test_outpaint_pyramid_regression.py`: smart-extend pyramid regressions
- `tests/test_metastore.py`: canonical metadata store
- `tests/test_identity_adapter.py`: IP-Adapter FaceID adapter
- `tests/test_identity_provider_ownership.py`: identity provider ownership/routing
- `tests/test_capability_driven_ui.py`: capability-driven frontend contract (no server)
- `tests/test_ui_sanity.py`: browser sanity check against a running server
- `tests/test_ui_model_switch_contract.py`: model-switch UI/state contract
- FLUX group: `test_flux_backend.py`, `test_flux_lora.py`, `test_flux_memory_metadata.py`, `test_flux_identity_personas.py`, `test_flux_identity_prep.py`, `test_flux_identity_reference_policy.py`, `test_flux2_identity_architecture.py`, `test_flux2_single_file.py`
- Krea 2 group: `test_krea2_backend.py`, `test_krea2_identity.py`, `test_krea2_identity_runtime.py`, `test_krea2_adaptive_runtime.py`, `test_krea2_identity_face.py`, `test_krea2_identity_multiref.py`, `test_krea2_identity_quality.py`, `test_krea2_identity_stability.py`, `test_krea2_lora.py`, `test_krea2_meta_overlay.py`, `test_krea2_native_fp8.py`, `test_krea2_runtime_policy.py`, `test_krea2_safe_runtime.py`, `test_krea2_single_file.py`
- Qwen group: `test_qwen_image_backend.py`
- Model discovery group: `test_model_catalog_api.py`, `test_model_descriptor.py`, `test_model_discovery.py`, `test_model_runtime_bridge.py`, `test_lora_namespace.py`, `test_lora_trigger_detection.py`
- Provider credentials group: `test_provider_credentials.py`, `test_provider_credentials_api.py`

See `tests/README.md` for the full test map grouped by subsystem.

`tests/test_ui_sanity.py` is not a pure unit test. It expects a running app and Playwright/browser dependencies.

## Documentation Rules

- Update docs in the same task when code changes routes, commands, workflows, file ownership, test expectations, or plugin contracts.
- Keep `AGENTS.md` and `.agents/*.md` aligned with the main tracked docs.
- Do not leave stale file paths, commands, or installer references in docs.
- If a referenced helper or script lives in an external plugin repo rather than WebbDuck, document it that way instead of implying it is bundled here.

## Useful Pairings

- Plugin changes: update `docs/PLUGINS.md` and `plugins/README.md`
- Frontend structure changes: update `ui/README.md`
- Test layout or workflow changes: update `tests/README.md`
- Repo map or file ownership changes: update `docs/ARCHITECTURE.md` and `.agents/repo-overview.md`
