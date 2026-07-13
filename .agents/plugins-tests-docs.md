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
- `tests/test_prompt_conditioning.py`: long-prompt conditioning
- `tests/test_pipeline.py`: pipeline and scheduler behavior
- `tests/test_runtime.py`: runtime profile and fallback logic
- `tests/test_embedding_loading.py`: embedding loading behavior
- `tests/test_thumbnails.py`: thumbnail race safety
- `tests/test_outpaint_pyramid_regression.py`: smart-extend pyramid regressions
- `tests/test_ui_sanity.py`: browser sanity check against a running server

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
