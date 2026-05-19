# WebbDuck Development Guide

Use this guide when you are changing WebbDuck itself. It focuses on where to edit, how the major pieces connect, and which checks to run before you stop.

## Start Here

Read these docs in order when you are new to the repo:

1. `docs/ARCHITECTURE.md` for the repo map and ownership.
2. `AGENTS.md` for project rules, guardrails, and standard commands.
3. `.agents/README.md` for the committed agent reference set.
4. `ui/README.md`, `tests/README.md`, or `docs/PLUGINS.md` if you are working in those areas.

## Setup

### Linux / WSL

```bash
conda create -n webbduck python=3.10 -y
conda activate webbduck
pip install -r requirements.txt
mkdir -p checkpoint/sdxl lora embeddings outputs weights
```

### Windows

Use `docs/WINDOWS_TESTING.md` for the full Windows-native setup and smoke-test flow.

## Run

### Scripted launcher

```bash
./startup.sh
./startup.sh --env webbduck --output ./outputs --port 8020
./startup.sh --models /path/to/models --hf-cache /path/to/huggingface-cache --port 8020
```

### Direct launcher

```bash
conda activate webbduck
python run.py --output ./outputs --port 8010
```

`startup.sh` is the easiest way to run with environment overrides on Linux/WSL. `run.py` is the simplest entrypoint for direct debugging.

## Architecture Rules

- Queue-first GPU execution: request handlers should enqueue heavy work for `core/worker.py`.
- Shared GPU lease: coordinate in-process GPU access through `core/gpu_lease.py`.
- Keep backend and frontend payloads aligned whenever request fields change.
- Preserve catalog refresh behavior for checkpoints, LoRAs, embeddings, and weights.
- Prefer app modals over browser popup APIs.
- Keep plugins optional and non-blocking for core generation.

## Where To Make Changes

### Backend and API

- `server/app.py`: routes, payload parsing, queue entrypoints, plugin endpoints, and API-facing job/error responses.
- `server/events.py`: WebSocket broadcasts.
- `server/state.py`: shared runtime status snapshot.
- `server/storage.py`: output metadata and gallery manifest.
- `server/thumbnails.py`: thumbnail generation.

### Runtime and Generation

`server/app.py` runs a background watcher that checks:
- `checkpoint/sdxl/`
- `lora/`
- `lora/loras.json`
- `checkpoint/sdxl/checkpoints.json`

- `core/worker.py`: queued execution.
- `core/generation.py`: normalized generation flow and mode selection.
- `core/pipeline.py`: Diffusers lifecycle, model load/unload, LoRA application.
- `core/runtime.py`: runtime profile and device helpers.
- `core/gpu_lease.py`: in-process GPU arbitration.
- `modes/*.py`: per-mode generation logic.
- `prompt/*.py`: prompt conditioning and long-prompt behavior.

### Models and Assets

- `models/registry.py`: checkpoint, LoRA, embedding, and asset discovery.
- `models/upscaler.py`: upscaler helpers.

### Frontend

- `ui/index.html`: markup for Studio, Gallery, dialogs, and controls.
- `ui/app.js`: main page wiring and screen-level behaviors.
- `ui/core/api.js`: client request wrappers.
- `ui/core/utils.js`: shared DOM, download, form-data, and toast helpers.
- `ui/core/state.js`: persisted Studio state.
- `ui/core/events.js`: local event bus.
- `ui/modules/*.js`: feature modules like Gallery, Lightbox, LoRAs, embeddings, masks, and progress.
- `ui/styles/`: tokens, layout, components, and theme styles.

## Common Update Recipes

### Add a New Control That Affects Generation

1. Add the markup in `ui/index.html`.
2. Read and write it in `ui/app.js`.
3. Persist it in `ui/core/state.js` if it should survive refreshes.
4. Send it through `ui/core/api.js`.
5. Parse and validate it in `server/app.py`.
6. Apply it in `core/generation.py`, `core/pipeline.py`, or the relevant mode file.
7. Add focused tests and update docs.

### Add a New Endpoint

1. Register the route in `server/app.py`.
2. Enqueue GPU-heavy work instead of running it inline.
3. Add a frontend API wrapper if the UI consumes it.
4. Add a test in `tests/test_server.py` or another focused test module.

### Change LoRA or Embedding Behavior

1. Start in `models/registry.py`.
2. Check any request/response changes in `server/app.py`.
3. Update `ui/modules/LoraManager.js` or `ui/modules/EmbeddingManager.js`.
4. Run focused tests such as `tests/test_embedding_loading.py` and `tests/test_server.py`.

### Change Gallery Behavior

1. Update persistence or search behavior in `server/storage.py`.
2. Update gallery endpoints in `server/app.py` if needed.
3. Update rendering and interaction logic in `ui/modules/GalleryManager.js` and `ui/modules/LightboxManager.js`.
4. Run `tests/test_thumbnails.py` plus the relevant server tests.

### Change Plugin Integration

1. Update `core/captioning_config.py`, `core/captioner.py`, or `core/web_plugins.py`.
2. Keep failures isolated so WebbDuck still works without the plugin.
3. Update `docs/PLUGINS.md` and `plugins/README.md`.

## Testing Tiers

### Fast targeted checks

```bash
conda run -n webbduck pytest tests/test_server.py -v
conda run -n webbduck pytest tests/test_prompt_conditioning.py -v
conda run -n webbduck pytest tests/test_ui_sanity.py -v
```

### Default non-slow suite

```bash
conda run -n webbduck pytest -v -m "not slow"
```

### Broader validation when needed

```bash
conda run -n webbduck pytest -v
```

Use `tests/README.md` to pick the narrowest suite that still covers your change.

## Environment Variables

- `WEBBDUCK_OUTPUT_DIR`
- `WEBBDUCK_PORT`
- `WEBBDUCK_MODELS_DIR`
- `WEBBDUCK_CHECKPOINT_DIR`
- `WEBBDUCK_HF_CACHE_DIR`
- `WEBBDUCK_WEIGHTS_DIR`
- `WEBBDUCK_PLUGINS_DIR`
- `WEBBDUCK_LORA_DIR`
- `WEBBDUCK_EMBEDDING_DIR`
- `WEBBDUCK_CATALOG_POLL_SECONDS`
- `WEBBDUCK_THUMB_CONCURRENCY`
- `WEBBDUCK_DEVICE`
- `WEBBDUCK_DTYPE`
- `WEBBDUCK_STRICT_DEVICE`
- `WEBBDUCK_GPU_LEASE_WAIT_SECONDS`
- `WEBBDUCK_USE_IPC_COLLECT`

## Documentation Update Checklist

When a change ships, update the smallest set of docs that now differ from reality:

- `README.md` for install, startup, or top-level capability changes.
- `.agents/*.md` when agent-facing repo orientation or workflow guidance changes.
- `docs/ARCHITECTURE.md` if file ownership or repo layout changes.
- `docs/DEVELOPMENT.md` if contributor workflows change.
- `docs/USER_GUIDE.md` or `docs/SIMPLE_GUIDE.md` if the user workflow changes.
- `docs/PLUGINS.md` and `plugins/README.md` if plugin contracts change.
- `ui/README.md` and `tests/README.md` when frontend or validation structure changes.
