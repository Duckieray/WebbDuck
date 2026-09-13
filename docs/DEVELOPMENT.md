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
- `server/model_catalog_api.py`: architecture-free model profile API (capabilities/defaults).
- `server/provider_credentials_api.py`: provider credential configuration endpoints.
- `server/runtime_readiness_api.py`: non-loading runtime readiness surface.

### Runtime and Generation

`server/app.py` runs a background watcher that checks:
- `checkpoint/sdxl/`
- `lora/`
- `lora/loras.json`
- `checkpoint/sdxl/checkpoints.json`
- `embeddings/` (EMBEDDING_ROOT)
- `embeddings/embeddings.json` (EMBEDDING_FILE)

- `core/worker.py`: queued execution.
- `core/generation.py`: normalized generation flow, runtime routing, and mode selection.
- `core/model_runtime.py`: checkpoint-driven runtime routing.
- `core/pipeline.py`: Diffusers lifecycle, model load/unload, LoRA application.
- `core/runtime.py`: runtime profile and device helpers.
- `core/gpu_lease.py`: in-process GPU arbitration.
- `core/backends/`: architecture-specific backends (SDXL, FLUX, Krea 2, Qwen image) behind the contract in `core/backends/base.py`.
- `modes/*.py`: per-mode generation logic.
- `prompt/*.py`: prompt conditioning and long-prompt behavior.

### Models and Assets

- `models/registry.py`: checkpoint, LoRA, embedding, and asset discovery.
- `models/discovery.py` and `models/catalog.py`: architecture-neutral checkpoint discovery and catalog used by runtime routing.
- `models/model_descriptor.py`: normalized descriptor with capabilities, constraints, defaults, and runtime hints.
- `models/quantization.py`: checkpoint quantization metadata helpers.
- `models/single_file_inspection.py`: structural inspection of single-file checkpoints via safetensors headers.
- `models/upscaler.py`: upscaler helpers.

### Frontend

- `ui/index.html`: markup for Studio, Gallery, dialogs, and controls.
- `ui/app.js`: stable browser composition entrypoint.
- `ui/app_main.js`: main page wiring and screen-level behaviors.
- `ui/core/api.js`: client request wrappers.
- `ui/core/utils.js`: shared DOM, download, form-data, and toast helpers.
- `ui/core/state.js`: persisted Studio state.
- `ui/core/events.js`: local event bus.
- `ui/core/modelCapabilities.js`: capability-driven gating from `/model-catalog`.
- `ui/modules/*.js`: feature modules like Gallery, Lightbox, LoRAs, embeddings, masks, progress, Identity/Persona, and provider credentials.
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

### Core

- `WEBBDUCK_OUTPUT_DIR`
- `WEBBDUCK_PORT`
- `WEBBDUCK_MODELS_DIR`
- `WEBBDUCK_CHECKPOINT_DIR`
- `WEBBDUCK_HF_CACHE_DIR`
- `WEBBDUCK_HF_CONFIG_DIR`
- `WEBBDUCK_WEIGHTS_DIR`
- `WEBBDUCK_PLUGINS_DIR`
- `WEBBDUCK_LORA_DIR`
- `WEBBDUCK_EMBEDDING_DIR`
- `WEBBDUCK_META_DIR` (default: `webbduck_meta/` in project root)
- `WEBBDUCK_CREDENTIALS_FILE` (default: `~/.webbduck/provider_credentials.json`)
- `WEBBDUCK_RUNTIME_HOME`
- `WEBBDUCK_RLIMIT_AS_GB` (`run.py`; address-space cap, disabled by default)

### Catalog and runtime behavior

- `WEBBDUCK_CATALOG_POLL_SECONDS` (default `3.0`)
- `WEBBDUCK_IDLE_UNLOAD_SECONDS` (default `300.0`, min `30.0`)
- `WEBBDUCK_THUMB_CONCURRENCY` (default `2`)
- `WEBBDUCK_DEVICE`
- `WEBBDUCK_DTYPE`
- `WEBBDUCK_STRICT_DEVICE`
- `WEBBDUCK_GPU_LEASE_WAIT_SECONDS` (default `180`)
- `WEBBDUCK_LEASE_IDLE_TIMEOUT`
- `WEBBDUCK_USE_IPC_COLLECT`
- `WEBBDUCK_SMART_EXTEND_DEBUG`
- `WEBBDUCK_SMART_EXTEND_DEBUG_DIR`

### Isolated backend runtimes

FLUX:

- `WEBBDUCK_FLUX`
- `WEBBDUCK_FLUX_PYTHON`
- `WEBBDUCK_FLUX_OFFLOAD`
- `WEBBDUCK_FLUX_TIMEOUT_SECONDS`
- `WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH`
- `WEBBDUCK_FLUX_IDENTITY_CACHE_DIR`

SDXL:

- `WEBBDUCK_SDXL_WORKER`
- `WEBBDUCK_SDXL_PYTHON`
- `WEBBDUCK_SDXL_TIMEOUT_SECONDS`
- `WEBBDUCK_SDXL_TOKENIZE_TIMEOUT_SECONDS`

Qwen image:

- `WEBBDUCK_QWEN_IMAGE_PYTHON`
- `WEBBDUCK_QWEN_IMAGE_OFFLOAD`
- `WEBBDUCK_QWEN_IMAGE_TIMEOUT_SECONDS`

Krea 2 (see `docs/KREA2_PERFORMANCE.md`):

- `WEBBDUCK_KREA`
- `WEBBDUCK_KREA2_COMPONENT_MODEL`
- `WEBBDUCK_KREA2_TURBO_COMPONENT_MODEL`
- `WEBBDUCK_KREA2_BASE_COMPONENT_MODEL`
- `WEBBDUCK_KREA2_OFFLOAD`
- `WEBBDUCK_KREA2_BLOCKS_PER_GROUP`
- `WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM`
- `WEBBDUCK_KREA2_IDENTITY_WEIGHT`
- `WEBBDUCK_KREA2_IDENTITY_REPO`
- `WEBBDUCK_KREA2_IDENTITY_PROCESSOR`
- `WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET`
- `WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS`
- `WEBBDUCK_KREA2_IDENTITY_STEPS`
- `WEBBDUCK_KREA2_IDENTITY_GUIDANCE`
- `WEBBDUCK_KREA2_IDENTITY_CFG_FREE`
- `WEBBDUCK_KREA2_IDENTITY_PAIRED`
- `WEBBDUCK_KREA2_IDENTITY_UPSCALE`
- `WEBBDUCK_KREA2_IDENTITY_MULTIREF`
- `WEBBDUCK_KREA2_IDENTITY_MAX_REFERENCE_EDGE`
- `WEBBDUCK_KREA2_IDENTITY_AUTO_FACE_CROP`
- `WEBBDUCK_KREA2_IDENTITY_FACE_FOCUS`
- `WEBBDUCK_KREA2_IDENTITY_SUBJECT_EDGE`
- `WEBBDUCK_KREA2_IDENTITY_SUBJECT_REF_BOOST`
- `WEBBDUCK_KREA2_IDENTITY_QUALITY_BASELINE`
- `WEBBDUCK_KREA2_IDENTITY_ALLOW_TEXT_ONLY`
- `WEBBDUCK_KREA2_DEBUG_PRINT`

## Runtime Requirements Files

Optional per-backend dependency lists live in `runtime_requirements/` (`flux.txt`, `krea2.txt`, `qwen_image.txt`, `sdxl.txt`) for isolated runtimes whose dependencies may differ from WebbDuck's stable environment. Tools under `tools/` (`prepare_model_runtimes.py`, `run_hardware_smoke.py`) help set up and probe those environments; results are tracked in `docs/HARDWARE_SMOKE_MATRIX.md`.

## Documentation Update Checklist

When a change ships, update the smallest set of docs that now differ from reality:

- `README.md` for install, startup, or top-level capability changes.
- `.agents/*.md` when agent-facing repo orientation or workflow guidance changes.
- `docs/ARCHITECTURE.md` if file ownership or repo layout changes.
- `docs/DEVELOPMENT.md` if contributor workflows change.
- `docs/USER_GUIDE.md` or `docs/SIMPLE_GUIDE.md` if the user workflow changes.
- `docs/PLUGINS.md` and `plugins/README.md` if plugin contracts change.
- `ui/README.md` and `tests/README.md` when frontend or validation structure changes.
