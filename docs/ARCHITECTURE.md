# WebbDuck Architecture Map

This is the quickest repo tour for contributors and agents. Use it when you need to find the right file before making a change.

## What WebbDuck Is

WebbDuck is a local-first SDXL studio:

- FastAPI serves the API, WebSocket updates, thumbnails, and plugin routes.
- A single queued worker handles GPU-heavy jobs.
- The browser UI is plain HTML, ES modules, and CSS with no Node build step.
- Outputs are stored on disk with sidecar metadata and a manifest-backed gallery index.
- Optional captioner and web-app plugins can extend the app without becoming hard dependencies.

### Planned architecture evolution

The current implementation is SDXL-shaped internally, but the intended product architecture is **model-first and architecture-agnostic**. Users should continue selecting a model/checkpoint while WebbDuck automatically detects its architecture, resolves a compatible backend/runtime, applies model-specific defaults, and exposes only supported capabilities.

Read `docs/ARCHITECTURE_AGNOSTIC_GENERATION.md` before implementing new model architectures such as FLUX or Krea 2. That document is the migration/design source of truth; this file continues to describe the current repository layout until the refactor lands.

## Runtime Flow

1. `ui/index.html` loads `ui/app.js` and the feature modules.
2. The UI calls API helpers in `ui/core/api.js`.
3. `server/app.py` validates requests, enqueues heavy work, and exposes read APIs.
4. `core/worker.py` processes generation and upscale jobs serially.
5. `core/generation.py` selects the mode and calls the pipeline/runtime helpers.
6. `server/storage.py` writes files, metadata, and gallery manifest entries.
7. `server/events.py` pushes `state`, `queue`, and `catalog` updates to connected clients.

## Repo Map

### Top-Level Folders

| Path | Purpose | Update here when... |
| --- | --- | --- |
| `server/` | FastAPI routes, WebSocket handling, gallery APIs, thumbnail serving | you add or change HTTP/WebSocket behavior |
| `core/` | worker loop, generation orchestration, pipeline lifecycle, GPU lease, plugins | you touch execution flow, runtime coordination, or plugin loading |
| `models/` | checkpoint, LoRA, embedding, and upscaler discovery helpers | you change asset discovery or model metadata |
| `modes/` | text2img, img2img, inpaint, outpaint, two-pass mode implementations | you change how a generation mode works |
| `prompt/` | SDXL prompt conditioning, chunking, prompt management | you change tokenization or long-prompt behavior |
| `ui/` | browser UI, styles, local state, feature modules | you change controls, layout, or client-side behavior |
| `plugins/` | bundled optional plugin examples and manifests | you update plugin examples or bundled plugin docs |
| `tests/` | pytest coverage for server, runtime, modes, UI sanity, and regressions | you add or update validation |
| `docs/` | user, contributor, plugin, and platform docs | behavior or workflows change |

### Backend Map

| Path | Responsibility |
| --- | --- |
| `server/app.py` | main application, route registration, queue entrypoints, plugin endpoints |
| `server/events.py` | broadcast helpers for UI realtime updates |
| `server/state.py` | runtime status snapshot shared with the UI |
| `server/storage.py` | output persistence, metadata files, gallery manifest management |
| `server/thumbnails.py` | thumbnail generation and caching |
| `core/worker.py` | queued execution of GPU-heavy jobs |
| `core/generation.py` | request normalization and mode dispatch |
| `core/pipeline.py` | Diffusers pipeline load/unload and LoRA application |
| `core/runtime.py` | runtime profile helpers and device/dtype detection |
| `core/gpu_lease.py` | shared in-process GPU ownership across core and plugins |
| `core/captioning_config.py` | captioner plugin discovery and plugin root resolution |
| `core/web_plugins.py` | local/remote web-app plugin discovery and mounting |
| `core/captioner.py` | optional captioner loading and caption execution |
| `models/registry.py` | discovered checkpoints, LoRAs, embeddings, and catalog refresh |
| `models/upscaler.py` | upscaler discovery/loading helpers |
| `models/metastore.py` | canonical metadata store (alongside registry.py and upscaler.py) |
| `core/schedulers.py` | scheduler implementations |
| `modes/base.py` | base class for generation modes |
| `core/perf.py` | performance tracking |
| `core/exceptions.py` | custom exceptions |

### Frontend Map

| Path | Responsibility |
| --- | --- |
| `ui/index.html` | app shell and control markup |
| `ui/app.js` | top-level wiring for Studio, Queue, Settings, plugins, and page lifecycle |
| `ui/core/api.js` | fetch wrappers for backend endpoints |
| `ui/core/events.js` | local event bus fed by WebSocket updates |
| `ui/core/state.js` | persisted Studio state and DOM sync helpers |
| `ui/core/utils.js` | DOM helpers, form-data helpers, downloads, and toast utilities |
| `ui/modules/GalleryManager.js` | gallery loading, search, filters, infinite scroll |
| `ui/modules/LightboxManager.js` | viewer interactions and metadata rendering |
| `ui/modules/LoraManager.js` | LoRA picker, weights, persistence |
| `ui/modules/EmbeddingManager.js` | embedding picker and token editing |
| `ui/modules/MaskEditor.js` | inpaint mask editing |
| `ui/modules/ProgressManager.js` | progress card and cancel flow |
| `ui/styles/` | design tokens, resets, layout, and theme styling |
| `ui/planning/` | scratch area for UI planning notes; currently empty in the tracked repo |

## Common Change Recipes

### Add or Change a Backend Setting

1. Update request parsing in `server/app.py`.
2. Thread the value into the generation settings/job payload.
3. Apply it in `core/generation.py`, `core/pipeline.py`, or the relevant file in `modes/`.
4. If the UI exposes it, wire it through `ui/index.html`, `ui/app.js`, and `ui/core/state.js`.
5. Add or update tests in `tests/`.
6. Update docs in `README.md`, `docs/DEVELOPMENT.md`, and any user-facing guide if behavior changed.

### Add a New API Endpoint

1. Add the route in `server/app.py`.
2. If it is GPU-heavy, enqueue work instead of running inline.
3. Add a client wrapper in `ui/core/api.js` if the UI needs it.
4. Update UI behavior in `ui/app.js` or a dedicated module.
5. Add focused tests, usually in `tests/test_server.py`.

### Change Model, LoRA, or Embedding Discovery

1. Start with `models/registry.py`.
2. Check whether catalog refresh signatures or watcher behavior in `server/app.py` also need updates.
3. Verify API payloads consumed by `ui/modules/LoraManager.js` or `ui/modules/EmbeddingManager.js` still match.
4. Add or update tests such as `tests/test_embedding_loading.py` or `tests/test_server.py`.

### Change Gallery or Storage Behavior

1. Update persistence/indexing code in `server/storage.py`.
2. Check route behavior in `server/app.py` and thumbnail behavior in `server/thumbnails.py`.
3. Verify UI expectations in `ui/modules/GalleryManager.js` and `ui/modules/LightboxManager.js`.
4. Run focused tests like `tests/test_thumbnails.py` and gallery-related server tests.

### Change Plugin Integration

1. Update plugin discovery/config in `core/captioning_config.py` or `core/web_plugins.py`.
2. Keep missing plugins non-fatal for the rest of the app.
3. Document any manifest, install, or routing changes in `docs/PLUGINS.md` and `plugins/README.md`.

## Important Constraints

- Queue GPU-heavy work; do not run it directly in request handlers.
- Keep frontend request payloads aligned with backend parsing.
- Preserve catalog refresh behavior for checkpoints, LoRAs, and embeddings.
- Coordinate in-process GPU work through `core/gpu_lease.py`.
- Prefer app modals over browser `alert`, `confirm`, or `prompt`.
- Keep optional plugins optional; failures should be visible but not stall core generation.

## Documentation Ownership

Use these docs as the main source of truth:

- `README.md`: product overview, setup, quickstart, docs index
- `docs/ARCHITECTURE.md`: repo map and file responsibilities
- `docs/ARCHITECTURE_AGNOSTIC_GENERATION.md`: target model-first generation architecture and migration plan
- `docs/DEVELOPMENT.md`: contributor workflows and change recipes
- `docs/USER_GUIDE.md`: end-user workflow
- `docs/PLUGINS.md`: captioner and web-app plugin integration
- `docs/WINDOWS_TESTING.md`: Windows-native setup and smoke tests
- `ui/README.md`: frontend structure and editing guidance
- `tests/README.md`: test selection and extension guide

When you add a feature, update the narrowest affected doc and then check whether one of the docs above should also change.
