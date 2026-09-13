# WebbDuck Architecture Map

This is the quickest repo tour for contributors and agents. Use it when you need to find the right file before making a change.

## What WebbDuck Is

WebbDuck is a local-first image generation studio with SDXL, FLUX, Krea 2, and Qwen image backends:

- FastAPI serves the API, WebSocket updates, thumbnails, and plugin routes.
- A single queued worker handles GPU-heavy jobs.
- The browser UI is plain HTML, ES modules, and CSS with no Node build step.
- Outputs are stored on disk with sidecar metadata and a manifest-backed gallery index.
- Optional captioner and web-app plugins can extend the app without becoming hard dependencies.
- Backends in `core/backends/` wrap architecture-specific runtimes behind a generic generation contract.

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
| `core/` | worker loop, generation orchestration, runtime routing, backend adapters, GPU lease, plugins | you touch execution flow, runtime coordination, or plugin loading |
| `core/backends/` | architecture-specific generation backends (SDXL, FLUX, Krea 2, Qwen) | you add or change a model backend |
| `models/` | checkpoint, LoRA, embedding, descriptor, and upscaler discovery | you change asset discovery or model metadata |
| `modes/` | text2img, img2img, inpaint, outpaint, two-pass mode implementations | you change how a generation mode works |
| `prompt/` | prompt conditioning, chunking, prompt management | you change tokenization or long-prompt behavior |
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
| `server/model_catalog_api.py` | architecture-free model profile API from the runtime catalog (capabilities/defaults) |
| `server/provider_credentials_api.py` | Hugging Face/Civitai provider credential configuration endpoints |
| `server/runtime_readiness_api.py` | non-loading runtime readiness surface for checkpoint-driven generation |
| `core/worker.py` | queued execution of GPU-heavy jobs |
| `core/generation.py` | request normalization, runtime routing, and mode dispatch |
| `core/model_runtime.py` | checkpoint-driven runtime routing for generation jobs |
| `core/pipeline.py` | Diffusers pipeline load/unload and LoRA application |
| `core/runtime.py` | runtime profile helpers and device/dtype detection |
| `core/gpu_lease.py` | shared in-process GPU ownership across core and plugins |
| `core/provider_credentials.py` | host-side provider credential storage; never returns secret values to clients |
| `core/captioning_config.py` | captioner plugin discovery and plugin root resolution |
| `core/web_plugins.py` | local/remote web-app plugin discovery and mounting |
| `core/captioner.py` | optional captioner loading and caption execution |
| `core/schedulers.py` | scheduler implementations |
| `core/perf.py` | performance tracking |
| `core/exceptions.py` | custom exceptions |
| `models/registry.py` | discovered checkpoints, LoRAs, embeddings, and catalog refresh |
| `models/discovery.py` | architecture-neutral checkpoint discovery (independent of execution) |
| `models/catalog.py` | architecture-neutral checkpoint catalog used by runtime routing |
| `models/model_descriptor.py` | normalized model descriptor: capabilities, constraints, defaults, runtime hints |
| `models/quantization.py` | architecture-neutral checkpoint quantization metadata helpers |
| `models/single_file_inspection.py` | cheap structural inspection of single-file checkpoints via safetensors headers |
| `models/upscaler.py` | upscaler discovery/loading helpers |
| `models/metastore.py` | canonical metadata store |
| `modes/base.py` | base class for generation modes |

#### `core/backends/` — generation backend adapters

Backends wrap architecture/runtime-specific behavior behind the contract in `core/backends/base.py`; higher layers route by model descriptor rather than architecture names.

| Path | Responsibility |
| --- | --- |
| `base.py` | backend contract for checkpoint-driven generation |
| `runtime_probe.py` | lightweight probes for isolated model-runtime interpreters |
| `flux.py` | FLUX image backend (isolated Python runtime) |
| `flux_worker.py` | standalone FLUX Diffusers worker |
| `flux_lora.py` | shared FLUX LoRA normalization and worker-side adapter loading |
| `flux_identity.py` | FLUX.2 persona preprocessing: face cropping, anchor weighting, face-focus prompt |
| `sdxl.py` | SDXL backend exposed through the generic generation contract |
| `sdxl_pipeline.py` | backend-owned SDXL pipeline manager |
| `sdxl_runtime.py` | SDXL isolated-runtime wiring |
| `sdxl_worker.py` | SDXL worker entrypoints |
| `qwen_image.py` | Qwen-Image-2512 backend in an isolated Diffusers runtime |
| `qwen_image_worker.py` | Qwen image worker entrypoint |
| `krea2.py` | Krea 2 isolated-runtime backend: identity payload normalization, component resolution, worker launch |
| `krea2_host_impl.py` | Krea 2 image backend entrypoint for an isolated Python runtime |
| `krea2_worker.py` | Krea phased GPU worker; identity edit run, FP8 LoRA residuals, OOM demotion |
| `krea2_worker_impl.py` | shared Krea worker implementation |
| `krea2_worker_adaptive.py` | request-level adaptive planning incl. identity token accounting and geometry scaling |
| `krea2_worker_safe.py` | safe-runtime wrapper dispatching identity runs to the phased worker |
| `krea2_identity.py` | Krea identity contract layer: validation, preset snapshot, weight resolution, edit geometry helpers |
| `krea2_identity_face.py` | Krea identity face-crop / face-focus reference handling |
| `krea2_identity_multiref.py` | Krea multi-reference identity handling |
| `krea2_identity_quality.py` | Krea identity quality gates and artifact/upscale handling |
| `krea2_identity_reference.py` | Krea identity reference policy and auto face-crop defaults |
| `krea2_identity_stability.py` | Krea identity stability handling |
| `krea2_lora.py` | Krea LoRA normalization/loading |
| `krea2_native_fp8.py` | native FP8 GEMM kernel + post-GEMM LoRA residual overlay |
| `krea2_weights.py` | Krea single-file/GGUF weight loading and metadata overlay |

### Frontend Map

| Path | Responsibility |
| --- | --- |
| `ui/index.html` | app shell and control markup |
| `ui/app.js` | stable browser composition entrypoint importing the main app |
| `ui/app_main.js` | Studio, Queue, Gallery, Settings, help modal, and plugin-tab orchestration |
| `ui/core/api.js` | fetch wrappers for backend endpoints |
| `ui/core/events.js` | local event bus fed by WebSocket updates |
| `ui/core/state.js` | persisted Studio state and DOM sync helpers (`webbduck_state_v2`) |
| `ui/core/utils.js` | DOM helpers, form-data helpers, downloads, and toast utilities |
| `ui/core/modelCapabilities.js` | capability-driven model profile layer gating controls from `/model-catalog` |
| `ui/modules/GalleryManager.js` | gallery loading, search, filters, infinite scroll |
| `ui/modules/LightboxManager.js` | viewer interactions and metadata rendering |
| `ui/modules/LoraManager.js` | LoRA picker, weights, persistence |
| `ui/modules/EmbeddingManager.js` | embedding picker and token editing |
| `ui/modules/MaskEditor.js` | inpaint mask editing |
| `ui/modules/ProgressManager.js` | progress card and cancel flow |
| `ui/modules/PersonaIdentityUI.js` | Identity / Persona presentation over the shared reference + preset manager |
| `ui/modules/PersonaIdentityUI_impl.js` | PersonaIdentityUI implementation details |
| `ui/modules/ProviderCredentialsSettings.js` | optional provider credential controls injected into Settings |
| `ui/styles/` | reset, design tokens, layout, components, and theme styling |
| `ui/lib/` | vendored frontend libraries (PhotoSwipe) |
| `ui/manifest.json` | PWA/app manifest |

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
- `docs/PROVIDER_CREDENTIALS.md`: model-provider credential setup and host-side storage
- `docs/KREA2_PERFORMANCE.md`: Krea 2 backend performance and tuning notes
- `docs/KREA2_IDENTITY_QUALITY_FIX.md`: Krea identity quality-fix implementation notes
- `docs/HARDWARE_SMOKE_MATRIX.md`: hardware smoke-test matrix results
- `ui/README.md`: frontend structure and editing guidance
- `tests/README.md`: test selection and extension guide

When you add a feature, update the narrowest affected doc and then check whether one of the docs above should also change.
