# WebbDuck Repo Overview

WebbDuck is a local-first image generation studio with a FastAPI backend, a queue-first GPU worker, and a zero-build browser UI. Generation is model-first and architecture-agnostic: the selected checkpoint drives backend resolution, and architecture-specific code lives in `core/backends/` (SDXL, FLUX, Krea 2, Qwen image).

## Runtime Flow

1. `run.py` parses CLI flags and launches `webbduck.server.app:app` with Uvicorn.
2. `server/app.py` mounts the UI, registers HTTP and WebSocket routes, and starts background tasks.
3. Heavy work from `/generate`, `/test`, and `/upscale` is enqueued onto `generation_queue`.
4. `core/worker.py` processes jobs serially, acquires the shared GPU lease, and calls generation or upscaling code.
5. `core/generation.py` prepares inputs, chooses a mode, and drives the pipeline.
6. `server/storage.py` saves outputs, metadata, favorites state, and manifest entries.
7. `server/events.py` broadcasts `state`, `queue`, and `catalog` events to the UI.

## Top-Level Ownership

| Path | What it owns |
| --- | --- |
| `server/` | API routes, WebSocket, queue entrypoints, gallery APIs, thumbnails, model-catalog/provider/runtime-readiness routers |
| `core/` | worker loop, generation orchestration, runtime routing, pipeline lifecycle, runtime helpers, plugins |
| `core/backends/` | architecture-specific generation backends (SDXL, FLUX, Krea 2, Qwen) behind the contract in `core/backends/base.py` |
| `models/` | checkpoint, LoRA, embedding, descriptor, and upscaler discovery |
| `modes/` | text2img, img2img, inpaint, outpaint, and two-pass execution |
| `prompt/` | SDXL long-prompt conditioning and prompt utilities |
| `ui/` | HTML, ES modules, CSS, and client-side state |
| `plugins/` | bundled optional captioner and web-app plugins |
| `tests/` | pytest coverage, regressions, and browser sanity checks |
| `docs/` | contributor, user, plugin, and platform documentation |

## First File To Open By Change Type

- API or route change: `server/app.py`
- Worker or queue change: `core/worker.py`
- Generation behavior change: `core/generation.py` and the relevant file in `modes/`
- New backend or architecture-specific change: `core/backends/` (contract in `core/backends/base.py`) and `core/model_runtime.py`
- Runtime routing or readiness change: `core/model_runtime.py`, `models/discovery.py`, `models/catalog.py`, `server/runtime_readiness_api.py`
- Pipeline or model loading change: `core/pipeline.py` or `models/registry.py`
- Prompt handling change: `prompt/conditioning.py` or `prompt/management.py`
- Frontend control or workflow change: `ui/index.html`, `ui/app_main.js`, and `ui/core/state.js`
- Plugin integration change: `core/web_plugins.py`, `core/captioner.py`, or `core/captioning_config.py`
- Gallery or metadata change: `server/storage.py`, `server/thumbnails.py`, `ui/modules/GalleryManager.js`, and `ui/modules/LightboxManager.js`

## Current Entry Points

- App launcher: `run.py`
- Linux/WSL helper: `startup.sh`
- FastAPI app: `server/app.py`
- UI shell: `ui/index.html`
- Browser composition entrypoint: `ui/app.js`
- Main client orchestration: `ui/app_main.js`

## Documentation Pairings

- Repo map updates: `docs/ARCHITECTURE.md`
- Contributor workflow updates: `docs/DEVELOPMENT.md`
- User-facing workflow updates: `docs/USER_GUIDE.md` and `docs/SIMPLE_GUIDE.md`
- UI structure updates: `ui/README.md`
- Test workflow updates: `tests/README.md`
- Plugin contract updates: `docs/PLUGINS.md` and `plugins/README.md`
