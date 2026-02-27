# WebbDuck Development Guide

This guide covers the current backend and frontend extension points.

## Project Layout

```text
webbduck/
|- server/
|  |- app.py            # FastAPI endpoints, queue metadata, ws wiring
|  |- events.py         # socket broadcast helpers
|  |- state.py          # runtime stage/progress/vram snapshot state
|  |- storage.py        # output persistence + searchable manifest index
|  |- thumbnails.py     # thumbnail generation
|- core/
|  |- worker.py         # GPU worker loop
|  |- generation.py     # mode selection + prompt/pipeline execution
|  |- pipeline.py       # pipeline lifecycle, scheduler + LoRA application
|  |- schedulers.py     # scheduler registry
|- models/
|  |- registry.py       # model/LoRA discovery and registry sync
|- ui/
|  |- app.js
|  |- core/
|  |- modules/
|  |- styles/
```

## High-Level Runtime Flow

1. UI submits an action (`/generate`, `/test`, `/upscale`, etc.).
2. Backend validates and enqueues GPU work.
3. `core/worker.py` processes jobs serially.
4. `core/generation.py` selects mode and runs the pipeline.
5. Outputs + metadata are written to `outputs/`.
6. WebSocket updates push `state`, `queue`, and `catalog` events to clients.

## Backend Patterns

### Queue-first GPU execution

Any GPU-heavy endpoint should enqueue a job and let `core/worker.py` process it.

Current queue endpoint behavior:
- `POST /generate`, `POST /test`, `POST /upscale` all enqueue jobs.
- `wait_for_result=false` returns immediately with queue metadata.
- Queue snapshots are pushed over WebSocket (`type=queue`).

### Gallery search index

- Search uses `outputs/manifest.jsonl` built and maintained by `server/storage.py`.
- Manifest entries are appended on save and pruned on image/run delete.
- Search endpoint:
  - `GET /gallery/search?q=...&start=...&limit=...`
  - keyword-based matching with relevance ranking.

### Job shape

```python
job = {
    "job_id": "uuid",
    "type": "batch|test|upscale|...",
    "settings": {...},
    "future": future,
    "on_start": callback,
    "on_finish": callback,
}
```

### WebSocket payload types

- `state`: progress/stage/VRAM runtime info.
- `queue`: queue/running/recent-completed metadata.
- `catalog`: model/LoRA catalog changed; UI should refresh model lists.

## Catalog Refresh

`server/app.py` runs a background watcher that checks:
- `checkpoint/sdxl/`
- `lora/`
- `lora/loras.json`
- `checkpoint/sdxl/models.json`

If signatures change, `refresh_registries()` runs and `catalog` is broadcast.

Env var:
- `WEBBDUCK_CATALOG_POLL_SECONDS` (default `3.0`).

## LoRA Behavior

- LoRAs are architecture-filtered per base model.
- Local `.safetensors` additions are auto-synced into `lora/loras.json`.
- API returns LoRA default `weight` and description.
- During generation, pipeline returns a combined trigger phrase.
- `core/generation.py` injects that trigger phrase into active prompt fields.

## Frontend Patterns

### State persistence

`ui/core/state.js` persists Studio settings to `localStorage` (`webbduck_state_v2`).

### Queue UI

- Queue list is updated from WebSocket `queue` events.
- Users can expand job details and cancel queued jobs.
- Running jobs can be cancellation-requested (status moves to `cancelling`).
- Queue modal is separate from Studio controls.

### Progress + settings surface

- Main status indicator is always visible in top-right (`Ready`/stage + percentage).
- Detailed generation progress card is docked in the Studio status bar.
- Settings modal contains long-run warning threshold and CLIP_SKIP2 toggle.

### Mobile Studio pane mode

- Studio on mobile supports explicit pane switching:
  - `Settings` pane (controls)
  - `Preview` pane (workspace)
- Toggle is managed in `ui/app.js` and styled in `ui/styles/theme-nova.css`.

### Catalog updates

`ui/core/events.js` emits `Events.CATALOG_UPDATE`; `ui/app.js` reloads models/loras when this event arrives.

## Adding an Endpoint

1. Add route in `server/app.py`.
2. If GPU-bound, enqueue and handle in `core/worker.py`.
3. Add wrapper in `ui/core/api.js`.
4. Add UI behavior in `ui/app.js` or a feature module.
5. Add/adjust tests under `tests/`.

## Testing

```bash
pytest -v
pytest tests/test_server.py -v
pytest tests/test_modes.py -v
```

Use the project conda environment if you run tests through WSL.

## API Surface (Common Endpoints)

- `POST /generate`
- `POST /test`
- `POST /upscale`
- `GET /models`
- `GET /second_pass_models`
- `GET /models/{base_model:path}/loras`
- `GET /models/{base_model:path}/embeddings`
- `GET /gallery`
- `GET /gallery/search`
- `GET /queue`
- `POST /queue/cancel`
- `POST /tokenize`
- `POST /caption`
- `GET /captioners`
- `GET /caption_styles`
- `GET /health`
- `GET /ws` (WebSocket)

## Runtime Environment Knobs

- `WEBBDUCK_OUTPUT_DIR`
- `WEBBDUCK_PORT`
- `WEBBDUCK_LORA_DIR`
- `WEBBDUCK_EMBEDDING_DIR`
- `WEBBDUCK_CATALOG_POLL_SECONDS`
- `WEBBDUCK_THUMB_CONCURRENCY`
- `WEBBDUCK_DEVICE`
- `WEBBDUCK_DTYPE`
- `WEBBDUCK_STRICT_DEVICE`
- `WEBBDUCK_GPU_LEASE_WAIT_SECONDS`

## Cleanup Notes

- Legacy UI entrypoints/modules from the pre-Nova path were removed:
  - `ui/app_old.js`
  - `ui/index_old.html`
  - lowercase legacy modules in `ui/modules/`
  - legacy `ui/ws.js` and old monolithic CSS files
- Active app flow is now centered on:
  - `ui/app.js`
  - `ui/core/*`
  - `ui/modules/*` (PascalCase modules)
  - `ui/styles/main.css` + Nova theme stack
- For new cleanup passes, keep enforcing “no browser popup APIs” (`alert`, `confirm`) in active UI code; use modal components instead.
