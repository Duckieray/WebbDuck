# WebbDuck Development Guide

This guide covers the current backend and frontend extension points.

## Project Layout

```text
webbduck/
|- server/
|  |- app.py            # FastAPI endpoints, queue metadata, ws wiring
|  |- events.py         # socket broadcast helpers
|  |- state.py          # runtime stage/progress/vram snapshot state
|  |- storage.py        # output path helpers
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

## Backend Patterns

### Queue-first GPU execution

Any GPU-heavy endpoint should enqueue a job and let `core/worker.py` process it.

Current queue endpoint behavior:
- `POST /generate`, `POST /test`, `POST /upscale` all enqueue jobs.
- `wait_for_result=false` returns immediately with queue metadata.
- Queue snapshots are pushed over WebSocket (`type=queue`).

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
- Queue modal is separate from Studio controls.

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
