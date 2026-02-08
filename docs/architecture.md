# Architecture Overview

WebbDuck is split into a FastAPI backend, a single GPU worker queue, and a zero-build ES module frontend.

## High-Level Flow

1. UI submits generation or utility action.
2. Backend validates request and enqueues GPU jobs.
3. `core/worker.py` processes jobs serially on GPU.
4. `core/generation.py` selects mode and executes pipeline.
5. Outputs and metadata are written to disk.
6. Queue/progress/catalog updates are pushed to clients over WebSocket.

## Backend Components

### `server/app.py`

- API endpoints for generate/test/upscale/gallery/models/loras/tokenize/caption.
- `generation_queue` (`asyncio.Queue`, maxsize 32).
- Job lifecycle tracking in `job_registry`.
- Queue endpoints:
  - `GET /queue`
  - `POST /queue/cancel`
- WebSocket endpoint: `GET /ws` (upgrades to ws) for push updates.
- Catalog watcher task scans model/LoRA files and broadcasts `catalog` updates.

### `core/worker.py`

- Dedicated async loop that dequeues and runs GPU work.
- Handles `batch`, `test`, and `upscale` job types.
- Invokes per-job start/finish hooks so queue metadata stays synchronized.

### `core/generation.py`

- Selects generation mode (txt2img, img2img, inpaint, two-pass).
- Applies LoRAs through pipeline manager.
- Injects computed LoRA trigger phrase into prompt(s) before generation.
- Runs mode with callback-based progress updates.

### `models/registry.py`

- Auto-discovers local checkpoints and LoRAs.
- Maintains `checkpoint/sdxl/models.json` defaults.
- Maintains and syncs `lora/loras.json` entries with local `.safetensors` files.
- Supports architecture-aware model/LoRA filtering.

## Frontend Components

### `ui/app.js`

- Bootstraps Studio, Gallery, queue modal, and lightbox actions.
- Builds `FormData` requests for generation endpoints.
- Renders queue modal rows and cancel/detail actions.
- Syncs resolution preset chips and custom state.
- Token counter and over-limit warning tooltip.

### `ui/modules/LoraManager.js`

- Loads LoRAs for selected base model.
- Uses registry default weights from API metadata.
- Stores selected LoRAs with live weight updates (`step=0.05`).
- Restores selected LoRAs from persisted state when possible.

### `ui/core/events.js`

- In-app event bus.
- WebSocket bridge emitting:
  - `status:update` for runtime/progress,
  - `queue:update` for queue/job metadata,
  - `catalog:update` for model/LoRA catalog changes.

## Runtime Data Paths

- Output images and metadata: `outputs/<run>/...`
- Input/upload temp files for img2img/inpaint/caption: `inpaint_input/`
- LoRA registry file: `lora/loras.json`
- Model defaults file: `checkpoint/sdxl/models.json`

## Performance Controls

- Queue serialization avoids concurrent GPU contention.
- Thumbnail generation is bounded by semaphore (`WEBBDUCK_THUMB_CONCURRENCY`).
- Catalog polling interval is configurable (`WEBBDUCK_CATALOG_POLL_SECONDS`).
