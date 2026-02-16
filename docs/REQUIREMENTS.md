# WebbDuck Current Requirements

This document tracks the current functional and technical requirements implemented in the project.

## 1. Core Stack

- Frontend: Vanilla JavaScript (ES modules), no framework build step.
- Styling: Vanilla CSS with design tokens and Nova theme layer.
- Backend: FastAPI.
- Runtime messaging: REST for actions + WebSocket push for live state.

Platform install notes:
- Linux/WSL: `requirements.txt`
- Windows: `requirements.windows.txt` (install PyTorch separately first using the official PyTorch wheel index for your CUDA/CPU target; newer GPUs may require nightly PyTorch wheels until stable kernels are available)
- Runtime defaults auto-select GPU precision and may fall back to CPU when the installed CUDA wheel cannot execute kernels on the detected GPU architecture.

## 2. Functional Requirements

### 2.1 Generation

- Text-to-image generation.
- Img2img using uploaded source image.
- Inpainting with mask editor and replace/keep modes.
- Optional second-pass refinement model.
- Upscale operation for generated images.

### 2.2 Queue and Orchestration

- All GPU actions enqueue through a single backend queue.
- Queue payload includes status, queue position, and compact request metadata.
- Queued jobs can be canceled before they start.
- Running jobs support cancellation request (`running -> cancelling -> cancelled` when successful).
- Queue updates are pushed over WebSocket to avoid constant frontend polling.

### 2.3 Model and LoRA Management

- Base model list is dynamically loaded from registry.
- LoRAs are filtered by model architecture.
- Local `lora/*.safetensors` files are synced into `lora/loras.json`.
- `loras.json` default `weight` is used by UI slider initialization.
- LoRA trigger phrases are injected into prompt text during generation.

### 2.4 Gallery and Viewer

- Gallery is paginated with `start` + `limit`.
- Images are rendered in a flat newest-first grid in the UI (backed by paginated API pages).
- Search supports global keyword matching across gallery metadata.
- Gallery supports filter chips (`All`, `HD`, `Favorites`), adjustable thumbnail size, and infinite scroll loading.
- Lightbox metadata panel includes prompt, negative, model, seed, settings, LoRAs.
- Lightbox metadata includes generation timing fields.
- Lightbox includes regenerate, upscale, inpaint, download, compare, delete actions.

### 2.5 State Persistence

- Core Studio fields persist to `localStorage` and restore on reload.
- Selected LoRAs and weights restore when compatible with the current model.
- Resolution preset chip state syncs with width/height; `Custom` indicates non-preset values.

### 2.6 Mobile UX

- Studio supports explicit mobile pane switching between `Settings` and `Preview`.
- Settings pane keeps batch size and generate controls accessible without full-page scroll lock.

## 3. Runtime and Performance Requirements

- Thumbnail generation is on-demand and concurrency-limited.
- Queue processing is serialized to avoid GPU contention.
- Catalog watcher auto-refreshes model/LoRA registries and notifies UI.
- No mandatory heavy frontend dependencies or client build chain.

## 4. API Surface (Current)

- `POST /generate`
- `POST /test`
- `POST /upscale`
- `GET /gallery`
- `GET /models`
- `GET /second_pass_models`
- `GET /schedulers`
- `GET /models/{base_model:path}/loras`
- `GET /gallery/search`
- `GET /queue`
- `POST /queue/cancel`
- `POST /tokenize`
- `GET /captioners`
- `GET /caption_styles`
- `POST /caption`
- `GET /health`
- `GET /thumbs/{path:path}`
- WebSocket `/ws`

## 5. Environment Knobs

- `WEBBDUCK_CATALOG_POLL_SECONDS`: catalog refresh interval (seconds).
- `WEBBDUCK_THUMB_CONCURRENCY`: concurrent thumbnail generation cap.
