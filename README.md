# WebbDuck

WebbDuck is a local-first SDXL image generation studio built for fast iteration, practical editing workflows, and filesystem-backed outputs you can keep on your own machine.

## What It Includes

- FastAPI backend with queue-based generation and WebSocket status updates.
- Zero-build web UI in `ui/` with Studio, Queue, Gallery, Lightbox, and plugin tabs.
- Text-to-image, img2img, inpaint, outpaint, two-pass refinement, and upscaling.
- LoRA and embedding discovery with live catalog refresh.
- Manifest-backed gallery search and thumbnail serving.
- Optional captioner and web-app plugin support.

## Quick Start

### Linux / WSL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p checkpoint/sdxl lora embeddings outputs weights
python run.py
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.windows.txt
mkdir checkpoint\sdxl, lora, embeddings, outputs, weights
python .\run.py
```

Open `http://localhost:8010`.

If you prefer the repo's conda-based workflow, use `startup.sh` on Linux/WSL or the commands in `docs/WINDOWS_TESTING.md` on Windows.

## Common Run Commands

```bash
python run.py --port 8020
python run.py --output ./outputs --models /path/to/models
python run.py --hf-cache /path/to/huggingface-cache
./startup.sh --env webbduck --output ./outputs --port 8010
```

## Key Runtime Settings

- `WEBBDUCK_OUTPUT_DIR`: output root for images, metadata, and manifest.
- `WEBBDUCK_MODELS_DIR`: shared models root; WebbDuck resolves checkpoints, LoRAs, embeddings, and weights under it.
- `WEBBDUCK_HF_CACHE_DIR`: Hugging Face hub cache override.
- `WEBBDUCK_DEVICE`: `cuda` or `cpu`.
- `WEBBDUCK_DTYPE`: `float16`, `bfloat16`, or `float32`.
- `WEBBDUCK_GPU_LEASE_WAIT_SECONDS`: bounded wait for shared GPU lease.
- `WEBBDUCK_CATALOG_POLL_SECONDS`: asset refresh interval.

See `docs/DEVELOPMENT.md` and `AGENTS.md` for the full list and when to use each variable.

## Repo Map

| Path | What lives there |
| --- | --- |
| `server/` | FastAPI routes, queue entrypoints, WebSocket, gallery APIs, thumbnails |
| `core/` | worker loop, generation orchestration, pipeline lifecycle, runtime helpers, plugins |
| `models/` | checkpoint, LoRA, embedding, and upscaler discovery |
| `modes/` | text2img, img2img, inpaint, outpaint, two-pass mode logic |
| `prompt/` | prompt conditioning and long-prompt chunking |
| `ui/` | HTML, ES modules, CSS, and client-side state |
| `plugins/` | bundled optional plugin examples |
| `tests/` | pytest coverage and regression checks |
| `docs/` | user, contributor, plugin, and platform docs |

For a more detailed file-by-file map, read `docs/ARCHITECTURE.md`.

## Documentation Guide

- `docs/ARCHITECTURE.md`: where everything is and which file owns what.
- `docs/DEVELOPMENT.md`: how to add features, run tests, and keep frontend/backend changes aligned.
- `docs/USER_GUIDE.md`: end-user features and workflows.
- `docs/SIMPLE_GUIDE.md`: short in-app help guide served by `/docs/simple-guide`.
- `docs/PLUGINS.md`: captioner and web-app plugin setup and contracts.
- `docs/WINDOWS_TESTING.md`: Windows-native setup, PyTorch guidance, and smoke tests.
- `ui/README.md`: frontend structure and editing guidance.
- `tests/README.md`: test suite map and how to choose focused verification.
- `plugins/README.md`: bundled plugin examples and local layout.

## Development Notes

- GPU-heavy work should go through `core/worker.py`, not request handlers.
- Keep backend request fields aligned with `ui/core/api.js`, `ui/app.js`, and `ui/core/state.js`.
- Optional plugins must stay optional; WebbDuck core should still work when they are missing.
- When behavior changes, update the relevant doc in `docs/`, `ui/README.md`, or `tests/README.md`.

## Tests

```bash
pytest -v -m "not slow"
pytest tests/test_server.py -v
pytest tests/test_prompt_conditioning.py -v
```

See `tests/README.md` for the rest of the suite.

## License

Apache-2.0
