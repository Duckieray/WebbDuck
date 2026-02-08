# WebbDuck

WebbDuck is a local SDXL studio focused on fast iteration, clean metadata, and practical workflows.

Everything runs on your machine: generation, queueing, gallery, inpaint, and optional captioning.

## Key Features

- Modern zero-build web UI (`ui/`) with Studio + Gallery views.
- Queue-based backend execution with real-time WebSocket updates.
- Text-to-image, img2img, inpaint, second-pass refinement, and upscaling.
- LoRA stack with per-LoRA weights, persisted UI state, and trigger phrase injection during generation.
- Token counter with warning when prompt exceeds the 77-token CLIP window.
- Gallery sessions with lazy thumbnails, search, lightbox metadata, and action toolbar.
- Live catalog refresh when checkpoints or LoRAs are added/removed from watched folders.
- Optional captioner plugin system (JoyCaption supported).

## Requirements

- OS: Windows 10/11 or Linux.
- Python: 3.10+.
- GPU: NVIDIA recommended for practical SDXL speed and memory headroom.
- Disk: enough space for checkpoints, LoRAs, and outputs.

## Installation

```bash
git clone https://github.com/Duckieray/webbduck.git
cd webbduck
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir checkpoint\sdxl, lora, outputs, weights
```

Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p checkpoint/sdxl lora outputs weights
```

## Run

```bash
python run.py
```

Open `http://localhost:8000`.

## Runtime Notes

- `run.py` starts FastAPI with reload and safe environment defaults.
- Queue and progress updates are pushed over `/ws`.
- Catalog changes are scanned on an interval and pushed to UI:
  - `WEBBDUCK_CATALOG_POLL_SECONDS` (default `3.0`)
- Thumbnail serving concurrency can be tuned:
  - `WEBBDUCK_THUMB_CONCURRENCY` (default `2`)

## Documentation

- `docs/SIMPLE_GUIDE.md`
- `docs/USER_GUIDE.md`
- `docs/DEVELOPMENT.md`
- `docs/architecture.md`
- `docs/REQUIREMENTS.md`
- `docs/PLUGINS.md`
- `ui/README.md`

## License

Apache-2.0
