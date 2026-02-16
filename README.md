# WebbDuck v1.0.0

WebbDuck is a local SDXL studio focused on fast iteration, clean metadata, and practical workflows.

Everything runs on your machine: generation, queueing, gallery, inpaint, and optional captioning.

## Key Features

- Modern zero-build web UI (`ui/`) with Studio + Gallery views.
- Mobile Studio pane toggle (`Preview` / `Settings`) for practical small-screen workflow.
- Queue-based backend execution with real-time WebSocket updates.
- Queue modal supports canceling queued jobs and requesting cancellation for active running jobs.
- Text-to-image, img2img, inpaint, second-pass refinement, and upscaling.
- LoRA stack with per-LoRA weights, persisted UI state, and trigger phrase injection during generation.
- Token counter with warning when prompt exceeds the 77-token CLIP window.
- Long-prompt chunking for SDXL conditioning (prompts beyond 77 tokens are chunk-encoded instead of hard-truncated).
- Flat gallery grid with thumbnail-size slider, infinite scroll, global keyword search, and lightbox action toolbar.
- Live catalog refresh when checkpoints or LoRAs are added/removed from watched folders.
- Optional captioner plugin system (JoyCaption supported).
- Generic web-app plugin host (`plugins/webapps`) so plugins can render their own UI page in WebbDuck.

## Requirements

- OS: Windows 10/11 or Linux.
- Python: 3.10+.
- GPU: NVIDIA recommended for practical SDXL speed and memory headroom.
- Disk: enough space for checkpoints, LoRAs, and outputs.

## Installation

```bash
git clone https://github.com/Duckieray/WebbDuck.git
cd webbduck
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.windows.txt
mkdir checkpoint\sdxl, lora, outputs, weights
```

PyTorch GPU compatibility note (important):
- The correct torch build depends on your GPU generation, driver, Python version, and available PyTorch wheels.
- Some newer GPUs may require nightly torch builds before stable wheels support their SM architecture.
- If `torch.cuda.is_available()` is `False`, or you see CUDA kernel errors (for example `no kernel image is available`), switch to a different PyTorch wheel channel.
- Example alternatives:
  - Stable: `https://download.pytorch.org/whl/cu124`
  - Nightly: `https://download.pytorch.org/whl/nightly/cu128` (or `cu126` if needed)
- Verify after install:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
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

Open `http://localhost:8010`.

Custom port example:

```bash
python run.py --port 8020
```

## Runtime Notes

- `run.py` starts FastAPI with reload and safe environment defaults.
- Queue and progress updates are pushed over `/ws`.
- Gallery search uses a manifest index (`outputs/manifest.jsonl`) for global matching.
- Catalog changes are scanned on an interval and pushed to UI:
  - `WEBBDUCK_CATALOG_POLL_SECONDS` (default `3.0`)
- Thumbnail serving concurrency can be tuned:
  - `WEBBDUCK_THUMB_CONCURRENCY` (default `2`)
- Runtime device/dtype can be overridden:
  - `WEBBDUCK_DEVICE` (`cuda` or `cpu`)
  - `WEBBDUCK_DTYPE` (`float16`, `bfloat16`, `float32`)
  - `WEBBDUCK_STRICT_DEVICE=1` to fail fast instead of auto-fallback when CUDA is requested but unusable

## Documentation

- `docs/SIMPLE_GUIDE.md`
- `docs/USER_GUIDE.md`
- `docs/DEVELOPMENT.md`
- `docs/architecture.md`
- `docs/REQUIREMENTS.md`
- `docs/WINDOWS_TESTING.md`
- `docs/PLUGINS.md`
- `ui/README.md`

## License

Apache-2.0
