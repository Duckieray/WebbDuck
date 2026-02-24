# WebbDuck Agent Guide

This file is the working playbook for AI/code agents contributing to WebbDuck.
Use it as the first-stop reference for architecture, workflows, project rules, and run/test commands.

## 1. Project Snapshot

- App type: local-first SDXL image generation studio.
- Backend: FastAPI + Uvicorn.
- Frontend: vanilla JavaScript (ES modules), no Node build pipeline.
- Execution model: queue-first GPU worker (single worker path for GPU-heavy work).
- Realtime: WebSocket events for status/queue/catalog updates.
- Storage: filesystem outputs + metadata JSON + manifest index for gallery search.

## 2. Tech Stack

- Python 3.10+
- FastAPI / Starlette / Uvicorn
- PyTorch + Diffusers + Transformers + PEFT + Safetensors
- Pillow / OpenCV / NumPy / SciPy
- Optional captioning/upscale dependencies (see `requirements*.txt`)
- Frontend:
  - `ui/app.js`
  - `ui/core/*.js` (API/events/state/utils)
  - `ui/modules/*.js` (Gallery/Lightbox/Lora/Embedding/etc.)
  - `ui/styles/*.css`

Key folders:

- `server/` API + websocket + gallery/thumb serving
- `core/` worker, generation orchestration, pipeline lifecycle
- `models/` registry/discovery for models, LoRAs, embeddings
- `modes/` txt2img, img2img, inpaint, outpaint, two-pass mode logic
- `prompt/` SDXL conditioning + long-prompt chunking
- `tests/` pytest suite
- `docs/` architecture and developer docs

## 3. Core Architecture Rules

1. Queue-first GPU work
- Do not run heavy generation directly in request handlers.
- GPU jobs should be enqueued and handled by `core/worker.py`.

2. Keep frontend/backend contracts aligned
- If request fields change in backend (`server/app.py`), update UI form wiring (`ui/app.js`) and persisted state (`ui/core/state.js`) as needed.

3. Catalog auto-refresh is expected behavior
- Model/LoRA/embedding changes should flow through registry refresh + websocket `catalog` updates.

4. Long-prompt behavior
- SDXL long prompts use chunked conditioning.
- Avoid reintroducing legacy experimental compression paths unless explicitly requested.

5. Modal UX over browser popups
- Prefer app modal components over native `alert/confirm/prompt`.

## 4. Coding Guidelines (Project-Specific)

- Make narrow, intentional edits; preserve existing patterns.
- Do not add new frameworks/build tooling unless explicitly requested.
- Prefer explicit naming and small helper functions over dense inline logic.
- Maintain backward-compatible defaults when changing UI controls.
- Keep server errors actionable; fail fast with clear messages where possible.
- Add/update tests for behavior changes when practical.
- Update docs when behavior or setup changes.

## 5. Git + Workspace Rules

- Never use destructive git commands (`reset --hard`, `checkout --`, etc.) unless explicitly requested.
- Do not revert user-owned or unrelated changes.
- Commit only relevant files.
- Exclude local artifacts/debug data from commits (examples: `outputs/`, `inpaint_input/`, local checkpoints, local LoRA/embedding/model binaries).
- Keep commit messages descriptive and scoped.

## 6. Environment Variables (Important)

- `WEBBDUCK_OUTPUT_DIR`
- `WEBBDUCK_PORT`
- `WEBBDUCK_LORA_DIR`
- `WEBBDUCK_EMBEDDING_DIR`
- `WEBBDUCK_CATALOG_POLL_SECONDS` (default: `3.0`)
- `WEBBDUCK_THUMB_CONCURRENCY` (default: `2`)
- `WEBBDUCK_DEVICE` (`cuda` or `cpu`)
- `WEBBDUCK_DTYPE` (`float16`, `bfloat16`, `float32`)
- `WEBBDUCK_STRICT_DEVICE=1` (fail fast instead of auto fallback)

## 7. WSL/Linux Commands

Recommended environment name: `webbduck` (legacy envs like `web_img` may still exist locally).

### Setup

```bash
cd <path-to-webbduck-repo>
conda create -n webbduck python=3.10 -y
conda activate webbduck
pip install -r requirements.txt
mkdir -p checkpoint/sdxl lora embeddings outputs weights
```

### Run (scripted)

```bash
./startup.sh
./startup.sh --port 8020
./startup.sh --env webbduck --output ./outputs --port 8010
```

### Run (direct)

```bash
conda activate webbduck
python run.py --output ./outputs --port 8010
```

### Tests

```bash
conda run -n webbduck pytest -v -m "not slow"
conda run -n webbduck pytest tests/test_server.py -v
conda run -n webbduck pytest tests/test_prompt_conditioning.py -v
```

## 8. Windows Commands (PowerShell)

### Setup

```powershell
cd <path-to-webbduck-repo>
conda create -n webbduck python=3.10 -y
conda activate webbduck
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.windows.txt
mkdir checkpoint\sdxl, lora, embeddings, outputs, weights
```

### Run

```powershell
conda activate webbduck
python .\run.py --output .\outputs\ --port 8010
```

### Tests

```powershell
python -m pytest -q -s tests\test_server.py -m "not slow"
python -m pytest -q -s tests\test_modes.py tests\test_prompt_conditioning.py
```

### Optional runtime overrides

```powershell
$env:WEBBDUCK_DEVICE="cuda"
$env:WEBBDUCK_DTYPE="float16"
$env:WEBBDUCK_STRICT_DEVICE="1"
python .\run.py --port 8010
```

## 9. GPU Compatibility Notes

- PyTorch wheel/channel must match GPU generation and driver support.
- If `torch.cuda.is_available()` is false or you get kernel-image errors, reinstall torch from a compatible channel.
- Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 10. Common Agent Tasks

1. Add backend option
- Update endpoint form fields in `server/app.py`.
- Thread value into settings payload.
- Apply setting in `core/generation.py` / relevant mode.
- Add UI control in `ui/index.html` + wiring in `ui/app.js` + state persistence.
- Add tests.

2. Add model/asset type discovery
- Extend `models/registry.py`.
- Ensure watcher refresh paths/signature include necessary files.
- Ensure API endpoints return compatible metadata for frontend managers.

3. Touch gallery behavior
- Keep infinite scroll + search + selection interactions compatible.
- Validate lightbox actions and metadata rendering.

## 11. Troubleshooting Quick Reference

- Port already in use:
  - Change port (`--port 8020`) or kill the existing process.
- `ModuleNotFoundError: webbduck`:
  - Run from repo root and use `python run.py` (path handling is built into `run.py`).
- Hugging Face symlink warning on Windows:
  - Enable Developer Mode or run elevated; warning is often non-fatal.
- `no kernel image is available for execution on the device`:
  - Torch/CUDA wheel mismatch for the GPU architecture.
- Slow generation after feature changes:
  - Check whether conditioning path changed, whether extra passes were enabled, and whether dtype/device fallback occurred.

## 12. Source-of-Truth Docs

- `README.md`
- `docs/DEVELOPMENT.md`
- `docs/WINDOWS_TESTING.md`
- `docs/USER_GUIDE.md`
- `docs/PLUGINS.md`
- `tests/README.md`
- `ui/README.md`

When behavior changes, keep these docs in sync.
