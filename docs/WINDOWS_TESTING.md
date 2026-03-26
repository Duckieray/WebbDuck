# Windows Testing Guide

Use this guide when you want to run WebbDuck directly on Windows instead of WSL.

## 1. Create a Conda Environment

```powershell
cd <path-to-webbduck-repo>
conda create -n webbduck python=3.10 -y
conda activate webbduck
```

## 2. Install Dependencies

Install PyTorch first, then install the repo requirements.

```powershell
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.windows.txt
```

Create the expected asset/output folders if they do not exist yet:

```powershell
mkdir checkpoint\sdxl, lora, embeddings, outputs, weights
```

If `torchaudio` was installed earlier and conflicts with the selected torch build, remove it:

```powershell
pip uninstall -y torchaudio
```

## 3. Check PyTorch / GPU Compatibility

- The torch wheel must match your GPU generation, driver, and supported CUDA wheel channel.
- Very new GPUs may need nightly wheels before stable channels include the correct kernels.
- If you see `no kernel image is available for execution on the device`, reinstall torch from a different channel.

Examples:

```powershell
# Stable
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Nightly fallback for newer GPUs
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```

Verify the install:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

## 4. Run WebbDuck

```powershell
python .\run.py --output .\outputs\ --port 8010
```

Open `http://localhost:8010`.

Useful overrides:

```powershell
$env:WEBBDUCK_DEVICE="cuda"
$env:WEBBDUCK_DTYPE="float16"
$env:WEBBDUCK_STRICT_DEVICE="1"
python .\run.py --port 8010
```

## 5. Smoke Tests

Run the narrowest useful suite for your change.

```powershell
python -m pytest -q -s tests\test_server.py -m "not slow"
python -m pytest -q -s tests\test_prompt_conditioning.py
python -m pytest -q -s tests\test_ui_sanity.py
```

For a broader non-slow pass:

```powershell
python -m pytest -q -s -m "not slow"
```

## 6. Common Problems

- `torch.cuda.is_available()` is false: wrong wheel channel, unsupported driver, or mismatched CUDA build.
- `ModuleNotFoundError: webbduck`: run from the repo root and use `python run.py`.
- Hugging Face symlink warning: enable Developer Mode or run elevated; the warning is often non-fatal.
- WebbDuck falls back to CPU unexpectedly: set `WEBBDUCK_STRICT_DEVICE=1` to fail fast while debugging.

## 7. Related Docs

- `README.md` for top-level setup and quickstart
- `docs/DEVELOPMENT.md` for contributor workflows
- `tests/README.md` for the full test suite map
