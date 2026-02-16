# Windows Testing Guide

This guide is for running WebbDuck directly on Windows (not WSL).

## 1. Create Environment

```powershell
cd C:\Users\<you>\scrape\img-gen\webbduck
conda create -n webbduck python=3.10 -y
conda activate webbduck
```

## 2. Install Dependencies

Install PyTorch first (pick the correct index for your setup), then install WebbDuck deps:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.windows.txt
```

If you do not have a CUDA-capable NVIDIA setup, use CPU wheels from PyTorch instead.
If `torchaudio` was installed previously and reports a torch version conflict, remove it:

```powershell
pip uninstall -y torchaudio
```

### PyTorch/GPU Compatibility Guidance

- The torch wheel must match your GPU generation and available CUDA wheel support.
- Newer GPUs can require nightly torch builds before stable channels include compatible kernels.
- If generation fails with kernel errors (for example `no kernel image is available for execution on the device`), reinstall torch from a different channel.

Common channels:

```powershell
# Stable
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Nightly fallback for newer GPUs
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
# or
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu126 torch torchvision torchaudio
```

Verify your install:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Runtime overrides (optional):

```powershell
# Force safer precision on mixed GPU setups
$env:WEBBDUCK_DTYPE=\"float16\"

# Force CPU fallback (very slow, but useful for validation)
$env:WEBBDUCK_DEVICE=\"cpu\"

# Fail immediately instead of auto-falling back when WEBBDUCK_DEVICE=cuda
$env:WEBBDUCK_STRICT_DEVICE=\"1\"
```

## 3. Run App

```powershell
python run.py
```

Open `http://localhost:8010`.

## 4. Smoke Tests

```powershell
python -m pytest -q -s tests\test_server.py -m "not slow"
python -m pytest -q -s tests\test_modes.py tests\test_prompt_conditioning.py
```

## 5. Notes

- `requirements.windows.txt` intentionally excludes Linux/CUDA lockfile-style packages and nonessential build-problem packages used in `requirements.txt`.
- The app currently targets NVIDIA/CUDA for practical generation performance.
