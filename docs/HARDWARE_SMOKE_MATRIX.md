# WebbDuck Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. The architecture migration is complete;
this document defines the first real hardware validation pass.

## 1. Runtime environments

Create the isolated image environments from a Nix development shell (or another
Python environment capable of creating venvs):

```bash
nix develop
python tools/prepare_model_runtimes.py all
```

The setup tool defaults to PyTorch 2.12.1 from the CUDA 13.0 wheel channel and
prints the environment variables that must be present when WebbDuck starts:

```bash
export WEBBDUCK_FLUX_PYTHON=...
export WEBBDUCK_KREA2_PYTHON=...
export WEBBDUCK_QWEN_IMAGE_PYTHON=...
```

SDXL remains in WebbDuck's host environment. FLUX, Krea and Qwen Image use
Diffusers 0.39.0 in isolated environments. Runtime preparation never downloads
model weights.

Before loading any model, start WebbDuck and check:

```text
GET /runtime-readiness
```

Every selected smoke target should report `runtime.ready=true` and CUDA should
identify the 5070 Ti from the interpreter that will actually run the backend.

## 2. Weight preparation

Use the normal Hugging Face cache; do not copy checkpoints into WebbDuck-specific
folders solely for discovery.

### FLUX.2 Klein 4B

Reference checkpoint:

```bash
hf download black-forest-labs/FLUX.2-klein-4B \
  --exclude "flux-2-klein-4b.safetensors" "*.jpg"
```

The excluded root single-file checkpoint duplicates the Diffusers transformer
for this test; WebbDuck's reference path uses the component layout.

### Krea 2 Turbo

Accept the Krea 2 license first, then cache the Diffusers component layout:

```bash
hf download krea/Krea-2-Turbo \
  --exclude "turbo.safetensors" "images/*"
```

Reference defaults are 1024x1024, 8 steps and guidance 0.0.

### Dark Beast 3 FP8

Keep the FP8 `.safetensors` as a local model file under any architecture-neutral
checkpoint root scanned by WebbDuck. The Krea backend should discover it from
its tensor structure and use Krea Raw shared components with the Dark Beast
transformer overlaid. INT8 ConvRot is intentionally a separate unsupported
format for this smoke pass.

### Qwen-Image-2512

```bash
hf download Qwen/Qwen-Image-2512
```

This is a large package (roughly 58 GB) and the runtime intentionally uses
sequential CPU offload on a 16 GB GPU.

### SDXL

Use one existing known-good local SDXL checkpoint plus the currently configured
LoRA/embedding/refiner assets. The purpose is regression coverage for the mature
backend, not another SDXL download.

## 3. Smoke order

Run tests in increasing memory/risk order. Do not start the next model until the
previous process has released the GPU lease and CUDA memory.

| Order | Target | Workflow | Minimal first pass | Pass condition |
|---|---|---|---|---|
| 1 | SDXL | T2I | 1024x1024, 20 steps, seed 0 | image saved + metadata + repeatable seed |
| 2 | SDXL | I2I | one source image | source conditioning works |
| 3 | SDXL | Inpaint | small mask | mask workflow works |
| 4 | SDXL | LoRA/refiner | one known asset each | assets + second pass work |
| 5 | FLUX.2 Klein 4B | T2I | 1024x1024, 4 steps, seed 0 | image generated in isolated runtime |
| 6 | FLUX.2 Klein 4B | edit | one source image | image-conditioned edit works |
| 7 | Krea-2-Turbo | T2I | 1024x1024, 8 steps, CFG 0, seed 0 | official checkpoint works |
| 8 | Dark Beast 3 FP8 | T2I | 1024x1024, model defaults, seed 0 | single-file FP8 overlay is complete and generates |
| 9 | Qwen-Image-2512 | T2I | model default resolution, seed 0 | sequential-offload run completes |

## 4. What to record

For every row record:

- selected public model name;
- runtime Python and library versions from readiness;
- GPU name, VRAM and offload mode;
- peak host RAM if available;
- load time;
- generation time;
- requested vs actual dimensions/steps/seed;
- output path and metadata path;
- any warning/fallback;
- whether CUDA memory is released after model switch/unload.

A backend is not considered validated because imports succeed. `runtime.ready`
only clears the environment gate; each row above requires a real generated
artifact.

## 5. Failure classification

Classify failures before changing architecture:

1. **runtime** — missing import/version/CUDA;
2. **weights** — missing/gated/incomplete checkpoint;
3. **memory** — OOM/host-RAM/offload issue;
4. **pipeline API** — upstream signature/behavior drift;
5. **adapter** — WebbDuck request/result translation bug;
6. **model behavior** — generation completes but output is invalid/poor.

Do not add architecture selectors to work around a failure. Fix the responsible
backend/runtime or adjust the selected model's public capabilities/constraints.
