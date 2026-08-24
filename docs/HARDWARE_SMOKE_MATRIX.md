# WebbDuck Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. The architecture migration is complete;
this document defines the first real hardware validation pass.

## 1. Runtime environments

Create the isolated image environments from any Python environment capable of
creating venvs:

```bash
python tools/prepare_model_runtimes.py all
```

Nix is not required. On NixOS, `nix develop` is one convenient way to obtain the
base Python; on Bazzite and conventional Linux hosts, run the script directly
with the Python used for local tooling.

The setup tool defaults to PyTorch 2.12.1 from the CUDA 13.0 wheel channel and
prints the environment variables that must be present when WebbDuck starts:

```bash
export WEBBDUCK_SDXL_PYTHON=...
export WEBBDUCK_FLUX_PYTHON=...
export WEBBDUCK_KREA2_PYTHON=...
export WEBBDUCK_QWEN_IMAGE_PYTHON=...
```

SDXL is isolated like the other image backends. Its job-scoped worker preserves
the mature SDXL T2I/I2I/inpaint/outpaint, LoRA, textual-inversion, second-pass
and identity-adapter paths while keeping Diffusers/model state out of the
WebbDuck service process. SDXL currently pins the mature Diffusers 0.36.0 stack;
FLUX, Krea and Qwen Image use their own independent requirement sets.
Runtime preparation never downloads model weights.

For a systemd deployment, add the four `WEBBDUCK_*_PYTHON` values to the service
environment rather than exporting them only in an interactive shell.

Before loading any model, start WebbDuck and check:

```text
GET /runtime-readiness
```

Every selected smoke target should report `runtime.ready=true` and CUDA should
identify the 5070 Ti from the interpreter that will actually run the backend.

Generation now enforces the same backend readiness contract immediately before
worker launch. A stale isolated environment therefore fails before a model
subprocess starts, and the error includes the missing runtime imports plus the
appropriate `python tools/prepare_model_runtimes.py <runtime>` repair command.
After repairing an isolated runtime, restart WebbDuck before retrying generation.

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

For a community/local single-file Krea checkpoint, that `.safetensors` file
replaces the transformer only. WebbDuck still needs the matching licensed Krea
Raw/Turbo Diffusers support components (scheduler, tokenizer/text encoder, VAE,
and transformer configuration). Before launching the Krea worker, WebbDuck now
preflights access to the support repository using the configured Hugging Face
credentials and reports license/token failures directly. An advanced fully-local
component bundle can be supplied with `WEBBDUCK_KREA2_COMPONENT_MODEL`; it must
be a complete Diffusers directory containing at least `model_index.json` and
`transformer/config.json`.

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
backend, not another SDXL download. The checkpoint and assets remain in their
existing shared model library; only the Python runtime is isolated.

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

## 4. API-driven runner

`tools/run_hardware_smoke.py` drives the live WebbDuck API rather than importing
backend implementation modules. That means a pass covers the public model
catalog, capability validation, request translation, queue/job path, storage and
backend resolver as one integrated path.

Preflight is safe and does not submit generation jobs:

```bash
python tools/run_hardware_smoke.py \
  --sdxl-model "<exact public SDXL model name>" \
  --dark-beast-model "<exact Dark Beast 3 FP8 public name>"
```

Rows whose runtime/weights are ready print `READY`; missing required targets are
`BLOCKED`. The optional SDXL asset row is skipped unless `--lora` or `--refiner`
is supplied.

When the preflight is green, explicitly opt into real generation:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --sdxl-model "<exact public SDXL model name>" \
  --dark-beast-model "<exact Dark Beast 3 FP8 public name>" \
  --lora "<known-good LoRA name>" \
  --refiner "<known-good refiner model name>"
```

Useful controls:

```text
--only flux-t2i             run one row
--only ltx-like-name        invalid here; row names are image rows only
--timeout 7200              per-generation timeout
--report-dir smoke_reports  JSON report destination
```

The runner creates deterministic synthetic source/mask fixtures for I2I/edit and
inpaint tests, uses seed 0, unloads all image models after every executed row,
and writes the report incrementally so a crash/OOM still leaves useful evidence.
It never downloads model weights.

## 5. What to record

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

The runner automatically captures the public readiness diagnostics, elapsed
request time and returned artifact references. Peak host RAM, exact backend
load time and externally observed CUDA memory may still require host monitoring
until those metrics are emitted by every worker.

A backend is not considered validated because imports succeed. `runtime.ready`
only clears the environment gate; each row above requires a real generated
artifact.

## 6. Failure classification

Classify failures before changing architecture:

1. **runtime** — missing import/version/CUDA;
2. **weights** — missing/gated/incomplete checkpoint;
3. **memory** — OOM/host-RAM/offload issue;
4. **pipeline API** — upstream signature/behavior drift;
5. **adapter** — WebbDuck request/result translation bug;
6. **model behavior** — generation completes but output is invalid/poor.

Do not add architecture selectors to work around a failure. Fix the responsible
backend/runtime or adjust the selected model's public capabilities/constraints.
