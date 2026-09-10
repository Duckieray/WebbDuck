# Krea 2 performance across GPU classes

Krea 2 is substantially larger than SDXL and needs a hardware-aware memory lifecycle. The public WebbDuck contract remains checkpoint-driven: users select the checkpoint; the Krea backend profiles the actual accelerator and chooses the execution strategy automatically.

## Hardware is internal execution metadata

WebbDuck does not expose a backend/GPU-profile selector in the generation UI. Runtime preparation and Krea execution share the same hardware policy:

- NVIDIA CUDA: select a compatible CUDA PyTorch wheel and enable CUDA-stream prefetch where validated.
- AMD ROCm: select the ROCm PyTorch wheel and use conservative synchronous group offload until streamed ROCm offload has a hardware smoke pass.
- Apple MPS: use the platform PyTorch build and resident execution only when the model actually fits.
- CPU: correctness fallback only.

`tools/prepare_model_runtimes.py` defaults to automatic hardware detection. For PyTorch 2.12.1 it selects CUDA 13.0 for current NVIDIA GPUs, CUDA 12.6 for Pascal/Volta-class NVIDIA GPUs, ROCm 7.2 for AMD, the normal PyPI build for Apple/MPS, or the CPU wheel. Explicit `--accelerator` and `--torch-index` overrides remain available for diagnosis.

The preparer also supports optional profile dependency overlays by convention, such as:

- `runtime_requirements/krea2.cuda.txt`
- `runtime_requirements/krea2.nvidia.txt`
- `runtime_requirements/krea2.rocm.txt`
- `runtime_requirements/krea2.amd.txt`

Only overlays that exist are installed. This lets future native FP8/attention/driver-specific packages be added without hard-coding package names in the generic runtime preparer.

WebbDuck intentionally does **not** run `pip install` inside a live generation request. Package installation belongs to runtime preparation/startup repair; generation only chooses among capabilities already present in that isolated runtime.

## Workload matters: Raw vs Turbo

Krea 2 Raw/Base and Krea 2 Turbo/TDM are different checkpoints with different intended inference settings.

- Raw/Base defaults: 1024x1024, 28 denoising steps, guidance 4.5.
- Turbo/TDM defaults: 1024x1024, 8 denoising steps, guidance 0.0.

Guidance greater than zero requires both conditional and unconditional transformer evaluations in the Krea pipeline. A default Raw image therefore performs 56 transformer forward passes (28 x 2), while a default Turbo image performs 8. Do not apply Turbo defaults to a Raw checkpoint merely to make it faster.

For local single-file checkpoints, WebbDuck infers Turbo/TDM from checkpoint filename and safetensors metadata tokens such as `turbo`, `distill`, and `tdm`. Unknown Krea single files remain conservative Base/Raw.

## Phase-oriented execution

The optimized path treats Krea as three mutually exclusive GPU phases rather than applying one offload strategy to the whole pipeline:

1. **Text encoding** — keep the transformer/VAE off the accelerator; run the Qwen3-VL encoder resident when it fits, otherwise use accelerator-aware block offload or CPU fallback.
2. **Denoising** — detach the text encoder and VAE, then choose a transformer profile from live free VRAM and actual stored transformer size.
3. **Decode** — request latent output from Krea, release the transformer completely, then move only the VAE onto the accelerator for image decode.

Prompt embeddings are cached on CPU between phases and reused for multiple images in one request.

This avoids making the Qwen encoder, Krea transformer, and VAE compete for VRAM simultaneously.

## Scaled FP8 fast path

For compatible community single-file checkpoints, large linear weights are kept as FP8 + scale storage instead of expanding the complete transformer to BF16.

Each `ScaledFP8Linear` dequantizes only its current layer weight to the activation dtype for the matmul. This is a storage optimization, not a claim of native FP8 tensor-core execution. It removes most of the persistent BF16 model-memory cost while keeping the existing Diffusers Krea graph.

FP8 preservation is enabled only when the live PyTorch accelerator reports that FP8 storage can actually be allocated. Other devices dequantize to the normal runtime dtype and use the dense fallback profile.

## Live headroom selection

Capacity alone is not enough. A 16 GB desktop GPU driving browsers/compositors is different from a dedicated 16 GB headless GPU.

Before denoising WebbDuck measures:

- accelerator/vendor and device name;
- total VRAM;
- **currently free VRAM**;
- BF16 support;
- FP8 storage support;
- stream-prefetch support;
- stored transformer size;
- resolution-scaled activation reserve.

`auto` selects resident execution only when:

`free VRAM >= stored transformer size + activation reserve`

Otherwise CUDA/ROCm uses transformer-only block group offload with **synchronous** transfers. Streamed block prefetch (`use_stream=True`) was abandoned for both FLUX.2 and Krea: after a manual per-block forward the streamed hooks never return the block's weights, so they accumulate ~one block per turn and the job OOMs mid-denoise regardless of resolution. Synchronous block offload returns every block and a 768x768 image completes reliably on 16 GB. Dense models can become resident automatically on larger cards when the same headroom rule passes.

This means two machines with the same nominal GPU capacity may intentionally choose different Krea profiles.

## Adaptive OOM downgrade

VRAM can change after the profile is selected. If resident denoising still hits a CUDA OOM, WebbDuck automatically:

1. reports that available VRAM changed;
2. moves the transformer back out of GPU memory;
3. clears the CUDA cache;
4. switches to transformer block offload;
5. retries the image with the same seed.

The final runtime metadata records both `initial_offload` and the effective `offload`, plus `fallback_reason=resident_oom` when this occurs.

## Execution overrides

These remain diagnostic escape hatches rather than required user configuration:

- `WEBBDUCK_KREA2_OFFLOAD=block`
- `WEBBDUCK_KREA2_OFFLOAD=group`
- `WEBBDUCK_KREA2_OFFLOAD=sequential`
- `WEBBDUCK_KREA2_OFFLOAD=model`
- `WEBBDUCK_KREA2_BLOCKS_PER_GROUP=<n>`
- `WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM=0|1`

The normal product path should remain `WEBBDUCK_KREA2_OFFLOAD=auto`.

## Identity edit cost and tuning

Identity edits are the expensive Krea case: every guidance-enabled denoise step
runs **two** full transformer forwards (CFG positive + negative), and each
forward streams all blocks CPU<->GPU. A default Raw identity image at 28 steps
is therefore ~56 forwarded streams on top of the grounded Qwen3-VL encode. Low
VRAM (~6 GB during identity editing) is by design — only a couple of blocks are
resident — and does not predict speed.

Identity-specific (no effect on text2img):

- **Paired block streaming** (`WEBBDUCK_KREA2_IDENTITY_PAIRED=0` to disable;
  default on for CUDA): `edit_transformer_forward_paired` loads each
  transformer block once per step and runs the CFG-positive and CFG-negative
  rows through it before offloading — identical math to two separate forwards,
  half the block-transfer/Python churn. `_configure_execution` reports the
  `paired-block` mode.
- **Resident block prefix** (`WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS=0` to
  disable; `auto` default): when paired mode is active the worker stages the
  first N transformer blocks on the GPU once per job (weights are constant
  across steps). N is budgeted from measured free VRAM minus this resolution's
  activation reserve minus a 1.5 GB workspace margin (~0.42 GB fp8/block;
  `paired-block-N` when partial, `paired-resident` when all blocks are resident).
  An OOM while the prefix is active drops the prefix and retries pure paired
  streaming before falling back to the hook-based block path.
- `WEBBDUCK_KREA2_IDENTITY_STEPS=<int>` — override identity denoise steps.
- `WEBBDUCK_KREA2_IDENTITY_GUIDANCE=<float>` — override the request cfg for
  identity only; `0` (or negative) skips the unconditional forward altogether
  (single forward per step).
- `WEBBDUCK_KREA2_IDENTITY_CFG_FREE=1` — shorthand for guidance `0`.
- `WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET=<int>` / `identity.token_budget` — A/B
  the adaptive target-grid token budget (the real identity size limit). An
  identity forward packs `[text | reference grid | target grid]` into **one**
  sequence (~2x image tokens/output grid + prompt), so the planner halves the
  card's text2img budget (3584) to 1792 by default, and any request ≥ ~800x1200
  collapses to ~528x784 effective regardless of the megapixel cap (default 2.0,
  which is *not* the binding limit). Request-level `token_budget` beats the env
  var.

These knobs exist so the two dominant levers (drop CFG for identity; cut
identity steps) can be A/B'd on live hardware without code changes. Steps and
CFG affect output character, so validate before promoting any value to a
default.

Hardware-validated identity size tiers (16 GB 5070 Ti, same prompt/ref, 28
steps, cfg 7.5, no OOM/fallback):

| token budget | requested | effective output | combined seq (+text) | wall time |
|---|---|---|---|---|
| 1792 (default) | 832x1216 | 528x784 (0.41 MP) | ~3468 (~3800) | ~166 s |
| 2048 | 832x1216 | 576x864 (0.50 MP) | ~3888 (~4240) | ~194 s |
| 2688 | 832x1216 | 672x1008 (0.68 MP) | ~5292 (~5650) | ~243 s |
| 3584 | 1152x1728 | 768x1152 (0.88 MP) | ~6912 (~7320) | ~340 s |

The ~4096-token training envelope is crossed between 2048 and 2688; past that,
outputs should be eyeballed (all tiers above completed without NaN/gray-wash
signals, but sequence-length extrapolation can degrade attention/identity
quality before stats show it).

## Artifact upscale (effective res → requested size)

The effective output above is what the worker **saves** — a 832x1216 portrait
request at the default budget lands as a 528x784 PNG. Viewed at the requested
size that reads as compressed/low-detail. To fix that, the server-side
(webbduck env, not the isolated runtime) applies `_maybe_upscale_identity_artifact`
to every returned identity image once the worker is done:

- Target is the true requested size (`settings["requested_*"]`, preserved by the
  adaptive plan) — never the adapted dims.
- Real-ESRGAN at the smallest factor whose upscale clears the target short edge
  (x2 for typical portrait down-steps, x4 for deeper ones), then LANCZOS to the
  exact requested dimensions. Plain LANCZOS fallback if the weights/libs are
  missing (weights resolve from `WEBBDUCK_WEIGHTS_DIR` → `WEBBDUCK_MODELS_DIR/weights` →
  repo `weights/`).
- Recorded in meta as `krea_upscale` (`{from, to, upscaler, upscale_error?}`) and
  `performance_timing.krea_upscale_seconds` (~2.1 s for 528x784 → 832x1216 on the
  5070 Ti; runs with the worker's GPU lease still held, so no extra lease).
- Disable with `WEBBDUCK_KREA2_IDENTITY_UPSCALE=0` (set in the server env, not the
  isolated runtime).

This is post-hoc magnification, not recovery: the native generation is still the
~0.4-0.5 MP effective grid, and upscaling adds plausible texture rather than true
missing detail. For true native sharpness at 832x1216+ the token budget must be
raised (see tiers above) or the GPU's VRAM increased.

## Measurements

Each Krea generation records backend timings in generation metadata:

- `krea_pipeline_load_seconds`
- `krea_prompt_encode_seconds`
- `krea_execution_setup_seconds`
- `krea_denoise_seconds`
- `krea_decode_seconds`
- `krea_denoise_decode_seconds`
- `krea_image_write_seconds`
- `krea_inference_seconds`
- `krea_worker_total_seconds`

`krea_runtime` also records:

- selected and initial execution/offload mode;
- resident-block prefix count (`resident_blocks`);
- any adaptive fallback reason;
- text-encoder execution mode;
- execution quantization and preserved FP8 linear count;
- hardware profile with total/free VRAM;
- stored transformer size and activation reserve;
- calculated `transformer_forward_passes`.

Use real hardware measurements before introducing persistent-worker caching or changing thresholds. Runtime readiness/import success alone is not a performance validation.