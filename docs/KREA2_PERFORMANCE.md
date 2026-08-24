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

Otherwise CUDA/ROCm uses transformer-only block group offload. CUDA uses streamed single-block prefetch; ROCm currently uses the conservative synchronous path. Dense models can become resident automatically on larger cards when the same headroom rule passes.

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
- any adaptive fallback reason;
- text-encoder execution mode;
- execution quantization and preserved FP8 linear count;
- hardware profile with total/free VRAM;
- stored transformer size and activation reserve;
- calculated `transformer_forward_passes`.

Use real hardware measurements before introducing persistent-worker caching or changing thresholds. Runtime readiness/import success alone is not a performance validation.