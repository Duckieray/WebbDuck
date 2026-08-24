# Krea 2 performance on constrained VRAM

Krea 2 is substantially larger than SDXL and needs a model-specific memory lifecycle on 16 GB GPUs. The public WebbDuck contract remains checkpoint-driven: users select the checkpoint; the Krea backend chooses the execution strategy.

## Workload matters: Raw vs Turbo

Krea 2 Raw/Base and Krea 2 Turbo/TDM are different checkpoints with different intended inference settings.

- Raw/Base defaults: 1024x1024, 28 denoising steps, guidance 4.5.
- Turbo/TDM defaults: 1024x1024, 8 denoising steps, guidance 0.0.

Guidance greater than zero requires both conditional and unconditional transformer evaluations in the Krea pipeline. A default Raw image therefore performs 56 transformer forward passes (28 x 2), while a default Turbo image performs 8. Do not apply Turbo defaults to a Raw checkpoint merely to make it faster.

For local single-file checkpoints, WebbDuck infers Turbo/TDM from checkpoint filename and safetensors metadata tokens such as `turbo`, `distill`, and `tdm`. Unknown Krea single files remain conservative Base/Raw.

## 16 GB execution lifecycle

The fast path is phase-oriented rather than whole-pipeline leaf offload:

1. Keep the large Krea transformer on CPU while loading the Qwen3-VL text encoder onto the GPU.
2. Encode positive conditioning once, plus negative conditioning only when guidance is enabled.
3. Move the text encoder back to CPU and release its CUDA allocations.
4. For scaled-FP8 single-file checkpoints, retain the large linear weights in FP8 storage and move the transformer resident to the GPU. Each FP8 linear is dequantized locally only for its current matmul instead of expanding the complete transformer to BF16.
5. For dense BF16/FP16 transformers that cannot fit resident, apply streamed block-level group offload to the transformer only. Leaf-level transformer offload is the compatibility fallback.
6. Decode with the VAE and return the generated image.

This avoids applying leaf-level CPU/GPU shuttling to the text encoder, transformer, and VAE simultaneously.

## Execution modes

`WEBBDUCK_KREA2_OFFLOAD=auto` selects:

- `resident-fp8` on a 15+ GB GPU when the single-file overlay preserved scaled FP8 linear weights;
- `transformer-block` for dense transformers below 32 GB;
- resident dense execution on larger GPUs.

Useful explicit overrides remain available for diagnosis:

- `WEBBDUCK_KREA2_OFFLOAD=block`
- `WEBBDUCK_KREA2_OFFLOAD=group`
- `WEBBDUCK_KREA2_OFFLOAD=sequential`
- `WEBBDUCK_KREA2_OFFLOAD=model`

`WEBBDUCK_KREA2_BLOCKS_PER_GROUP` controls block-level group size and defaults to 2. `WEBBDUCK_KREA2_GROUP_LOW_CPU_MEM` can trade additional execution overhead for reduced pinned host memory.

## Measurements

Each Krea generation records backend timings in generation metadata:

- `krea_pipeline_load_seconds`
- `krea_prompt_encode_seconds`
- `krea_execution_setup_seconds`
- `krea_denoise_decode_seconds`
- `krea_image_write_seconds`
- `krea_inference_seconds`
- `krea_worker_total_seconds`

`krea_runtime` also records the chosen execution/offload mode, execution quantization, count of preserved FP8 linears, device/VRAM information, and calculated `transformer_forward_passes`.

Use real hardware measurements before introducing persistent-worker caching or changing thresholds. Runtime readiness/import success alone is not a performance validation.