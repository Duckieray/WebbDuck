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

## Img2img (optional)

Krea 2 supports img2img. The staged source image is VAE-encoded and normalized
into Krea's packed latent space, then noised to the sigma implied by the denoise
strength (`x_t = (1 - s) * x0 + s * noise`) and only the tail of the sigma
schedule is executed via `pipe(latents=..., sigmas=...)`.

- **Strength.** `1.0` is plain text-to-image. Below `1.0` the request runs as
  img2img against the staged image. The executed step count is
  `floor(steps * strength)` (floored at 1), and progress reporting is scaled to
  that count so the UI reaches 100% rather than stalling at a fraction.
- **Dynamic shifting.** Krea's scheduler uses `use_dynamic_shifting=True`, so the
  sigma list must *not* be truncated and handed back naively - the scheduler would
  apply its exponential shift a second time. The implementation generates the full
  post-shift schedule, truncates it, then inverse-shifts the retained values before
  passing them as `sigmas`. Turbo uses a fixed `mu=1.15`; Raw derives `mu` from the
  packed image sequence length (resolution shift).
- **Batching.** Multi-image requests encode the source once and reuse the clean
  latent with a distinct noise seed per image. The OOM retry ladder replays the
  same initial latents rather than re-noising.
- **Planner.** The host reports the packed latent shape and truncated step count to
  the adaptive planner, so tokens and VRAM are budgeted from the real numbers.
- **Conflicts.** Krea identity edit is itself image-driven and consumes its own
  reference as the denoise anchor, so identity and img2img cannot run together.
  The combination is rejected with an explicit error rather than silently dropping
  the staged image. Krea inpaint and outpaint remain unsupported.

## Automatic realism routing

Most requests need no configuration at all. Before generation the host resolves a
realism profile (`core/realism_profile.py`) and, when it resolves to `realistic`,
selects the photographic decoder and enables the face/skin detailer automatically.
Two passes that were previously behind opt-in switches now happen on their own.

- **Signals.** Curated model metadata first (`modes` / `recommended_mode` in
  `webbduck_meta/`), then prompt wording: photographic terms (`portrait photograph`,
  `skin pores`, `85mm`, `natural lighting`, `bokeh`) accumulate a score, and stylized
  terms (`anime`, `cel shading`, `lineart`, `manga`) cancel it.
- **Conservative default.** An ambiguous prompt resolves to `unknown` and changes
  nothing. Silent is safer than wrong: the photographic decoder and face pass both
  damage line art.
- **Hard veto.** An explicit stylized request (`anime`, `manga`, `cel shading`,
  `lineart`, `pixel art`, `chibi`) wins even when the prompt also says
  `photorealistic`.
- **Neutral words.** `cinematic`, `portrait`, `masterpiece` and similar artistic
  terms are ignored rather than counted as evidence either way.
- **Override.** **Advanced Settings** in the Studio exposes `realism_mode`
  (`auto` | `always` | `never`). `always` forces the photographic path, `never`
  disables it. The override is sent on every request, and unchecking the Advanced
  **Refine faces** box sets `face_detailer_explicit`, which stops automatic
  detailer enablement without affecting the decoder choice.
- **Reporting.** The resolved mode, its reasons and the confidence are recorded as
  `realism` in the run metadata; the Parameters section shows a one-line status.

## Face/skin detailer (automatic)

A face that is small in the final image covers only a handful of 16 px latent
tokens, so no decoder - including the 2x realism decoder - can recover pore detail
that was never generated. This pass fixes the generation rather than the decode:

1. Detect faces on the decoded image with OpenCV YuNet (`FaceDetectorYN`), which
   still resolves faces around 32 px where Haar cascades fail.
2. Crop each face with padding so hair and jaw context are included.
3. Upscale the crop to roughly one megapixel, making it an ordinary normal-sized
   Krea job that fits a 16 GB card.
4. Re-denoise the crop with the same prompt on its **own** schedule
   (`refine_steps` / `refine_strength`, see below), reusing the already-loaded
   transformer.
5. Downscale to the crop's original pixel size, color-match against the original
   crop, and composite back **only inside a feathered facial ellipse**.

- **Automatic for realistic prompts.** Enabled by the realism router above, with no
  user action. Advanced Settings exposes the overrides for models advertising the
  `face_detailer` capability (Krea 2). The API field is `face_detailer` (a JSON
  object with `enabled`, `refine_strength`, `refine_steps`, `min_face_px`,
  `max_faces`, `max_face_fraction`, `pad_ratio`, `feather_px`, and `detector_path`).
- **The refine has its own schedule, on purpose.** `refine_strength` is *not* a
  percentage of noise. `img2img_schedule` derives its floor from the *time-shifted*
  schedule, so for Krea's `linspace(1, 1/steps, steps)` base sigmas pushed through
  the Turbo shift (`mu=1.15`, `shift~3.158`), `steps=10, strength=0.30` starts at
  **shifted sigma 0.575**. Reusing the main pass's step count therefore re-noised
  the crop past halfway and re-rendered it rather than refining it. The default
  `refine_steps=24` / `refine_strength=0.16` lands at sigma ~0.26 for 3 executed
  steps - the same compute, but a genuine refinement. `steps` and `strength` are
  still honoured when supplied explicitly, so saved presets keep working.
- **Latent normalization is an exact inverse pair.** `encode_source_latent` must
  emit `(latent - latents_mean) / config.latents_std`, because Krea's decode is
  `latents / (1/latents_std) + latents_mean`. Encoding with `* latents_std` instead
  inflates the latent by `latents_std**2` - for Krea-2-Turbo, 2.1x-7.9x per channel -
  and the transformer discards a latent that far outside its unit-variance working
  space and re-renders from scratch. Symptom: the composite leaves hard-edged,
  oddly coloured discs on faces, because the elliptical paste exposes an
  independently generated crop. This encoder is shared with latent-init img2img, so
  the same inversion degraded img2img. `tests/test_krea2_img2img.py` pins both
  directions.
- **Facial-interior mask.** Step 5 is the quality safeguard. Pasting the refined crop
  back wholesale replaces hair, ears, clothing and background with whatever the model
  redrew, which is what makes faces drift and can turn a good image uncanny. The
  composite is restricted to a feathered ellipse fitted to the detected box, so a
  regenerated jaw does not silently rewrite the rest of the head. Measured on a real
  Krea 2 run, 44.6% of pixels inside the facial region changed versus 0.16% outside.
- **Cost.** Roughly one extra generation per detected face. Automatic enablement is
  limited to prompts that actually look photographic, and the face-selection gates
  below skip faces that would gain little.
- **Face selection.** Faces smaller than `min_face_px` (default 32) or wider than
  40% of the frame are skipped: an already-resolved close-up gains little and only
  risks shifting the likeness. The largest `max_faces` detections win.
- **Detector model.** Resolved from an explicit request value, then
  `WEBBDUCK_KREA2_FACE_DETECTOR`, then `weights/faces/`. If nothing is cached it is
  downloaded once (~227 KB) from `opencv/opencv_zoo`
  (`models/face_detection_yunet/face_detection_yunet_2023mar.onnx`). Set
  `WEBBDUCK_WEIGHTS_DIR` to relocate the cache.
- **Memory.** While the detailer is enabled the transformer and prompt embeds are
  deliberately kept resident past the decode, since the crop pass needs them; both
  are released afterwards. VAE tiling is forced on in that path to bound decode
  memory.
- **Never fatal.** No faces, a missing detector, a detector that fails to load, and
  a crop that blows up mid-pass all return the unrefined image with a note. The
  reason, the detected/refined counts, the face and crop boxes, and per-pass timing
  are recorded in `face_detailer` in the run metadata.
- **Verified on hardware.** Two-friend photoreal prompt, production entry point
  (`krea2_worker_adaptive`), realism decoder on, compared against a matched
  no-detailer control. Before the normalization fix: 84.3% of facial pixels changed
  by >12/255 with a +12.9 blue bias, and the error survived a 4 px blur (21.4 mean),
  i.e. a large tone rewrite rather than detail. After: 13.8% of pixels changed, the
  tone error fell to 3.5, and the per-channel bias is neutral (-0.2/+0.4/+0.7).
  Change outside the ellipse is unchanged at 0.78, confirming the mask still
  contains the pass.

## Realism 2x decoder (optional)

Krea 2 latents use the Qwen-Image VAE latent space, which is identical to
Wan2.1's (same 16 channels, same `latents_mean`/`latents_std`).
[`spacepxl/Wan2.1-VAE-upscale2x`](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)
is a decoder-only finetune trained on real photos. Its last layer emits 12
channels that pixel-shuffle into a **2x** RGB image. Compared with the stock
decoder it renders sharper skin pores and hair, without the speckle or
airbrushed look. (The Civitai "Krea 2 Real VAE" is the same decoder with its 2x
head averaged back to 3 channels, which throws away the extra resolution.
WebbDuck uses the 2x head directly.)

- **Automatic for realistic prompts.** Selected by the realism router above. The API
  fields are `vae_decoder` (`stock` | `realism`) and `vae_decoder_output`
  (`requested` | `2x`).
- **Precedence.** Advanced Settings `realism_mode` (`always` / `never`) >
  deployment defaults > the automatic router > an explicit per-request value that
  the router did not set. `WEBBDUCK_KREA2_DECODER` and
  `WEBBDUCK_KREA2_DECODER_OUTPUT` supply the deployment default.
- **Measured trade-off.** At a matched 832x1216 output, the realism decoder measured
  about 13% more high-frequency energy than the stock decoder on a photoreal
  prompt. Some of that is the intended pore/hair detail, but it is not a free
  win, and the automatic router is the thing to revisit if results read grainy.
- **Output sizing.**
  - `requested` downsamples the 2x decode with Lanczos to the requested size,
    *after* the face pass so refinement runs at full decode detail. This also
    replaces the reduced-size output that otherwise appeared when the adaptive
    planner shrank the grid to fit the token budget. The resample matters for
    appearance, not just size: shipping the raw 2x buffer meant the gallery
    displayed it magnified, so every artifact read as coarse grain.
  - `2x` keeps the full decoder resolution (2x the effective grid).
  - Identity runs that use the realism decoder skip the Real-ESRGAN restore
    above.
- **Cost.** On a 5070 Ti, the latent for a 1024x1024 image decodes to 2048x2048 in about
  0.5 s, the same as the stock decoder. Peak decode memory is about 4.6 GB, and
  the stock VAE stays on the CPU. The first use downloads about 508 MB into
  the Hugging Face cache; after that, loading takes about 0.1 s.
- **Weights override.**
  - `WEBBDUCK_KREA2_REALISM_DECODER` takes a repo id or a local diffusers
    folder. The default is `spacepxl/Wan2.1-VAE-upscale2x`.
  - `WEBBDUCK_KREA2_REALISM_DECODER_SUBFOLDER` sets the subfolder; use `.` for
    the root. The default is `diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1`
    for the default repo.
- **Failure handling.** If the decoder cannot load or decode (offline, latent
  mismatch, OOM), the worker decodes with the stock VAE instead and records
  `error` and `fallback: "stock-vae"`. The generation does not fail.
- **Metadata.**
  - `krea_decoder` records
    `{mode, decoder, output, source, applied, decoded, to, resize?, load_seconds?, error?, fallback?}`.
  - `performance_timing.krea_decoder_load_seconds` records the decoder load time.
  - `width`/`height` record the saved artifact size, and `requested_*` records
    the original request.
- **Limits.** It is trained on real photos, so it can over-sharpen or slightly
  shift colours on anime, line art, and text. It improves how latents are
  rendered, but it cannot add detail that the latent grid never contained.
  Distant faces that cover only a few 16 px tokens need a detailer pass
  (backlog TASK-013).

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