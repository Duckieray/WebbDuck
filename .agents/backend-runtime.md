# Backend And Runtime Reference

Use this file when working in `server/`, `core/`, `models/`, `modes/`, or `prompt/`.

## Server Layer

- `server/app.py`: main FastAPI app, route registration, queue management, plugin endpoints, `/docs/simple-guide`, `/ws`, `/health`, `/caption*`, gallery APIs, queue controls, and shutdown/unload actions.
- `server/events.py`: broadcasts JSON events to connected WebSocket clients.
- `server/state.py`: shared in-memory progress, stage, and VRAM snapshot for UI updates.
- `server/storage.py`: saves images, run metadata, manifest entries, and favorites.
- `server/thumbnails.py`: on-demand thumbnail generation with per-path locking.

## Worker And Pipeline Layer

- `core/worker.py`: serial GPU worker for generation and Real-ESRGAN upscaling; acquires and releases GPU leases around jobs.
- `core/generation.py`: normalizes dimensions, prepares input/mask state, injects LoRA triggers, selects generation mode, and records timing.
- `core/pipeline.py`: runtime profile resolution, SDXL pipeline build/load/unload, scheduler swaps, LoRA application, embedding loading, and second-pass attachment.
- `core/runtime.py`: device and dtype resolution with CUDA compatibility probing and strict-mode fail-fast behavior.
- `core/gpu_lease.py`: in-process GPU ownership coordination for WebbDuck core and local plugins.
- `core/perf.py`: per-thread generation timing helpers.
- `core/exceptions.py`: shared cancellation exception.

## FLUX Identity Personas (Baked Tuning)

`core/backends/flux.py` resolves native FLUX.2 identity conditioning and
`core/backends/flux_identity.py` applies three generic, A/B-validated persona
upgrades before the isolated worker runs:

- `face_crop` ("auto" default / "off"): each reference is re-framed around the
  strongest detected face (tight square, ~80% face fill, `BORDER_REPLICATE`
  padding so nothing is squished, resized to 1024px) and cached on disk.
  InsightFace "buffalo_l" is used when installed, OpenCV Haar cascade otherwise;
  refs with no detectable face pass through unchanged.
- `flux2_anchor_dup` (bool): duplicates the first reference into slot 1 because
  FLUX.2 conditions strongest on the earliest reference slot. Total stays capped
  at the worker's 5-reference limit (trailing refs dropped first).
- `face_focus` (bool): appends close-up portrait framing guidance to the prompt
  so the scene keeps the face large, centered, and sharp.

Request fields take priority over presets; saved personas persist these keys in
`BASE/.faceid_presets.json`. The 5-reference cap is enforced in
`core/backends/flux_worker.py` (`_MAX_FLUX_IDENTITY_REFERENCES`).

## FLUX.2 Latent-Init Img2Img

`core/backends/flux_worker.py` treats a source body + `strength < 1.0` as a
latent-init img2img pass: `_build_img2img_init` VAE-encodes the body to the
BN-normalized `(B,128,h//2,w//2)` init space, noises it to the selected start
sigma (`_img2img_sigma_schedule` truncating the real dynamic-shifted Klein
schedule), and the worker passes `latents=`/`sigmas=` into
`Flux2KleinPipeline.__call__`. Identity references remain image conditioning. The
init is built inside the per-tier OOM `try`, so an OOM while re-building a retry
tier demotes cleanly (832x1216 → 752x1096 → 664x976) instead of aborting; the VAE
is offloaded back to CPU after the body encode to give the transformer max
headroom. `core/backends/flux.py` forwards `strength` in the worker payload.
Note: `enable_sequential_cpu_offload` is incompatible with GGUF-quantized
transformers and is auto-guarded to the diffusers path.

## Krea 2 Identity Edit (krea2_identity_edit)

`core/backends/krea2_identity.py` is the pure-Python contract layer for the
instruction-based, identity-preserving persona adapter built on the community
LoRA `conradlocke/krea2-identity-edit` run on a Krea 2 checkpoint. It mirrors
the FLUX.2 persona concept but through the Krea dual-conditioning recipe
(semantic path via the image-grounded Qwen3-VL text encoder + appearance path
via the VAE-encoded normalized source latent prepended as clean tokens; the
sequence is `[text | source(frame=1) | target(frame=0)]` and only the target is
decoded). The module owns:

- the identity request snapshot / validation contract
  (`identity_settings_snapshot`, single-anchor ref in v1);
- identity weight resolution (`WEIGHT_SPECS` full/r128/r64, env overrides
  `WEBBDUCK_KREA2_IDENTITY_REPO` / `WEBBDUCK_KREA2_IDENTITY_WEIGHT`, GPU-ranked
  `preferred_rank`) — weights are never vendored in-repo;
- the ai-toolkit/ComfyUI -> Diffusers LoRA key conversion
  (`convert_lora_keys`, strict failures on unresolved tensors);
- dual-conditioning geometry (`edit_target_size`, `edit_position_ids`,
  `ref_boost_bias`, grid/token accounting the adaptive planner feeds on);
- the GQA-safe, mask-compatible attention processor factory
  (`mask_compat_processor`) and the source-preserving transformer forward
  (`edit_transformer_forward`).

The GPU runtime lives in `core/backends/krea2_worker.py`: a request carrying a
top-level `identity` dict dispatches out of the shared phased `_run` into
`_run_identity` before any text2img encode (same dispatch mirrored in
`krea2_worker_safe._run`). Sequence is text encode -> reference encode -> LoRA
install -> bordered denoise loop, with an OOM ladder that reconfigures
resident->`transformer-block` mode. The Qwen3-VL grounded encode
(`_grounded_encode_krea`) uses the processor from `_vlm_processor_source`
(`WEBBDUCK_KREA2_IDENTITY_PROCESSOR`, default `Qwen/Qwen3-VL-4B-Instruct`),
the pipeline's `text_encoder_select_layers` (12) and
`prompt_template_encode_start_idx` (34), and only passes
`mm_token_type_ids` when the installed transformers accepts it. The reference
is VAE-encoded into normalized source latents (`latents_mean/std`) and packed
with `pipe._pack_latents`; denoise uses the upstream identity schedule
(`sigmas = linspace(1.0, 1/steps, steps)`, `mu=1.15`, `set_begin_index(0)`)
and `edit_transformer_forward` with `ref_boost`.

Identity LoRAs install as low-rank residuals: dense `nn.Linear` modules bake
the delta into the weight (upstream `fuse_lora` math), while `ScaledFP8Linear`
keeps its FP8 `qweight` storage and applies `y = BaseFP8(x) + (alpha/rank)*B(A(x))`
at forward time — including the `torch._scaled_mm` native path in
`core/backends/krea2_native_fp8.py`, so identity edits never force a BF16
base. `krea2_worker_adaptive._adaptive_request` no longer gates identity jobs
behind `_guard_identity_staged`; it preserves identity geometry when planning
(skips resolution-scaling and default-step tuning) and records
`identity_enabled` on the plan.

Krea checkpoints advertise `identity_adapter=True` but deliberately NOT generic
`img2img` in `models/model_descriptor.py`; the UI keys the persona section off
the capability, not string detection.

## Captioning And Plugins

- `core/captioning_config.py`: plugin search roots and captioner discovery.
- `core/captioner.py`: captioner module loading, unload hooks, style prompts, and caption generation.
- `core/web_plugins.py`: local and remote web plugin discovery, asset mounting, router inclusion, and persisted remote plugin state.

## Models And Asset Discovery

- `models/registry.py`: discovers checkpoints, LoRAs, embeddings, and Hugging Face cache assets; persists `models.json`, `loras.json`, and `embeddings.json`.
- `models/upscaler.py`: Real-ESRGAN loader and weights-path resolution.

## Mode Selection Order

`modes/__init__.py` selects modes in this order:

1. `InpaintMode`
2. `TwoPassMode`
3. `Img2ImgMode`
4. `Text2ImgMode`

The complex smart-extend and pyramid outpaint logic lives in `modes/inpaint.py` and `modes/outpaint.py`.

## Prompt System

- `prompt/conditioning.py`: SDXL dual-encoder long-prompt chunking and conditioning injection.
- `prompt/management.py`: token counting, truncation, chunking, and prompt compression helpers.
- `prompt/experimental.py`: conditioning dispatch helpers and refiner conditioning.

## Common Backend Rules

- Queue GPU-heavy work instead of running it inline in request handlers.
- Keep request parsing, UI payloads, and persisted client state aligned.
- Preserve `catalog` refresh behavior when model discovery changes.
- Release GPU leases in `finally` blocks.
- Keep optional plugins non-fatal for the core app.
