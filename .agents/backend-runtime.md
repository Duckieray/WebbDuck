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
