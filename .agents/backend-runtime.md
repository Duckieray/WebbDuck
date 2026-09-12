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

### Identity-aware adaptive planning + OOM retry reconfigure (Phase 5)

`core/backends/krea2_worker.py::configure_krea_identity_transformer` is the
single auditable reconfiguration path shared by the identity initial setup and
the CUDA OOM retry ladder: CPU-offload the transformer, gc + `empty_cache`,
re-probe live hardware, then select the profile (optionally forced to
`mode_override`, e.g. `"transformer-block"` on a resident OOM). `_run_identity`
uses it for both its setup and its fallback attempt, so the retry is no longer
inlined offload/cleanup code.

`krea2_worker_adaptive.py` gives identity jobs the same request-level guard the
text2img path gets. `_identity_token_budget` halves the accelerator's text2img
target-token budget because an identity forward packs reference + target grids
into one combined image sequence (`combined_image_tokens` = 2 × grid area for a
single reference). `_identity_token_plan` mirrors the worker's `edit_target_size`
containment (source AR, `fit_mode`, `max_megapixels`, 16-px grid) to predict the
effective output — with `reference_tokens`, `target_tokens`, and
`combined_image_tokens` — and, only when the reference is readable, shrinks the
requested box (same-AR fit) so the predicted target grid fits. The plan's
`identity` dict carries this accounting; steps are never tuned for identity.
Without a readable reference the planner takes no geometry action.

Krea checkpoints advertise `identity_adapter=True` but deliberately NOT generic
`img2img` in `models/model_descriptor.py`; the UI keys the persona section off
the capability, not string detection.

### 5070 Ti validation: sync block offload + ref_boost default (Phase 6/7)

On a real 16 GB RTX 5070 Ti the initially-chosen streamed block offload
(`use_stream=True`) accumulates GPU-resident weights — each manual per-block
forward during denoise leaves the block on the GPU (+~0.45 GiB each) until it
OOMs at block ~9 regardless of resolution. `_configure_execution` now forces
`use_stream=False` for the block path, giving `transformer-block-sync-2`; the
fp8 checkpoint keeps 12.2 GiB on CPU, ~0.42 GiB/block touches the GPU, and
768x768 10-step finishes in ~4 m 45 s / 28-step in ~11 m on 15.5 GiB.

A/B tuning showed `ref_boost` — not steps — governs likeness-vs-prompt balance:
4.0 correlates ~0.81 with the reference (composition copied wholesale), 1.0
drops to ~0.695 and follows the prompt. `DEFAULT_REF_BOOST` and the server
preset default are therefore 2.0. The UI exposes one unified 0..1 Identity
Strength slider (`adapter_scale = slider` on SDXL/FLUX; `ref_boost = 1 + 10*slider`
on Krea), and persona presets round-trip the same mapping.

### Cost breakdown and tuning (2026-09-09)

Per-step cost is dominated by **two full transformer forwards** (CFG cond +
uncond — any `guidance > 0` encodes the negative prompt, and the UI cfg passes
through) over a ~30B fp8 model whose blocks are streamed CPU<->GPU per forward.
Low VRAM (~6 GB) is by design, not headroom.

- **Paired streaming (`edit_transformer_forward_paired`):** when both CFG rows
  run, `_denoise_one_identity` uses the `paired-block` execution mode
  (`WEBBDUCK_KREA2_IDENTITY_PAIRED`, default on for CUDA). No accelerate hooks:
  the transformer sits on CPU with outer modules pinned, and each block is
  loaded once per step and applied to positive then negative hidden streams
  before offload. Identical math to two separate forwards. Halves transfer +
  per-block Python dispatch. `paired_oom` falls back to hook-based
  `transformer-block`.
- **A/B knobs (env-only, identity path only):** `WEBBDUCK_KREA2_IDENTITY_STEPS`
  (int), `WEBBDUCK_KREA2_IDENTITY_GUIDANCE` (float; `0`/negative skips the
  uncond forward), `WEBBDUCK_KREA2_IDENTITY_CFG_FREE=1` (= guidance 0). Applied
  by `apply_identity_perf_overrides` right after the request defaults resolve.
- **Resident block prefix (2026-09-08):** the transformer's weights never change
  during denoising, so `_configure_execution` stages the first N leading blocks
  on the GPU once per job instead of re-streaming them every step. N comes from
  `_identity_resident_blocks(free_vram, reserve, per_block, n_blocks)` —
  budgeted as measured free VRAM minus this resolution's activation reserve
  minus `_KREA_RESIDENT_WORKSPACE_GB` (1.5), floored at 0 and capped at
  `n_blocks` (~0.42 GB fp8/block). `WEBBDUCK_KREA2_IDENTITY_RESIDENT_BLOCKS`
  (default `auto`; `0` disables) forces the count. The count is recorded on
  `pipe.transformer._krea_resident_blocks` and threaded into
  `edit_transformer_forward_paired` / `edit_transformer_forward_nohooks` as
  `n_resident`; resident-prefix blocks are never offloaded. Execution-mode
  strings: `paired-block-N` (partial) or `paired-resident` (all resident).
  `_run_identity` reports `resident_blocks` in the runtime dict and the OOM
  ladder steps down: `paired_resident_oom` (drop the prefix, keep paired
  streaming) before the full `paired_oom` -> `transformer-block` retry.
- **Megapixel cap is not the binding limit — the token budget is:** an identity
  forward packs `[text | reference grid | target grid]` into ONE combined
  sequence (~2x image tokens per output grid + prompt tokens), so on this card
  the adaptive planner caps the target grid at 1792 tokens (half the 3584
  text2img budget) and any request ≥ ~800x1200 collapses to ~528x784 effective
  (~0.41 MP) regardless of `max_megapixels` (default 2.0, clamp 0.125..2.0).
- **`identity.token_budget` / `WEBBDUCK_KREA2_IDENTITY_TOKEN_BUDGET`:** A/B
  escape hatch for that target-grid token budget (`auto`/empty = 1792 halved
  default; integer = explicit, snapped to the 16-token grid; request-level
  `token_budget` in the identity payload beats the env var). Hardware-validated
  tiers on the 5070 Ti: 1792 → 528x784; 2048 → 576x864 (~4240 combined+text,
  just past the ~4096 training envelope); 2688 → 672x1008; 3584 → 768x1152
  (~7320 combined+text). All tiers completed without NaN/gray-wash/fallback but
  past ~2048 the combined sequence exceeds the LoRA's training envelope, so
  visual quality must be eyeballed before promoting a tier.
- **Artifact upscale to requested size:** the worker saves at the VRAM-capped
  effective resolution (e.g. 528x784 for a 832x1216 portrait request); without
  post-scaling that reads as a compressed/low-detail render. When a request is
  adapted, `krea2.py` (server process, not the isolated runtime) applies
  `_maybe_upscale_identity_artifact` to each returned image — Real-ESRGAN x2/x4
  (smallest factor whose upscale clears the target, then LANCZOS to the exact
  `requested_width`/`requested_height`), plain-LANCZOS fallback if the upscaler
  weights/libs are missing — and records the note in `settings["krea_upscale"]`
  plus `performance_timing.krea_upscale_seconds` (measured ~2.1 s on the 5070
  Ti). Original dims come from `settings["requested_*"]` (set by
  `_apply_effective_request_settings`); the worker's own `request` holds only
  the adapted dims. Disable with `WEBBDUCK_KREA2_IDENTITY_UPSCALE=0`.

## Captioning And Plugins

- `core/captioning_config.py`: plugin search roots and captioner discovery.
- `core/captioner.py`: captioner module loading, unload hooks, style prompts, and caption generation.
- `core/web_plugins.py`: local and remote web plugin discovery, asset mounting, router inclusion, and persisted remote plugin state.

## Models And Asset Discovery

- `models/registry.py`: discovers checkpoints, LoRAs, embeddings, and Hugging Face cache assets; persists `models.json`, `loras.json`, and `embeddings.json`. LoRA arch detection (`detect_lora_arch`) distinguishes FLUX.1 vs FLUX.2 using `ss_base_model_version` metadata first (`flux1-dev`/`flux1-schnell` vs `flux2_klein_9b`/`klein`), then structural keys (`to_qkv_mlp_proj`/`time_guidance_embed` = FLUX.2; `guidance_in`/`time_text_embed` = FLUX.1), then Klein block-geometry caps (24 single / 8 double) and hidden-size shapes (3072 vs 4096). Kohya `diffusion_model.`/`lora_unet_` prefixes are version-agnostic and never decide the arch.
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
