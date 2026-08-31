# WebbDuck Agent Guide

This file is the working playbook for AI/code agents contributing to WebbDuck.
Use it as the first-stop reference for architecture, workflows, project rules, and run/test commands.

## 1. Project Snapshot

- App type: local-first SDXL image generation studio.
- Backend: FastAPI + Uvicorn.
- Frontend: vanilla JavaScript (ES modules), no Node build pipeline.
- Execution model: queue-first GPU worker (single worker path for GPU-heavy work).
- Shared in-process GPU lease coordinates WebbDuck core and local web plugins.
- Realtime: WebSocket events for status/queue/catalog updates.
- Storage: filesystem outputs + metadata JSON + manifest index for gallery search.
- Plugin model: optional web-app plugins (`plugins/webapps`) can run local or remote. The `plugins/webapps/` directory is gitignored and created dynamically by external integrations (dnaduck, duckmotion).
- DuckMotion/DNADuck are separately managed plugin repos; WebbDuck should not hard-require them.

## 1.1 Read These Docs First

When you need repo orientation, read these in order:

1. `.agents/README.md` - agent-specific index for the committed repo reference set.
2. `.agents/repo-overview.md` - runtime flow, subsystem map, and where to start editing.
3. `.agents/backend-runtime.md` - backend, worker, pipeline, model, and prompt internals.
4. `.agents/frontend.md` - UI structure, state flow, modules, and styles.
5. `.agents/plugins-tests-docs.md` - plugin contracts, test map, and doc-maintenance rules.
6. `docs/ARCHITECTURE.md` - top-level repo map and file ownership.
7. `docs/DEVELOPMENT.md` - contributor workflows and common update recipes.
8. `ui/README.md` - frontend structure and editing guidance.
9. `tests/README.md` - test suite map and focused verification.
10. `docs/PLUGINS.md` - captioner and web-app plugin contracts.

If repo layout or ownership changes, update `docs/ARCHITECTURE.md` and the affected `.agents/*.md` references before or alongside code changes.

## 2. Tech Stack

- Python 3.10+
- FastAPI / Starlette / Uvicorn
- PyTorch + Diffusers + Transformers + PEFT + Safetensors
- Pillow / OpenCV / NumPy / SciPy
- Optional captioning/upscale dependencies (see `requirements*.txt`)
- Frontend:
  - `ui/app.js`
  - `ui/core/*.js` (API/events/state/utils)
  - `ui/modules/*.js` (Gallery/Lightbox/Lora/Embedding/etc.)
  - `ui/styles/*.css`

Key folders:

- `server/` API + websocket + gallery/thumb serving
- `core/` worker, generation orchestration, pipeline lifecycle
- `models/` registry/discovery for models, LoRAs, embeddings; includes `metastore.py` — the canonical rich-metadata store
- `modes/` txt2img, img2img, inpaint, outpaint, two-pass mode logic
- `prompt/` SDXL conditioning + long-prompt chunking
- `plugins/` bundled optional plugin examples and manifests
- `tests/` pytest suite
- `docs/` architecture, contributor, user, plugin, and platform docs

## 3. Core Architecture Rules

1. Queue-first GPU work
- Do not run heavy generation directly in request handlers.
- GPU jobs should be enqueued and handled by `core/worker.py`.

2. Keep frontend/backend contracts aligned
- If request fields change in backend (`server/app.py`), update UI form wiring (`ui/app.js`) and persisted state (`ui/core/state.js`) as needed.

3. Catalog auto-refresh is expected behavior
- Model/LoRA/embedding changes should flow through registry refresh + websocket `catalog` updates.
- The metadata store (`models/metastore.py`) is auto-reloaded alongside registries.

4. WebbDuck is the canonical metadata owner
- Rich metadata (tags, families, descriptions, recommendations) lives in `webbduck_meta/`.
- External tools like PhoenixDraw read it via the `/meta/*` API instead of maintaining their own copy.
- The store auto-refreshes alongside the catalog watcher; call `reload_meta()` or hit `POST /meta/reload` to force re-read from disk.

5. Long-prompt behavior
- SDXL long prompts use chunked conditioning.
- Avoid reintroducing legacy experimental compression paths unless explicitly requested.

6. Modal UX over browser popups
- Prefer app modal components over native `alert/confirm/prompt`.

6. Shared GPU lease discipline
- All GPU-heavy runtimes in-process (core + plugins) must coordinate through `core/gpu_lease.py`.
- Always release lease tokens in `finally` blocks.
- Prefer bounded waits over indefinite blocking (`WEBBDUCK_GPU_LEASE_WAIT_SECONDS`).

7. Optional plugin isolation
- WebbDuck core must remain fully usable when optional plugins are missing, disabled, or failing.
- Plugin faults should surface clear errors and must not silently stall WebbDuck generation.

## 4. Coding Guidelines (Project-Specific)

- Make narrow, intentional edits; preserve existing patterns.
- Do not add new frameworks/build tooling unless explicitly requested.
- Prefer explicit naming and small helper functions over dense inline logic.
- Maintain backward-compatible defaults when changing UI controls.
- Keep server errors actionable; fail fast with clear messages where possible.
- Add/update tests for behavior changes when practical.
- Update docs when behavior, setup, commands, contracts, or repo structure changes.
- Treat documentation maintenance as part of the same task, not follow-up cleanup.

## 4.1 Agent Operating Rules

- Instruction precedence: system/developer instructions first, then explicit user request, then this file and repo conventions.
- No silent failure: never claim a command/test/check ran when it did not; explicitly report what was run and what was skipped.
- Strict scope control: implement only the requested task; if you find unrelated issues, create/follow up with a separate backlog task.
- Stop and report when requirements are ambiguous, credentials are missing, a requested action is destructive/irreversible, or instructions conflict.
- Read-only mode: for analysis/planning/review-only requests, inspect and explain without editing files or changing backlog state.

## 5. Git + Workspace Rules

- Never use destructive git commands (`reset --hard`, `checkout --`, etc.) unless explicitly requested.
- Do not revert user-owned or unrelated changes.
- Commit only relevant files.
- Exclude local artifacts/debug data from commits (examples: `outputs/`, `inpaint_input/`, local checkpoints, local LoRA/embedding/model binaries).
- Keep commit messages descriptive and scoped.
- Do not amend commits unless explicitly requested.
- Do not force-push unless explicitly requested.
- Worktrees are recommended (not required) for parallel work or high-conflict changes.
- If not using a dedicated worktree, verify `git status` before and after edits and keep touched files tightly scoped.

## 5.1 Backlog Privacy Rules

- Public backlog tasks live in `backlog/` and are commit-safe by default.
- Personal/environment-specific details must NOT be written to public Backlog.md task fields.
- Store personal/local details in `backlog-private/` only (this directory is gitignored).
- If a task needs local context, keep the public task high level and reference a local note like `backlog-private/TASK-<id>.local.md`.
- Never commit secrets, local paths, hostnames, tokens, or machine-specific instructions to `backlog/` or other tracked files.

## 5.2 Backlog Execution Rules

- Search existing backlog tasks before creating a new task.
- Create a task for non-trivial implementation work that requires planning/decisions.
- Skip task creation for trivial/mechanical edits unless the user explicitly requests task tracking.
- Keep public backlog entries outcome-focused and collaborator-safe; move personal execution details to `backlog-private/`.

## 5.3 Verification Policy (Tiered)

- Tier 0 (default): run fast targeted checks for touched areas (focused `pytest` modules/tests, quick smoke checks).
- Tier 1 (feature-level): run broader integration validation when behavior crosses API/UI/worker boundaries.
- Tier 2 (broad): run larger/full suite only when risk is high, shared interfaces changed, or user requests it.
- Always report exactly which verification commands were executed and their result.

## 6. Environment Variables (Important)

- `WEBBDUCK_OUTPUT_DIR`
- `WEBBDUCK_PORT`
- `WEBBDUCK_MODELS_DIR` (model library root override)
- `WEBBDUCK_CHECKPOINT_DIR` (explicit checkpoint root override)
- `WEBBDUCK_HF_CACHE_DIR` (explicit Hugging Face hub cache override)
- `WEBBDUCK_WEIGHTS_DIR` (explicit upscaler weights root override)
- `WEBBDUCK_PLUGINS_DIR` (optional plugin root override)
- `WEBBDUCK_LORA_DIR`
- `WEBBDUCK_EMBEDDING_DIR`
- `WEBBDUCK_META_DIR` (default: `webbduck_meta/` in project root)
- `WEBBDUCK_CATALOG_POLL_SECONDS` (default: `3.0`)
- `WEBBDUCK_IDLE_UNLOAD_SECONDS` (default: `300.0`)
- `WEBBDUCK_THUMB_CONCURRENCY` (default: `2`)
- `WEBBDUCK_DEVICE` (`cuda` or `cpu`)
- `WEBBDUCK_DTYPE` (`float16`, `bfloat16`, `float32`)
- `WEBBDUCK_STRICT_DEVICE=1` (fail fast instead of auto fallback)
- `WEBBDUCK_GPU_LEASE_WAIT_SECONDS` (default: `180`; fail fast if GPU lease is stuck)
- `WEBBDUCK_USE_IPC_COLLECT=1` (optional; enables aggressive CUDA IPC cleanup)

## 7. WSL/Linux Commands

Recommended environment name: `webbduck`.

### Setup

```bash
cd <path-to-webbduck-repo>
conda create -n webbduck python=3.10 -y
conda activate webbduck
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
pip install -r requirements.txt
mkdir -p checkpoint/sdxl lora embeddings outputs weights
```

### Run (scripted)

```bash
./startup.sh
./startup.sh --port 8020
./startup.sh --env webbduck --output ./outputs --port 8010
```

### Run (direct)

```bash
conda activate webbduck
python run.py --output ./outputs --port 8010
```

### Tests

```bash
conda run -n webbduck pytest -v -m "not slow"
conda run -n webbduck pytest tests/test_server.py -v
conda run -n webbduck pytest tests/test_prompt_conditioning.py -v
```

## 8. Windows Commands (PowerShell)

### Setup

```powershell
cd <path-to-webbduck-repo>
conda create -n webbduck python=3.10 -y
conda activate webbduck
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
mkdir checkpoint\sdxl, lora, embeddings, outputs, weights
```

### Run

```powershell
conda activate webbduck
python .\run.py --output .\outputs\ --port 8010
```

### Tests

```powershell
python -m pytest -q -s tests\test_server.py -m "not slow"
python -m pytest -q -s tests\test_modes.py tests\test_prompt_conditioning.py
```

### Optional runtime overrides

```powershell
$env:WEBBDUCK_DEVICE="cuda"
$env:WEBBDUCK_DTYPE="float16"
$env:WEBBDUCK_STRICT_DEVICE="1"
python .\run.py --port 8010
```

## 9. GPU Compatibility Notes

- PyTorch wheel/channel must match GPU generation and driver support.
- If `torch.cuda.is_available()` is false or you get kernel-image errors, reinstall torch from a compatible channel.
- Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 10. Common Agent Tasks

1. Add backend option
- Update endpoint form fields in `server/app.py`.
- Thread value into settings payload.
- Apply setting in `core/generation.py` / relevant mode.
- Add UI control in `ui/index.html` + wiring in `ui/app.js` + state persistence.
- Add tests.

2. Add model/asset type discovery
- Extend `models/registry.py`.
- Ensure watcher refresh paths/signature include necessary files.
- Ensure API endpoints return compatible metadata for frontend managers.

3. Add/edit rich metadata
- Edit files directly in `webbduck_meta/` (the metadata store) — this is the canonical source.
- Use the `MetaStore` API from Python: `meta_store.update_checkpoint_meta(name, tags=[...], ...)`.
- External consumers (including PhoenixDraw) should read from the `/meta/*` HTTP endpoints.
- The store auto-refreshes alongside the catalog watcher; call `reload_meta()` or hit `POST /meta/reload` to force re-read from disk.
- Extend `models/registry.py`.
- Ensure watcher refresh paths/signature include necessary files.
- Ensure API endpoints return compatible metadata for frontend managers.

4. Touch gallery behavior
- Keep infinite scroll + search + selection interactions compatible.
- Validate lightbox actions and metadata rendering.

4. Integrate/update optional web plugins
- Follow plugin search precedence: `WEBBDUCK_PLUGINS_DIR` -> `<repo>/plugins` -> `~/.webbduck/plugins`.
- If an external plugin repo ships its own installer, target the WebbDuck repo root when running it.
- Keep plugin integration optional and non-blocking for core generation flow.

## 11. Troubleshooting Quick Reference

- Port already in use:
  - Change port (`--port 8020`) or kill the existing process.
- `ModuleNotFoundError: webbduck`:
  - Run from repo root and use `python run.py` (path handling is built into `run.py`).
- Hugging Face symlink warning on Windows:
  - Enable Developer Mode or run elevated; warning is often non-fatal.
- `no kernel image is available for execution on the device`:
  - Torch/CUDA wheel mismatch for the GPU architecture.
- Generation stuck at `0%` or waiting forever:
  - Check `GET /gpu/lease` to see current holder.
  - If lease holder is stale, restart WebbDuck.
  - Tune `WEBBDUCK_GPU_LEASE_WAIT_SECONDS` to avoid indefinite waits.
- WebbDuck performance regressed after plugin testing:
  - Confirm no local plugin job is still running.
  - Verify plugin install path precedence (a stale plugin copy in a higher-priority path can override expected code).
- Slow generation after feature changes:
  - Check whether conditioning path changed, whether extra passes were enabled, and whether dtype/device fallback occurred.

## 12. IP-Adapter FaceID Feature Reference

### Backend API
- `GET /ip-adapter/refs` — list uploaded ref images (name, path, url)
- `POST /ip-adapter/refs/upload` — upload ref image (multipart); returns `{url: "/outputs/refs/filename.png"}`
- `DELETE /ip-adapter/refs/{filename}` — delete uploaded ref
- `GET /ip-adapter/presets` — list saved face presets
- `POST /ip-adapter/presets` — save preset (JSON body: `{name, refs, adapter_scale, lora_scale}`)
- `DELETE /ip-adapter/presets/{name}` — delete preset

### Refs storage
- Uploaded refs: `BASE/refs/` (e.g. `outputs/refs/`)
- Served via StaticFiles at `/outputs/refs/{filename}`
- Filesystem resolution: `resolve_web_path()` converts `/outputs/refs/x.png` → `BASE/refs/x.png`

### Presets storage
- Single JSON file: `BASE/.faceid_presets.json`
- Stores `{name: {refs: [...], adapter_scale: 0.55, lora_scale: 0.60}}`

### Frontend wiring
- **IP-Adapter section** (`section-ip-adapter`): collapsible with enable checkbox, ref image grid, upload button, preset save/load/delete, scale sliders
- **Hidden input** `ip-adapter-refs-json`: stores selected ref paths as JSON array; synced to `state.identityAdapterRefs`
- **`setupIpAdapterManager()`**: initializes ref grid (fetches server refs, toggles selection), upload handler, preset CRUD, MutationObserver to reload on section expand
- **`collectFormData()`**: reads `ip-adapter-refs-json`, passes `reference_images` array in `identity_adapter` JSON payload
- **State keys**: `identityAdapterEnabled`, `identityAdapterRefs` (array), `identityAdapterScale`, `identityAdapterLoraScale`

### Pipeline
- `apply_identity_adapter()`: resolves ref paths via `resolve_web_path()` before `cv2.imread()`
- FaceID LoRA loaded internally by `load_ip_adapter()` as adapter `faceid_0`

## 13. Source-of-Truth Docs

- `README.md`
- `.agents/README.md`
- `.agents/repo-overview.md`
- `.agents/backend-runtime.md`
- `.agents/frontend.md`
- `.agents/plugins-tests-docs.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/WINDOWS_TESTING.md`
- `docs/USER_GUIDE.md`
- `docs/SIMPLE_GUIDE.md`
- `docs/PLUGINS.md`
- `plugins/README.md`
- `tests/README.md`
- `ui/README.md`

When behavior changes, keep these docs in sync.

If code, commands, routes, file ownership, plugin contracts, or test workflow changes, update the relevant docs in the same task before stopping.

Doc ownership notes:

- Update `README.md` for install/startup/top-level capability changes.
- Update `.agents/*.md` when repo navigation guidance, subsystem ownership, or agent workflow details change.
- Update `docs/ARCHITECTURE.md` when folder ownership or file responsibilities change.
- Update `docs/DEVELOPMENT.md` when contributor workflows or common recipes change.
- Update `docs/USER_GUIDE.md` or `docs/SIMPLE_GUIDE.md` when user-visible workflows change.
- Update `docs/PLUGINS.md` and `plugins/README.md` when plugin contracts or install steps change.
- Update `ui/README.md` and `tests/README.md` when frontend or validation structure changes.

## 14. PR Expectations

- Summarize what changed and why, scoped to the task.
- List verification commands run and whether they passed/failed/skipped.
- Call out risk areas, trade-offs, and any follow-up tasks created.
- For UI changes, include screenshots or a short recording in the PR description when practical.

<!-- BACKLOG.MD MCP GUIDELINES START -->

<CRITICAL_INSTRUCTION>

## BACKLOG WORKFLOW INSTRUCTIONS

This project uses Backlog.md MCP for all task and project management activities.

**CRITICAL GUIDANCE**

- If your client supports MCP resources, read `backlog://workflow/overview` to understand when and how to use Backlog for this project.
- If your client only supports tools or the above request fails, call `backlog.get_workflow_overview()` tool to load the tool-oriented overview (it lists the matching guide tools).

- **First time working here?** Read the overview resource IMMEDIATELY to learn the workflow
- **Already familiar?** You should have the overview cached ("## Backlog.md Overview (MCP)")
- **When to read it**: BEFORE creating tasks, or when you're unsure whether to track work

These guides cover:
- Decision framework for when to create tasks
- Search-first workflow to avoid duplicates
- Links to detailed guides for task creation, execution, and finalization
- MCP tools reference

You MUST read the overview resource to understand the complete workflow. The information is NOT summarized here.

</CRITICAL_INSTRUCTION>

<!-- BACKLOG.MD MCP GUIDELINES END -->

## 14. Session Log — Official FaceID Wrapper Removal (2026-08-26)

The official `third_party/IP-Adapter` wrapper (`core/backends/sdxl_faceid.py`, `core/run_official.py`) was removed after side-by-side testing across 4 SDXL models confirmed it produces functionally identical output to the diffusers-native path. The PlusV2 variant was also removed — it looked plasticky and was non-functional in diffusers 0.36.0 anyway (Perceiver Resampler unsupported). Only the native `faceid_sdxl` path remains.

### Server-side preset resolution (2026-07-06)
- Added `_resolve_identity_adapter_preset()` in `server/app.py` that merges a saved preset into the identity adapter config when `preset_name` is provided.
- An external app can now POST `/test` or `/generate` with just `{"preset_name": "my-preset", "adapter_scale": 0.7}` — the preset's `type`, `refs`, `adapter_scale`, and `lora_scale` are merged as defaults, with request fields taking priority.
- The helper is wired into both endpoints right after `json.loads(identity_adapter)`.
- `reference_images` already supports both web URLs (`/outputs/refs/face.png`) and absolute filesystem paths — `resolve_web_path()` handles both transparently.

### Soft vs hard model unload (2026-07-08)
- `pipeline_manager.unload_all()` is a soft unload by default: active pipelines are offloaded and dropped, CUDA cache is cleared, but CPU-side component caches stay warm.
- Use `pipeline_manager.unload_all(clear_caches=True)` or `POST /models/unload_all?clear_caches=true` only when a hard unload is needed.
- Normal idle unload and model switching should preserve `_UNET_CACHE`, `_TEXT_COMPONENT_CACHE`, `_VAE_CACHE`, `_SCHEDULER_CACHE`, and `_TOKENIZER_CACHE` so reloads stay fast.

### FLUX.2 worker VRAM resilience (2026-08-27)
- `core/backends/flux_worker.py` no longer dies on the first `torch.cuda.OutOfMemoryError`. It retries within the same process through `_flux_demotion_tiers`: pixel area first (0.9x/0.8x snaps), then identity/conditioning image count (4->3->2->1), then LoRA adapter count, then a 0.75x last-resort config. Every tier is non-increasing in all resource dimensions and re-seeds the generator so demoted output stays reproducible from the requested seed.
- Worker records `oom_retry_count`, `demotion_tier`, `effective_width/height`, `effective_conditioning_image_count`, and `effective_lora_count` in the runtime dict (`flux.py` surfaces it as `settings["flux_runtime"]`).
- Pure `_flux_demotion_tiers` helper is importable without a GPU and covered in `tests/test_flux_memory_metadata.py`.
- On exhaustion (no tier fits) the original OOM propagates with the worker log detail; the UI keeps surfacing an actionable error card.

### Generic FLUX.2 persona tuning baked into the identity adapter (2026-08-28)
- `core/backends/flux_identity.py` (new) applies three A/B-validated persona upgrades inside `_resolve_flux_identity` (`core/backends/flux.py`) before the worker starts:
  - `face_crop` ("auto" default / "off"): re-frames every reference around the strongest detected face (tight square, ~80% face fill, `BORDER_REPLICATE` padding — never squished, resized to 1024px, disk-cached). InsightFace "buffalo_l" used when installed; OpenCV Haar cascade fallback. Refs with no detectable face pass through unchanged, so `face_crop` is safe by default.
  - `flux2_anchor_dup` (bool): duplicates the first reference into slot 1 because FLUX.2 conditions strongest on the earliest reference slot. Capped at the worker's 5-reference limit (trailing refs dropped).
  - `face_focus` (bool): appends close-up portrait framing guidance to the identity-augmented prompt.
- All three keys flow through persona presets (`BASE/.faceid_presets.json` via `server/app.py` `_resolve_identity_adapter_preset`/`save_preset`), so a Della-style caller can POST `/generate` with `preset_name` plus optional field overrides and get the tuned behavior in one request.
- UI exposes the three toggles in the Identity / Persona section, visible only when a FLUX.2 model is selected (`ui/app_main.js`, `ui/modules/PersonaIdentityUI.js`, `ui/core/state.js`).
- Identity A/B numbers that justify the defaults: anchor×2 close-up config scores ~0.784 mean face-cosine (max 0.820, 3/4 in high band) vs 0.729 for the loose baseline; LoRA reduction hurts (0.740/0.682); guidance is a no-op on distilled klein; one-ref is worst (0.694).
- Covered by `tests/test_flux_identity_prep.py`; updated `tests/test_flux_identity_personas.py` for the richer `identity_adapter_debug` dict.

### Native diffusers-path FLUX.2 LoRA support (2026-08-30)
- `core/backends/flux_lora.py` now handles FLUX.2 LoRAs saved with native diffusers module paths (`transformer.*`) but the Kohya leaf suffix (`lora_down`/`lora_up`/`alpha`). Diffusers 0.39/0.40 misdetect this layout as flat Kohya and run `_convert_kohya_flux2_lora_to_diffusers`, which only understands `lora_unet_*` — zero adapters register, surfacing as "Adapter name(s) ... not in the list of present adapters".
- `_maybe_convert_diffusers_suffix_lora_to_peft()` detects the layout (native `transformer.` paths + `.lora_down.weight` suffix) and applies a suffix-only PEFT conversion (drop `dora_scale`, rename `lora_down`→`lora_A` scaled by `alpha/rank`, `lora_up`→`lora_B`), then loads the converted dict via `pipe.load_lora_weights(dict, ...)`.
- Only that specific layout triggers the conversion; PEFT (`lora_A`) and flat OneTrainer (`lora_unet_*`) LoRAs keep the existing file-path load path unchanged.
- Covered by `tests/test_flux_lora.py`; the real Areola LoRA yields 288 valid PEFT keys and passes `Flux2KleinPipeline.lora_state_dict` unchanged.

### FLUX-dev LoRAs are NOT portable to FLUX.2 Klein (2026-08-30)
- Investigated `flux_zendaya.safetensors` (PEFT `lora_A`/`lora_B`, `transformer.*` paths). Conclusion: a FLUX.1-dev-style LoRA cannot be meaningfully applied to a FLUX.2 transformer — the base-model geometry is incompatible, not just the file format:
  - Block count: LoRA targets 38 single + 19 double blocks; FLUX.2 Klein has only 24 single + 8 double (over half the LoRA's keys point to nonexistent blocks).
  - Feature dim: LoRA `inner_dim` 3072 vs Klein 4096 (zero-padding would not reproduce the intended style).
  - Single-block structure: FLUX.1 uses separate `attn.to_q/to_k/to_v`; FLUX.2 replaces these with a fused `attn.to_qkv_mlp_proj`, so the single-block keys have no matching module.
- No remap/conversion recovers this in effect. FLUX-dev LoRAs should be used on a real FLUX.1-dev/schnell checkpoint, or a FLUX.2-native retrain sourced. Do not attempt to loft/port FLUX.1 LoRAs into FLUX.2 Klein.

### LoRA URL Backfill & Metastore Schema Update (2026-08-30)
- Added `url: str | None = None` to the `_LoraMeta` and `_EmbeddingMeta` schemas in `models/metastore.py` (and updated related `_CheckpointMeta` formatting).
- Backfilled 57 missing CivitAI URLs for on-disk LoRAs in the local metastore by parsing the raw CivitAI downloads-page HTML and using best-effort API searches.
- Enforced a strict domain policy (`civitai.red` for NSFW, `civitai.com` for non-NSFW) during backfill.
- Highly ambiguous models and API-hidden extreme NSFW items were left unset to avoid polluting the metadata with incorrect provenance. Local, in-house, and plugin entries remain deliberately unset.
