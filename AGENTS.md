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
- Plugin model: optional web-app plugins (`plugins/webapps`) can run local or remote.
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
- `models/` registry/discovery for models, LoRAs, embeddings
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

4. Long-prompt behavior
- SDXL long prompts use chunked conditioning.
- Avoid reintroducing legacy experimental compression paths unless explicitly requested.

5. Modal UX over browser popups
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
- `WEBBDUCK_CATALOG_POLL_SECONDS` (default: `3.0`)
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
pip install -r requirements.windows.txt
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

3. Touch gallery behavior
- Keep infinite scroll + search + selection interactions compatible.
- Validate lightbox actions and metadata rendering.

4. Integrate/update optional web plugins
- Follow plugin search precedence: `WEBBDUCK_PLUGINS_DIR` -> `webbduck/plugins` -> `~/.webbduck/plugins`.
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

## 12. Source-of-Truth Docs

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

## 13. PR Expectations

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
