---
id: TASK-005
title: Make WebbDuck generation architecture-agnostic
status: To Do
assignee: []
created_date: '2026-08-20 22:20'
updated_date: '2026-08-20 22:20'
labels:
  - architecture
  - models
  - generation
  - refactor
dependencies: []
references:
  - models/registry.py
  - models/metastore.py
  - core/pipeline.py
  - core/generation.py
  - core/worker.py
  - modes/
  - server/app.py
  - ui/app.js
  - ui/core/api.js
  - ui/core/state.js
documentation:
  - docs/ARCHITECTURE_AGNOSTIC_GENERATION.md
  - docs/ARCHITECTURE.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor WebbDuck so the selected model/checkpoint is the only normal user-facing routing choice. WebbDuck must introspect the selected model, resolve the correct generation backend/runtime automatically, apply model-specific defaults and constraints, and expose only capabilities supported by that model.

The implementation must preserve the current SDXL/Pony/Illustrious workflows while removing SDXL-specific assumptions from top-level generation orchestration. FLUX and Krea 2 are initial non-SDXL validation targets, not special product modes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The normal Studio UI has one model selector and does not require an architecture/engine/backend selector.
- [ ] #2 Every runnable model resolves to a normalized descriptor containing capabilities, constraints, defaults, runtime hints, and detection diagnostics.
- [ ] #3 Unknown single-file or directory models are not silently classified as SDXL; unsupported models expose actionable diagnostics.
- [ ] #4 Generation backend selection is performed by a backend resolver from the selected model descriptor, with architecture-specific code isolated behind backend adapters.
- [ ] #5 Existing SDXL text2img, img2img, inpaint/outpaint, LoRA, embedding, second-pass, seed, and metadata behavior remains regression-covered.
- [ ] #6 Model capabilities automatically enable/disable/hide incompatible Studio controls without requiring the user to know model architecture.
- [ ] #7 LoRA/embedding/adapter compatibility is resolved structurally rather than only by exact `arch` string equality.
- [ ] #8 Token/prompt diagnostics are capability-specific and no longer assume every model has SDXL dual-CLIP/77-token semantics.
- [ ] #9 The worker, storage layer, gallery, and request handlers use generic model/backend contracts and do not contain architecture-routing conditionals.
- [ ] #10 Backend runtimes may execute in-process or in isolated environments without changing the public generation API, and all GPU work obeys the shared GPU lease.
- [ ] #11 A FLUX checkpoint can be selected from the same model list as existing checkpoints and generate through the normal API/UI without an engine selector.
- [ ] #12 A Krea 2 checkpoint can be selected from the same model list and automatically applies its compatible controls/defaults without an engine selector.
- [ ] #13 Switching between structurally different models safely unloads/reloads runtime resources without stale adapters or VRAM leaks.
- [ ] #14 Output metadata records the selected model plus a safe descriptor/backend/effective-settings snapshot for reproducibility and diagnostics.
- [ ] #15 Architecture-neutral checkpoint roots are supported while legacy `checkpoint/sdxl` paths remain compatible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests that lock down the existing SDXL model discovery, API, generation modes, adapters, seed behavior, and metadata before refactoring.
2. Introduce model candidates/descriptors, capability/constraint schemas, and an extensible detector chain while preserving the current registry response fields.
3. Generalize checkpoint discovery/catalog signatures to neutral image-model roots with backward-compatible scanning of existing roots.
4. Add an image-backend protocol, registry, resolver, and opaque runtime-session abstraction.
5. Wrap the current `core/pipeline.py`, SDXL prompt conditioning, and `modes/` behavior behind the first backend adapter without changing user-visible behavior.
6. Replace direct `pipeline_manager` assumptions in `core/generation.py`, `core/worker.py`, health/idle-unload paths, and server validation with generic runtime-session/backend services.
7. Enrich model API responses with capabilities/defaults/constraints and make the UI capability-driven while retaining the existing single model selector.
8. Replace exact architecture filtering for LoRAs/embeddings/second pass with a compatibility service sourced from model/asset descriptors plus metadata overrides.
9. Add isolated-runtime support for backends whose dependencies conflict with WebbDuck's stable environment; preserve GPU lease, progress, cancellation, and normalized errors.
10. Add FLUX detection/backend support as the first non-SDXL proof of the abstraction.
11. Add Krea 2 detection/backend support as the second independent proof, including model-specific defaults and limited capabilities where appropriate.
12. Harden quantized/single-file discovery and backend selection after official-format paths are stable.
13. Remove remaining SDXL-first naming/assumptions from core documentation only after compatibility shims are no longer needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The authoritative design and migration rules are in `docs/ARCHITECTURE_AGNOSTIC_GENERATION.md`.

Key invariant: architecture is internal metadata. Model selection drives automatic backend resolution. Do not solve this task by adding SDXL/FLUX/Krea tabs or an engine dropdown.

Initial external validation targets documented on 2026-08-20 are `black-forest-labs/FLUX.2-klein-4B` and `krea/Krea-2-Turbo`; the design must not hard-code those IDs.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Existing real SDXL pipelines work, not mock-only tests
- [ ] #2 FLUX real-model smoke test passes through the normal model-centric workflow
- [ ] #3 Krea 2 real-model smoke test passes through the normal model-centric workflow
- [ ] #4 No architecture selector is required by UI or API clients
- [ ] #5 No frontend console errors and no backend tracebacks in supported flows
- [ ] #6 VRAM/resource cleanup validated across model switching
- [ ] #7 LoRA/adapter compatibility verified for each backend where supported
- [ ] #8 Seed behavior verified as deterministic where the backend supports deterministic seeding
- [ ] #9 UI state accurately reflects selected-model capabilities/defaults
- [ ] #10 API compatibility tests cover existing clients using `base_model`
- [ ] #11 Architecture/design/user/developer docs updated to reflect implemented state
- [ ] #12 Committed with descriptive changes and targeted/full verification results recorded
<!-- DOD:END -->
