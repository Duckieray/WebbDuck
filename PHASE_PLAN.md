# WebbDuck Phase Plan

## Scope
This file tracks implementation status for the 19-item improvement list discussed on branch `too_many_updates`.

## Phase Map

### Phase A: Stability + Control Plane
Includes: #10, #13, #14, #15, #16

Status: `DONE` (commits: `60c2189`)

Completed:
- #10 Auto-normalize width/height to multiples of 8 on frontend and backend.
- #13 Cancel running generation jobs from UI/queue without killing app.
- #14 Strengthened model unload path when switching models and clearing caches.
- #15 Added web endpoint/button to shut down app.
- #16 Added web endpoint/button to unload all models.

---

### Phase B: Settings + UX Baselines
Includes: #1, #2, #3, #4, part of #6

Status: `DONE` (commits: `7e5d176`)

Completed:
- #1 LoRA dropdown excludes active LoRAs and restores them on removal.
- #2 Coffee warning threshold moved into Settings modal (top-bar access).
- #3 Added always-visible status percentage in top-right pill.
- #4 Randomize seed button now sets true random mode (clears seed), not fixed random seed.
- #6 (partial) Better visible status percentage behavior.

---

### Phase C: Timing + Status Accuracy
Includes: #5, remainder of #6

Status: `DONE` (commits: `0024d5c`)

Completed:
- #5 Added generation timing metadata and surfaced it in lightbox settings.
  - Shows total run time and per-image timing for batch runs.
- #6 Refined progress stage mapping to reduce misleading jumps (load/denoise/decode/save bands).

---

### Phase D: Gallery Overhaul
Includes: #7, #8, #9

Status: `DONE` (working tree changes pending commit at time of writing)

Completed in code:
- #7 Replaced grouped session-row feel with flat, true gallery image grid.
- #7 Added thumbnail size slider in gallery header.
- #8 Improved empty-state presentation and messaging for empty/search-empty states.
- #9 Added automatic infinite scroll loading using an IntersectionObserver sentinel.

Files:
- `ui/modules/GalleryManager.js`
- `ui/index.html`
- `ui/styles/theme-nova.css`
- `ui/modules/LightboxManager.js` (flat-grid metadata compatibility)

---

## Remaining Work (In Branch Scope)

### Phase E: Remaining Functional UX Tasks
Status: `TODO`

Pending items:
- #11 Move generation/progress panel away from preview overlay to lower status area near seed.
- #12 Add CLIP_SKIP2 toggle with tooltip explaining behavior.

Open quality checks still recommended:
- Re-verify #14 under repeated model-switch stress (3+ heavy model switches).
- Manual UI validation pass for gallery behavior on mobile and large libraries.

---

## Out of Scope for This Branch
As requested by user:
- #17 Windows validation/testing for WebbDuck.
- #18 GPU auto-tuning for multiple GPU classes (e.g., 3060 Ti).
- #19 Multi-family model support expansion (FLUX-dev, z-image-turbo, SD1.5, etc.).

---

## Test Notes
- Non-slow server tests passing repeatedly:
  - `python -m pytest -q -s tests/test_server.py -m "not slow"`
- Slow generation tests were previously hanging due to test fixture lifecycle, fixed by context-managed TestClient in `tests/conftest.py`.
