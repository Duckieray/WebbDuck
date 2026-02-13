# Pyramid Outpainting Implementation Guide (Project-Aligned)

This guide is tailored to this repository's current codebase and avoids stale recommendations.

## Current Reality In This Repo

- Smart extend/outpaint execution lives in `modes/inpaint.py`.
- Progress estimation already exists in `core/generation.py` via `estimate_total_steps`.
- Large non-auto outpaint already has a two-stage path (`chunked_two_stage_then_seam`) in `modes/inpaint.py`.
- Debug sessions already exist (`outputs/_debug/...`) via:
  - `WEBBDUCK_SMART_EXTEND_DEBUG`
  - `WEBBDUCK_SMART_EXTEND_DEBUG_DIR`
- Upscaling is already implemented with Real-ESRGAN in:
  - `models/upscaler.py`
  - `core/worker.py` (`/upscale` job path)

## What Was Incorrect In The Old Guide

- Suggested adding RealESRGAN as a "future improvement" (already implemented here).
- Used non-repo paths like `/home/claude/...` and `webbduck/modes/inpaint.py`.
- Recommended global defaults (`strength`, `cfg`) that would affect all inpainting, not just outpaint paths.
- Assumed a missing progress model even though `core/generation.py` already estimates staged passes.

## Goal

Add an optional **pyramid outpaint path** for very large expansions while preserving current behavior for small/medium jobs.

## Recommended Integration Strategy

Use pyramid only for large expansions, keep existing logic as default otherwise.

### 1. Use Existing Module Location

- Keep pyramid logic in `modes/outpaint.py` (already present in this repo).
- Import from `modes/inpaint.py` only when needed:

```python
from webbduck.modes import outpaint as pyramid_outpaint
```

Do not create duplicate `core/outpaint.py` unless you intentionally refactor.

### 2. Add A Strict Feature Flag + Eligibility Gate

In `modes/inpaint.py`, before entering non-auto repeat logic, gate pyramid with both:

- Feature flag: `smart_extend_pyramid_enable` (default `False` initially)
- Eligibility:
  - `smart_extend == True`
  - `smart_extend_auto_step == False`
  - expansion ratio >= configurable threshold (e.g. `2.4`)

This avoids silently changing behavior for all outpaints.

### 3. Scope Hyperparameters To Pyramid Path Only

Do **not** change top-level defaults like:

- `strength = settings.get("strength", ...)`
- `guidance_scale = settings.get("cfg", ...)`

Instead, define pyramid-local values:

- `smart_extend_pyramid_cfg_stage1` (e.g. `7.0-8.0`)
- `smart_extend_pyramid_cfg_stage2` (e.g. `8.5-10.0`)
- `smart_extend_pyramid_strength_stage1` (e.g. `0.50-0.65`)
- `smart_extend_pyramid_strength_stage2` (e.g. `0.50-0.65`)
- `smart_extend_pyramid_refine_strength` (e.g. `0.20-0.45`)

This keeps base inpainting behavior stable.

### 4. Keep Existing Repeat Path As Fallback

If pyramid is disabled or fails, use existing logic in `modes/inpaint.py`:

- `full_canvas_then_seam`
- `chunked_two_stage_then_seam`

Do not remove current paths until pyramid is proven across seeds and sizes.

### 5. Add Debug Events Specific To Pyramid

Use existing `_debug_log_event(...)` and `_debug_save_image(...)` helpers in `modes/inpaint.py`.

Add event types like:

- `pyramid_selected`
- `pyramid_stage1_complete`
- `pyramid_stage2_complete`
- `pyramid_refine_complete`

Record:

- stage sizes
- per-stage cfg/strength
- seed offsets
- mask leak stats

This keeps analysis workflow consistent with current debug tooling.

### 6. Update Step Estimation (Only If Pyramid Enabled)

`core/generation.py` already estimates smart-extend steps.

Add a pyramid-specific branch in `estimate_total_steps(settings)` only when:

- `smart_extend_pyramid_enable == True`
- and pyramid eligibility criteria are met

Estimate total as:

- stage1 steps
- stage2 steps
- optional seam/refine steps

Leave non-pyramid estimates untouched.

### 7. UI/Request Plumbing (Optional, Minimal First)

Initial rollout can be backend-only via hidden settings in request payload.

If exposing controls later, use `ui/app.js` with simple fields:

- `smart_extend_pyramid_enable`
- `smart_extend_pyramid_trigger_ratio`
- `smart_extend_pyramid_scale`

Do not add many controls before behavior is stable.

## Suggested Initial Defaults (Conservative)

Use these as starting points for large jobs only:

- `smart_extend_pyramid_enable = false` (turn on explicitly while testing)
- `smart_extend_pyramid_trigger_ratio = 2.4`
- `smart_extend_pyramid_scale = 0.60`
- `smart_extend_pyramid_cfg_stage1 = 7.5`
- `smart_extend_pyramid_cfg_stage2 = 9.0`
- `smart_extend_pyramid_strength_stage1 = 0.55`
- `smart_extend_pyramid_strength_stage2 = 0.58`
- `smart_extend_pyramid_refine_strength = 0.35`

## Test Plan For This Repo

Use the same three-size regression pattern you are already using:

1. Small: ~`1.8x` scale
2. Medium: ~`2.0x` scale
3. Large: ~`2.8x-3.0x` scale

For each run, compare:

- `meta.json` strategy/event sequence
- pass `02_raw.png` vs `03_composited.png`
- seam delta events

Success criteria:

- No recursive "image-in-image" duplication in large runs.
- No regression in small/medium quality.
- Reasonable step estimate and progress behavior.

## Notes On Upscaling

This project already supports Real-ESRGAN upscaling through `/upscale`.

For pyramid outpaint integration:

- Keep generation-stage resizing inside pipeline logic (PIL/Lanczos is fine for internal stage transitions).
- Treat Real-ESRGAN as a separate user-facing upscale feature, unless you intentionally wire ESRGAN into pyramid stages later.

## Implementation Order (Practical)

1. Wire feature flag + eligibility gate in `modes/inpaint.py`.
2. Call `modes/outpaint.py` only when gate passes.
3. Add pyramid debug events/images.
4. Add optional step estimation branch in `core/generation.py`.
5. Validate against small/medium/large seed set.
6. Only then expose UI controls.

## Keep / Remove Guidance

Keep:

- existing smart-extend debug system
- existing fallback repeat paths
- existing progress framework

Remove or avoid:

- global default cfg/strength changes for all inpaint
- duplicate upscaler integration work
- path instructions that reference external folders

