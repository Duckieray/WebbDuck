# OUTPAINT HANDOFF

> Historical handoff snapshot from the outpaint stabilization effort on 2026-02-10.
> This file is intentionally preserved as a point-in-time record, not as a description of the full current app state.

## 1. Mission / Context

Project: `webbduck`

Primary issue: large outpainting quality regressions (nested frame seams, repeated body structures, stripe/smear artifacts, missed final expansion behavior), while small/medium outpainting looked relatively stable.

The user explicitly asked to:
- revisit pyramid logic,
- compare against small/medium behavior,
- provide reproducible curl-based testing,
- and make large mode behave more like known-good paths.

This handoff captures all major paths attempted, code changes made, test coverage added, and current state.

---

## 2. Environment and Runtime

- Repo root: `<repo_root>/webbduck`
- Runtime env: conda env `<env_name>`
- App startup script (example): `<repo_root>/../webbduck.sh`
- Pytest location: available in the active conda env

Common validation commands used:
- `conda run -n <env_name> pytest -q --capture=no tests/test_outpaint_pyramid_regression.py`
- `conda run -n <env_name> pytest -q --capture=no tests/test_thumbnails.py tests/test_outpaint_pyramid_regression.py`
- `conda run -n <env_name> python -m py_compile ...`

---

## 3. Baseline Payloads / Repro Geometry

User provided concrete `/generate` payloads for:
- Small target: `1584 x 1840`, offsets `475,75`, pyramid disabled
- Medium target: `1728 x 2176`, offsets `583,140`, pyramid disabled
- Large target: `2464 x 3072`, offsets `1101,203`, pyramid enabled (originally)

Source box effectively used across problematic runs:
- `x=1101, y=203, w=832, h=1216`

Expansion ratios:
- Small: ~`1.90x` width, `1.51x` height
- Medium: ~`2.08x` width, `1.79x` height
- Large: ~`2.96x` width, `2.53x` height

---

## 4. What Small/Medium Actually Do (Important)

Small/medium are primarily using **repeat mode**, strategy:
- `full_canvas_then_seam`

Key characteristics:
- Start from full target canvas (`curr_full = image.convert("RGB")`) in non-auto repeat path.
- First pass does full outpaint using request `mask_image`.
- Subsequent pass(es) are seam refinement.
- No pyramid staged pass chain.

Relevant code:
- `modes/inpaint.py:936` (`strategy` logging)
- `modes/inpaint.py:954` (`curr_full = image.convert("RGB")`)
- `modes/inpaint.py:1187+` (full_canvas_then_seam loop)

---

## 5. Major Debug Runs and Findings

### Early failing/diagnostic runs (from user reports)
- `20260210_131313_772068492_seed2344759489`
- `20260210_133122_335850752_seed2344759489`
- `20260210_134450_646091755_seed2344759489`
- `20260210_142013_291132129_seed2344759489`
- `20260210_143321_215622990_seed2344759489`
- `20260210_144550_502159260_seed2344759489`
- `20260210_150052_986188000_seed2344759489`

Observed issues across these runs:
- Nested “image-in-image” seams / frame boxes at each pass boundary.
- Heavy vertical stripe blocks in lower regions in some runs.
- In some runs, apparent final-step seam cleanup instability.

### Later controlled experiment runs (with new toggles)
Newest run set and what they confirm:

- `20260210_175750_129187986_seed2929654345`
  - small, repeat full_canvas_then_seam, seed init none
- `20260210_175855_806143972_seed858200210`
  - medium, repeat full_canvas_then_seam, seed init none
- `20260210_180020_798252765_seed2344759489`
  - large, pyramid on, initializer edge_strips
- `20260210_180459_376693884_seed2344759489`
  - large, pyramid on, initializer soft_context
- `20260210_180931_594260023_seed2344759489`
  - large, pyramid off, full_canvas_then_seam, seed init none
- `20260210_181219_181750966_seed2344759489`
  - large, pyramid off, full_canvas_then_seam, seed init soft_context
- `20260210_181506_265309904_seed2344759489`
  - large, pyramid off, full_canvas_then_seam, seed init edge_strips
- `20260210_181719_205641093_seed2344759489`
  - large, pyramid off, chunked_two_stage_then_seam, seed init none
- `20260210_202336_478829470_seed2344759489`
  - large, pyramid off, full_canvas_then_seam, repeat passes forced `2`, seed init none

Conclusion from these experiments:
- Toggle plumbing is now working (initializers and path selection are reflected in `meta.json`/events).
- Quality remains unsatisfactory in multiple large variants, i.e., root quality issue not fully solved yet.

---

## 6. Paths Attempted (Chronological)

### Path A: Tune existing pyramid without structural rewrite
Tried:
- Adaptive canvas initializer (`auto` => edge_strips vs soft_context).
- Stronger stage cfg/strength defaults.
- Lock-box preservation adjustments.
- Seam-history accumulation and union seam masks.
- Minimum effective denoise step guard.

Outcome:
- Some runs improved stability but persistent nested seam artifacts remained.
- Seam-history approach likely amplified interior seam boxes.

### Path B: Rewrite pyramid pass chaining to mirror staged growth semantics
Implemented:
- New unified progressive builder for stage geometry/canvas/mask.
- Hard-mask + optional blur with explicit source protection.
- Removed historical seam union.
- Seam polish limited to current boundary.
- Later refined to final-pass-only seam by default.

Outcome:
- Structural behavior became deterministic and easier to reason about.
- Debug events became clearer (`seam_mode`, `seam_applied`, etc.).
- Quality still not at desired level for large cases.

### Path C: Make large testing path configurable like small/medium
Implemented endpoint flags and script controls for easy A/B:
- disable pyramid,
- disable chunked repeat,
- choose repeat/pyramid initializers,
- force repeat passes.

Outcome:
- Repro is now easy and reliable.
- Exposed that multiple large strategies still underperform in quality.

### Path D: Address unrelated but blocking server error
Error seen by user:
- `RuntimeError: Response content longer than Content-Length`

Likely trigger:
- Thumbnail serving concurrency/race during stream responses.

Fixes:
- thumbnail generation: per-path lock + atomic temp write + `os.replace`
- `/thumbs` response switched to in-memory `Response` bytes (non-streaming)

Outcome:
- hardened thumbnail serving path; regression test added.

---

## 7. Code Changes Made

### 7.1 `modes/outpaint.py`
Core large pyramid changes.

Key additions/changes:
- `def _build_progressive_stage(...)` at `modes/outpaint.py:350`
- `run_pyramid_outpaint(...)` at `modes/outpaint.py:399`
- default pyramid initializer now consumed via:
  - `settings.get("smart_extend_pyramid_initializer", "edge_strips")` at `modes/outpaint.py:486`
- seam timing control:
  - `seam_each_step = bool(settings.get("smart_extend_pyramid_seam_each_step", False))` at `modes/outpaint.py:479`
  - seam run decision at `modes/outpaint.py:539`
- debug event fields:
  - `seam_mode`, `seam_applied`

Notable behavior now:
- Pyramid seam denoise defaults to final pass only (`final_only`) unless explicitly overridden.

### 7.2 `modes/inpaint.py`
Non-auto repeat path and large preseed control.

Key additions/changes:
- repeat strategy debug logs now include seed initializer:
  - `repeat_seed_initializer` at `modes/inpaint.py:937`
- large-scale preseed is now optional and default off:
  - `seed_initializer = settings.get("smart_extend_repeat_seed_initializer", "none")` at `modes/inpaint.py:960`
  - preseed only if in `{edge_strips, soft_context}` at `modes/inpaint.py:961`

Important implication:
- Large repeat path can now match small/medium style by default (`none` preseed) when configured.

### 7.3 `server/app.py`
Request plumbing + thumbnail response robustness.

Added `/generate` form fields:
- `smart_extend_repeat_chunked` (`server/app.py:637`)
- `smart_extend_repeat_passes` (`server/app.py:638`)
- `smart_extend_repeat_seed_initializer` (`server/app.py:639`)
- `smart_extend_pyramid_initializer` (`server/app.py:642`)

Added to settings map:
- lines `681-686`

Queue mode-details summaries include initializer labels:
- `repeat seed ...`, `pyr init ...` around `server/app.py:175-179`

Thumbnail endpoint change:
- `/thumbs/{path:path}` now returns `Response(content=bytes, media_type="image/jpeg")` at `server/app.py:1227+`

### 7.4 `server/thumbnails.py`
Concurrency-safe thumbnail generation.

Added:
- per-path lock map and lock helper at `server/thumbnails.py:14-25`
- atomic generation in `ensure_thumbnail(...)`:
  - `tempfile.mkstemp` + `os.replace`
  - no mtime-based rewrite churn

### 7.5 Test and tooling files

#### `tests/test_outpaint_pyramid_regression.py`
Added/updated checks for:
- pass count geometry
- initializer behavior helper
- min-step guard
- stage lock preservation
- seam mode defaults
- final-only seam behavior

Key tests:
- `test_large_ratio_stage_lock_is_preserved` (`tests/test_outpaint_pyramid_regression.py:99`)
- `test_pyramid_default_seam_is_final_only` (`tests/test_outpaint_pyramid_regression.py:152`)

#### `tests/test_thumbnails.py`
New concurrency regression test:
- `test_ensure_thumbnail_is_thread_safe` (`tests/test_thumbnails.py:11`)

#### `tests/curl_test.sh`
Significant improvement for controlled experiments.

New env toggles:
- `REPEAT_PASSES`
- `REPEAT_CHUNKED`
- `REPEAT_SEED_INIT`
- `PYRAMID_INIT`
- `PYRAMID_ENABLE_OVERRIDE`
- `LARGE_SEED`

Relevant lines:
- env vars at `tests/curl_test.sh:11-16`
- form fields at `tests/curl_test.sh:92-101`
- large seed handling at `tests/curl_test.sh:129,134`

---

## 8. Validation Status

Passing tests (latest local runs):
- `tests/test_outpaint_pyramid_regression.py`
- `tests/test_thumbnails.py`

Observed result examples:
- `6 passed, 2 warnings` (warnings are pre-existing pytest/scipy warnings)

Compile checks passed for modified Python files.

---

## 9. Known Open Problems

Even with toggles and rewrites, **large-quality output remains subpar** in many variants.

Potential causes still open:
1. Large-pass denoise/cfg schedule may be under- or over-constraining semantics.
2. Seam masks may still be too broad/narrow for specific geometry, causing visible structure transitions.
3. Repeat full-canvas first pass with current large-specific defaults (`pass_guidance_scale`, `pass_strength`) may induce drift.
4. Chunked strategy may still amplify composition mismatch if intermediate semantics diverge early.
5. Strong dependence on model behavior under large masked expansion; algorithmic improvements may need tighter source anchoring or latent-space continuity constraints.

### New empirical finding (high priority)
User reported that when a **medium outpaint result is used as the input for a subsequent large outpaint**, quality improves significantly (\"mostly worked\") versus doing the full large jump directly from the original source.

Interpretation:
- A chained expansion (`small/medium -> large`) appears more stable than a single large jump.
- This suggests semantic continuity benefits from progressive context growth in separate requests, even when single-request staged logic still fails.
- This should be treated as a strong direction for the next implementation iteration.

---

## 10. Recommended Next Steps (for next engineer)

### Step 1: Prioritize chained continuation workflow
Implement and test an explicit two-request continuation path:
- Request A: generate medium outpaint from original source.
- Request B: use Request A output as input image for large outpaint.

Implementation notes:
- Add a first-class \"continuation outpaint\" mode/flag in request settings so this workflow is intentional and reproducible.
- Persist and expose continuation metadata in debug and output `meta.json`:
  - parent run id,
  - parent image path,
  - continuation step index,
  - source box remap used for second stage.
- Ensure source-box/offset handling is deterministic when continuing from prior generated outputs.

### Step 2: Pick one canonical single-request large path for focused tuning
Use exactly one path first (recommended):
- `pyramid off`
- `repeat chunked off`
- `repeat passes 2`
- `repeat seed initializer none`

Reason: closest to small/medium behavior; fewer moving parts.

### Step 3: Tune large-only first pass defaults
Investigate these large-only overrides in `modes/inpaint.py`:
- `smart_extend_large_first_strength` (currently clamp in repeat loop)
- `smart_extend_large_first_cfg`
- second pass equivalents

Likely to test:
- raise first-pass cfg (currently can drop too low)
- adjust first-pass strength away from artifact-prone regime

### Step 4: Add objective debug metrics
Current logs show mask leak stats and seam deltas, but not quality semantics.
Add:
- center-region identity drift metric (source region CLIP/cosine or feature diff)
- seam-band gradient discontinuity metric
- body-part duplication heuristics (simple detector pass) if feasible

### Step 5: Only after repeat-path stabilizes, revisit pyramid
If pyramid remains desired, constrain it to:
- final-only seam (already default)
- explicit initializer test matrix (`soft_context` vs `edge_strips`)
- maybe reduced stage count or altered growth for this geometry

### Step 6: Capture side-by-side run matrix in one artifact
Automate running 8-12 controlled variants with fixed seed and output a markdown/CSV summary of:
- strategy
- init mode
- seam mode
- direct-large vs continuation-large
- continuation parent run id
- run time
- chosen metrics

Minimum matrix should include:
- Direct large from original source
- Medium->large continuation
- For each of above: repeat full-canvas, repeat chunked, pyramid

---

## 11. Repro Commands (Current)

From repo root:

```bash
cd <repo_root>/webbduck
chmod +x tests/curl_test.sh
```

Start app:

```bash
<repo_root>/../webbduck.sh
```

Useful variant commands:

- Large, closest to small/medium style:

```bash
PYRAMID_ENABLE_OVERRIDE=false REPEAT_CHUNKED=false REPEAT_PASSES=2 REPEAT_SEED_INIT=none LARGE_SEED=2344759489 tests/curl_test.sh large
```

- Large, pyramid on soft context:

```bash
PYRAMID_ENABLE_OVERRIDE=true PYRAMID_INIT=soft_context LARGE_SEED=2344759489 tests/curl_test.sh large
```

- Large, pyramid on edge strips:

```bash
PYRAMID_ENABLE_OVERRIDE=true PYRAMID_INIT=edge_strips LARGE_SEED=2344759489 tests/curl_test.sh large
```

- Large, chunked repeat path:

```bash
PYRAMID_ENABLE_OVERRIDE=false REPEAT_CHUNKED=true REPEAT_PASSES=4 REPEAT_SEED_INIT=none LARGE_SEED=2344759489 tests/curl_test.sh large
```

---

## 12. Current Working Tree Snapshot

Modified files at handoff time:
- `modes/inpaint.py`
- `modes/outpaint.py`
- `server/app.py`
- `server/thumbnails.py`
- `tests/curl_test.sh`
- `tests/test_outpaint_pyramid_regression.py`
- `tests/test_thumbnails.py` (new)
- `OUTPAINT_HANDOFF.md` (this file)

---

## 13. Final Notes for Next Engineer

- The major win of this iteration is **control and observability**: path selection, initializer selection, and seam timing are now explicit and logged.
- The major unresolved item is **image quality at large scale**, not path ambiguity anymore.
- Focus next on one path and hyperparameter discipline rather than introducing another structural strategy.
