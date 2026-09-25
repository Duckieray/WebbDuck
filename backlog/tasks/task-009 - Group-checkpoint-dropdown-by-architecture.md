---
id: TASK-009
title: Group checkpoint dropdown by architecture
status: Done
assignee: []
created_date: '2026-09-25 00:00'
updated_date: '2026-09-25 00:00'
labels:
  - ui
  - models
dependencies: []
references:
  - models/model_descriptor.py
  - ui/core/api.js
  - ui/core/utils.js
  - ui/styles/theme-nova.css
  - ui/README.md
  - docs/ARCHITECTURE.md
documentation:
  - docs/ARCHITECTURE.md
  - ui/README.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Checkpoints dropdown listed every catalog entry in one flat alphabetical
run, so an installation with SDXL, FLUX.1, FLUX.2, Krea 2, and Qwen Image
checkpoints mixed them together with no way to tell them apart. Group the
options under an `<optgroup>` per architecture so the families are visually
separable, while keeping a single model selector.

`ModelDescriptor.to_dict()` now exposes a display-only `arch` hint.
`normalizeCatalog()` in `ui/core/api.js` maps it to a human label, sorts
group-first by a fixed architecture order, and tags each option with `group`.
`populateSelect()` in `ui/core/utils.js` renders grouped options as
`<optgroup>`s and still accepts plain strings and ungrouped objects, so no
other caller changes behavior.

This is presentation only. It does not introduce an architecture/engine
selector and does not change routing, which stays model-first per
`docs/ARCHITECTURE_AGNOSTIC_GENERATION.md` (see also TASK-005, which
explicitly rules out solving that task with architecture tabs or an engine
dropdown).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `ModelDescriptor.to_dict()` exposes a display-only `arch` field.
- [x] #2 `normalizeCatalog()` assigns a display label, sorts group-first by a fixed architecture order, then alphabetically within each group, and tags each option with `group`.
- [x] #3 `populateSelect()` renders grouped options as `<optgroup>`s and is backward compatible with plain strings and ungrouped objects.
- [x] #4 Styling for `select optgroup` matches the active theme.
- [x] #5 Docs updated for the new catalog `arch` hint and the grouping behavior.
- [x] #6 Targeted tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `models/model_descriptor.py`: add `"arch": self.architecture` to `to_dict()`.
2. `ui/core/api.js`: add `ARCH_ORDER`/`ARCH_LABELS`, an `architectureGroup()` mapper, and group-first sorting in `normalizeCatalog()`.
3. `ui/core/utils.js`: make `populateSelect()` emit `<optgroup>` for options carrying `group`.
4. `ui/styles/theme-nova.css`: style `select optgroup`.
5. Update `docs/ARCHITECTURE.md` and `ui/README.md`.
<!-- SECTION:PLAN:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] #1 The dropdown renders one `<optgroup>` per architecture, in a fixed order.
- [x] #2 No behavior change for callers passing ungrouped options.
- [x] #3 Targeted tests pass
- [x] #4 Docs updated per project conventions
<!-- DOD:BEGIN -->

## Verification Notes

<!-- SECTION:VERIFICATION:BEGIN -->
Ran and passing:
- `pytest tests/test_model_descriptor.py` - covers the `arch` payload field
- `node --check` on `ui/core/api.js` and `ui/core/utils.js`

The `<optgroup>` rendering itself was not exercised in a browser: the full app
page crashes headless Chromium in this environment on a font-stack defect
(`FATAL: SkFontMgr_FontConfigInterface.cpp: Not implemented`), reproducible with
JS disabled and all subresources blocked. Please eyeball the dropdown once in a
real browser.
<!-- SECTION:VERIFICATION:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
This work was found already implemented in the working tree with no tracking
task, and was committed as-is after a targeted test pass. Recorded here so the
history stays traceable per AGENTS.md section 5.2.
<!-- SECTION:NOTES:END -->
