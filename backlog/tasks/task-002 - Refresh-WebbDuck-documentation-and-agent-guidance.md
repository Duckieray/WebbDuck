---
id: TASK-002
title: Refresh WebbDuck documentation and agent guidance
status: Done
assignee:
  - OpenCode
created_date: '2026-03-26 17:22'
updated_date: '2026-03-26 17:30'
labels: []
dependencies: []
references:
  - README.md
  - AGENTS.md
  - docs/DEVELOPMENT.md
  - docs/USER_GUIDE.md
  - docs/PLUGINS.md
  - docs/WINDOWS_TESTING.md
  - docs/SIMPLE_GUIDE.md
  - tests/README.md
  - ui/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the tracked documentation so contributors and agents can quickly locate major subsystems, understand what each area does, and follow current workflows for extending or updating WebbDuck. Remove or replace stale docs and align cross-references across the main guides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Core contributor-facing docs explain the repo layout, subsystem responsibilities, and where to make common types of changes.
- [x] #2 Agent guidance and developer docs point to the current source-of-truth docs and workflows for updating backend, frontend, tests, and plugins.
- [x] #3 Stale or redundant docs are removed or replaced, and remaining docs have consistent cross-links and terminology.
- [x] #4 Documentation changes are verified with targeted checks for obvious broken references or mismatched commands.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the current tracked docs (`README.md`, `AGENTS.md`, `docs/*`, `ui/README.md`, `tests/README.md`, `plugins/README.md`) against the live repo layout and commands so stale references are identified before editing.
2. Consolidate contributor-facing guidance around a clearer source-of-truth set: refresh the top-level README, expand the development/agent docs with an accurate repo map and common change recipes, and decide whether `docs/SIMPLE_GUIDE.md` should be kept, replaced, or folded into another guide.
3. Update specialized docs (`docs/USER_GUIDE.md`, `docs/PLUGINS.md`, `docs/WINDOWS_TESTING.md`, `ui/README.md`, `tests/README.md`, `plugins/README.md`) so each explains responsibilities, extension points, and how to update the corresponding area.
4. Remove or replace stale documentation where needed, then run targeted verification focused on broken markdown links, command references, and obvious mismatches with the current filesystem layout.
5. Record the verification results and completion summary in the task once the documentation set is internally consistent.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audited the tracked documentation against the live repo layout and current entrypoints, then introduced a new `docs/ARCHITECTURE.md` repo map so contributors and agents have a single place to find file ownership and common change recipes.

Refreshed the main contributor-facing docs (`README.md`, `AGENTS.md`, `docs/DEVELOPMENT.md`, `ui/README.md`, `tests/README.md`, `docs/PLUGINS.md`, `plugins/README.md`) and rewrote the user-facing guides to remove stale wording while keeping `docs/SIMPLE_GUIDE.md` because it is still served by `/docs/simple-guide` in the UI.

Verification: ran a Python path-existence check over the documented key files and a markdown-link validation pass across the refreshed docs; both completed successfully. Project-level Definition of Done defaults are feature-oriented and not all applicable to this documentation-only task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the WebbDuck documentation set so contributors and agents can quickly locate major subsystems, understand file ownership, and follow current update workflows. Added `docs/ARCHITECTURE.md` as the new repo map, modernized the top-level README and development/agent guidance, and aligned the frontend, tests, plugins, Windows, and user guides around the current repo structure and behaviors.

Verification:
- `python` path-existence check covering the documented key repo files and directories
- `python` markdown-link validation across the refreshed documentation files

Notes:
- Kept `docs/SIMPLE_GUIDE.md` and refreshed it instead of removing it because the app still serves it from `/docs/simple-guide` for in-app help.
- Did not change unrelated working-tree files already modified outside this task (`models/registry.py`, `models/upscaler.py`, `run.py`, `startup.sh`).
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Feature works in real Diffusers pipeline (not mock only)
- [ ] #2 No frontend console errors and no backend tracebacks
- [ ] #3 VRAM usage validated and no memory leaks introduced
- [ ] #4 LoRA loading and scaling verified (if affected)
- [ ] #5 Seed behavior verified (deterministic when expected)
- [ ] #6 UI state accurately reflects backend configuration
- [ ] #7 README or documentation updated if behavior changed
- [ ] #8 Committed with descriptive message
- [ ] #9 Generation speed benchmarked against previous version
<!-- DOD:END -->
