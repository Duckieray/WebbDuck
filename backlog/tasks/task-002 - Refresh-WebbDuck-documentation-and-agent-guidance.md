---
id: TASK-002
title: Refresh WebbDuck documentation and agent guidance
status: In Progress
assignee:
  - OpenCode
created_date: '2026-03-26 17:22'
updated_date: '2026-03-26 17:23'
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
- [ ] #1 Core contributor-facing docs explain the repo layout, subsystem responsibilities, and where to make common types of changes.
- [ ] #2 Agent guidance and developer docs point to the current source-of-truth docs and workflows for updating backend, frontend, tests, and plugins.
- [ ] #3 Stale or redundant docs are removed or replaced, and remaining docs have consistent cross-links and terminology.
- [ ] #4 Documentation changes are verified with targeted checks for obvious broken references or mismatched commands.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the current tracked docs (`README.md`, `AGENTS.md`, `docs/*`, `ui/README.md`, `tests/README.md`, `plugins/README.md`) against the live repo layout and commands so stale references are identified before editing.
2. Consolidate contributor-facing guidance around a clearer source-of-truth set: refresh the top-level README, expand the development/agent docs with an accurate repo map and common change recipes, and decide whether `docs/SIMPLE_GUIDE.md` should be kept, replaced, or folded into another guide.
3. Update specialized docs (`docs/USER_GUIDE.md`, `docs/PLUGINS.md`, `docs/WINDOWS_TESTING.md`, `ui/README.md`, `tests/README.md`, `plugins/README.md`) so each explains responsibilities, extension points, and how to update the corresponding area.
4. Remove or replace stale documentation where needed, then run targeted verification focused on broken markdown links, command references, and obvious mismatches with the current filesystem layout.
5. Record the verification results and completion summary in the task once the documentation set is internally consistent.
<!-- SECTION:PLAN:END -->

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
