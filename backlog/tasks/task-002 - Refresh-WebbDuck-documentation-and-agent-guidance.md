---
id: TASK-002
title: Refresh WebbDuck documentation and agent guidance
status: To Do
assignee: []
created_date: '2026-03-26 17:22'
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
