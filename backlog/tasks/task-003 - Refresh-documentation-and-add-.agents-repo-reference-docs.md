---
id: TASK-003
title: Refresh documentation and add .agents repo reference docs
status: To Do
assignee:
  - '@OpenCode'
created_date: '2026-04-03 17:46'
labels:
  - documentation
  - agents
milestone: m-0
dependencies: []
references:
  - AGENTS.md
  - README.md
  - docs/ARCHITECTURE.md
  - docs/DEVELOPMENT.md
  - docs/PLUGINS.md
  - docs/USER_GUIDE.md
  - docs/SIMPLE_GUIDE.md
  - docs/WINDOWS_TESTING.md
  - ui/README.md
  - tests/README.md
  - plugins/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the tracked documentation so it matches the current WebbDuck codebase, add a committed `.agents/` documentation set that gives agents complete repo orientation, and update AGENTS.md so agents are directed to keep docs current whenever code or workflows change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tracked documentation reflects the current repo structure, runtime entrypoints, and documented commands.
- [ ] #2 AGENTS.md points to the new `.agents/` reference docs and explicitly instructs agents to update documentation when behavior or structure changes.
- [ ] #3 A committed `.agents/` documentation set exists and provides full-repo orientation covering backend, frontend, plugins, tests, and key workflows.
- [ ] #4 Cross-links between AGENTS.md, `.agents/`, and the main docs are internally consistent and do not point to stale files or behaviors.
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
