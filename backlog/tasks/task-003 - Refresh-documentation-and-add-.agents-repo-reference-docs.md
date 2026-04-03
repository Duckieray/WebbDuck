---
id: TASK-003
title: Refresh documentation and add .agents repo reference docs
status: In Progress
assignee:
  - '@OpenCode'
created_date: '2026-04-03 17:46'
updated_date: '2026-04-03 17:53'
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
  - .agents/README.md
  - .agents/repo-overview.md
  - .agents/backend-runtime.md
  - .agents/frontend.md
  - .agents/plugins-tests-docs.md
  - plugins/captioners/joycaption/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the tracked documentation so it matches the current WebbDuck codebase, add a committed `.agents/` documentation set that gives agents complete repo orientation, and update AGENTS.md so agents are directed to keep docs current whenever code or workflows change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tracked documentation reflects the current repo structure, runtime entrypoints, and documented commands.
- [x] #2 AGENTS.md points to the new `.agents/` reference docs and explicitly instructs agents to update documentation when behavior or structure changes.
- [x] #3 A committed `.agents/` documentation set exists and provides full-repo orientation covering backend, frontend, plugins, tests, and key workflows.
- [x] #4 Cross-links between AGENTS.md, `.agents/`, and the main docs are internally consistent and do not point to stale files or behaviors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare current docs against the live codebase and note stale references or missing subsystems.
2. Add `.agents/` reference docs covering repo overview, backend/runtime, frontend, plugins, and tests/docs workflows.
3. Update `AGENTS.md` to point to `.agents/` and require docs to be kept current.
4. Refresh the main tracked docs where they drift from current code or commands.
5. Run focused verification for markdown links and referenced paths, then record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a committed `.agents/` reference set with repo overview, backend/runtime, frontend, and plugins/tests/docs guidance so agents have a stable in-repo orientation path.

Updated `AGENTS.md` to point to the `.agents/` docs first, expanded the source-of-truth doc list, and made documentation updates a same-task requirement when behavior, routes, commands, or structure change.

Refreshed contributor-facing docs to match the current codebase and removed the stale implication that WebbDuck bundles `tools/install_webbduck_plugin.py`.

Updated test and UI docs to reflect current structure and expectations, including the browser/runtime requirements for `tests/test_ui_sanity.py` and the role of `ui/core/utils.js`.
<!-- SECTION:NOTES:END -->

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
