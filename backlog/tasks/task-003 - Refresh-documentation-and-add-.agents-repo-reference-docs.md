---
id: TASK-003
title: Refresh documentation and add .agents repo reference docs
status: Done
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

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a committed `.agents/` documentation set for repo orientation, updated `AGENTS.md` to point to it first, and tightened the standing rule that documentation must be updated in the same task when code or workflows change.

Refreshed the main contributor-facing docs to match the current codebase, clarified current UI/test expectations, and removed the stale implication that WebbDuck bundles a generic `tools/install_webbduck_plugin.py` helper.

Verification:
- Repo-aware referenced-path validation across the updated docs (`python` script run from the repo root) passed.
- Grep verification confirmed there are no remaining `Webbduck` casing mistakes in markdown docs.
- Checked remaining `tools/install_webbduck_plugin.py` reference and confirmed it is now only a clarifying note in `docs/PLUGINS.md`, not a stale bundled-script instruction.
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
