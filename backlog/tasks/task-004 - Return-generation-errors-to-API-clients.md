---
id: TASK-004
title: Return generation errors to API clients
status: In Progress
assignee: []
created_date: '2026-05-17 21:38'
labels:
  - api
  - error-handling
milestone: API Reliability
dependencies: []
references:
  - server/app.py
  - core/worker.py
  - tests/test_server.py
documentation:
  - docs/ARCHITECTURE.md
  - docs/DEVELOPMENT.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure third-party clients calling the generation API receive actionable error details when a generation request fails, instead of only seeing a generic or silent failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When an API generation request fails before work is queued the HTTP response includes a clear error message.
- [ ] #2 When a queued API generation job fails the API client can retrieve the failure status and error details from the API response path used for third-party generation workflows.
- [ ] #3 Existing successful generation responses remain unchanged.
- [ ] #4 Focused tests cover the error response behavior for third-party API clients.
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
