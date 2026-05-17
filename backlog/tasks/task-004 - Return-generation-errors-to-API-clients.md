---
id: TASK-004
title: Return generation errors to API clients
status: In Progress
assignee:
  - OpenCode
created_date: '2026-05-17 21:38'
updated_date: '2026-05-17 21:42'
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
- [x] #2 When a queued API generation job fails the API client can retrieve the failure status and error details from the API response path used for third-party generation workflows.
- [ ] #3 Existing successful generation responses remain unchanged.
- [ ] #4 Focused tests cover the error response behavior for third-party API clients.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the generation enqueue/worker failure path to find where API responses lose error details.
2. Return normalized JSON error payloads for waited generation requests when the queued job future fails.
3. Add a per-job queue status endpoint so third-party async clients can poll by job_id and read failed status/error details.
4. Add focused server tests for both synchronous failure responses and async failed-job lookup.
5. Run targeted verification and record any environment limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented JSON failure responses from queued generation/upscale/test endpoints by catching failed futures in enqueue and returning status/job_id/error payloads.

Added GET /queue/{job_id} so async API clients can poll a specific job and retrieve failed status plus stored error details.

Added focused tests for synchronous worker failure reporting and failed-job lookup; py_compile passed, but focused pytest via conda could not run because the local 'webbduck' conda environment was unavailable.
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
