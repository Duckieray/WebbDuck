---
id: TASK-001
title: Add pre-push branch guard hook
status: Done
assignee: []
created_date: '2026-02-27 23:16'
updated_date: '2026-02-27 23:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a local Git hook at `.git/hooks/pre-push` with this script content:

```bash
#!/bin/bash

branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "$branch" == "main" ]]; then
  echo "❌ Direct push to main is blocked."
  exit 1
fi

if [[ ! "$branch" =~ ^feature/ ]]; then
  echo "❌ Only feature/* branches may be pushed."
  exit 1
fi

exit 0
```
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `.git/hooks/pre-push` exists with the specified script content.
- [x] #2 The hook is executable and runs on `git push`.
- [x] #3 Push attempts from `main` are blocked with the configured message and non-zero exit.
- [x] #4 Push attempts from branches not matching `feature/*` are blocked with the configured message and non-zero exit.
- [x] #5 Push attempts from branches matching `feature/*` are allowed to proceed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented `.git/hooks/pre-push` with the requested script content.

Set executable permission on the hook file.

Validated behavior with mocked branch values: `main` blocked (exit 1), `dev/test` blocked (exit 1), and `feature/demo` allowed (exit 0).
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
