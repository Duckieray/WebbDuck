---
id: TASK-001
title: Add pre-push branch guard hook
status: Backlog
assignee: []
created_date: '2026-02-27 23:16'
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
- [ ] #1 `.git/hooks/pre-push` exists with the specified script content.
- [ ] #2 The hook is executable and runs on `git push`.
- [ ] #3 Push attempts from `main` are blocked with the configured message and non-zero exit.
- [ ] #4 Push attempts from branches not matching `feature/*` are blocked with the configured message and non-zero exit.
- [ ] #5 Push attempts from branches matching `feature/*` are allowed to proceed.
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
