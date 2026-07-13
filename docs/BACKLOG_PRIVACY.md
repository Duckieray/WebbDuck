# Backlog Public vs Private Notes

Use this convention to keep collaboration data public while keeping personal context local.

## Public backlog (`backlog/`)

Store only information that is safe and useful for collaborators:

- task title, description, acceptance criteria, status
- implementation notes that should be shared
- links to repo files, docs, and issues

Do not include:

- secrets, tokens, credentials, private URLs
- machine-specific local paths
- personal-only notes not needed by reviewers

## Private notes (`backlog-private/`)

Store personal or machine-specific details here.

- folder is ignored by Git
- suggested per-task file name: `TASK-<id>.local.md`
- keep sensitive details out of commit messages and PR descriptions

## Working style

1. Track commitments in `backlog/`.
2. Keep personal execution details in `backlog-private/`.
3. Before commit/push, verify no private details leaked into tracked files.
