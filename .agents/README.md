# WebbDuck Agent References

This directory is the committed reference set for agents working in WebbDuck.
Use it with `AGENTS.md` and the main docs in `docs/`, `ui/`, `tests/`, and `plugins/`.

## Read Order

1. `AGENTS.md`
2. `.agents/repo-overview.md`
3. `.agents/backend-runtime.md`
4. `.agents/frontend.md`
5. `.agents/plugins-tests-docs.md`
6. `docs/ARCHITECTURE.md`
7. `docs/DEVELOPMENT.md`

## What Lives Here

- `repo-overview.md`: runtime flow, top-level ownership, and common edit starting points.
- `backend-runtime.md`: `server/`, `core/`, `models/`, `modes/`, and `prompt/` responsibilities.
- `frontend.md`: `ui/index.html`, `ui/app.js`, `ui/core/`, `ui/modules/`, and `ui/styles/`.
- `plugins-tests-docs.md`: plugin contracts, test map, verification guidance, and documentation upkeep rules.

## Doc Maintenance Rule

Agents should treat these files as part of the source of truth.
When code changes behavior, routes, commands, structure, workflows, or contracts, update the affected docs in the same task.
