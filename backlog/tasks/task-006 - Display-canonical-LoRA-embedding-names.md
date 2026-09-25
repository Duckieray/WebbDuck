---
id: TASK-006
title: Display canonical LoRA/embedding names in pickers
status: Backlog
assignee: []
created_date: '2026-09-18 00:00'
updated_date: '2026-09-18 00:00'
labels:
  - ui
  - metadata
  - models
dependencies: []
references:
  - models/metastore.py
  - models/registry.py
  - server/app.py
  - server/model_catalog_api.py
  - ui/modules/LoraManager.js
  - ui/modules/EmbeddingManager.js
documentation:
  - .agents/backend-runtime.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LoRA and embedding pickers currently display the raw safetensors filename stem. Add a canonical display `title` to the rich-metadata store (`_LoraMeta`, `_EmbeddingMeta`, `AssetInfo`) and surface it through the LoRA/embedding listing endpoints. The UI pickers should show `title` when present while keeping the file-stem `name` as the identity used in generation payloads and persisted state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Metastore schema carries an optional `title` for LoRAs and embeddings, round-trips through parse/save, and appears in `AssetInfo` and API responses.
- [ ] #2 `/models/{base_model}/loras`, `/models/{base_model}/embeddings`, and `/model-catalog/{model}/loras` include a `title` field.
- [ ] #3 LoraManager and EmbeddingManager display `title` (fallback to `name`) while keeping `name` as the identity/payload value.
- [ ] #4 A backfill tool derives titles from CivitAI model pages (via existing metastore `url`) with safetensors metadata (`modelspec.title`/`name`) as offline fallback.
- [ ] #5 Search matches canonical titles; existing behavior (identity, weights, tokens, saved state) is unchanged.
- [ ] #6 Targeted tests cover schema round-trip, endpoint title surfacing, and UI fallback.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `title` to `_LoraMeta`, `_EmbeddingMeta`, and `AssetInfo`; wire parsing, saving, `update_*_meta`, merge helpers, and fuzzy search.
2. Merge `title` into `list_model_loras` / `list_model_embeddings` (app.py) and the catalog loras endpoint (including the Krea namespace path).
3. Update LoraManager/EmbeddingManager to display `title || name` in options and cards while preserving name identity.
4. Add `tools/backfill_canonical_titles.py` and run it over existing assets.
5. Add tests and run focused regression coverage.
6. Update `.agents/backend-runtime.md` notes about the metadata store.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Real pickers show canonical names; generation and saved state still keyed on file-stem name
- [ ] #2 API responses include `title` with `name` unchanged
- [ ] #3 Backfill populated titles without errors and is re-runnable
- [ ] #4 Targeted tests pass
- [ ] #5 Docs updated per project conventions
<!-- DOD:END -->