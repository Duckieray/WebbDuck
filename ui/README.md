# WebbDuck UI Guide

WebbDuck's frontend is a zero-build web app built with plain HTML, ES modules, and CSS.

## Folder Map

```text
ui/
|- index.html
|- app.js
|- core/
|  |- api.js
|  |- events.js
|  |- state.js
|  `- utils.js
|- modules/
|  |- EmbeddingManager.js
|  |- GalleryManager.js
|  |- LightboxManager.js
|  |- LoraManager.js
|  |- MaskEditor.js
|  `- ProgressManager.js
|- styles/
|  |- reset.css
|  |- base.css
|  |- design-tokens.css
|  |- main.css
|  |- theme-nova.css
|  |- components/
|  `- layouts/
`- planning/
```

## File Responsibilities

- `ui/index.html`: app shell markup, controls, modals, and mount points.
- `ui/app.js`: top-level wiring for Studio, Queue, Gallery, Settings, help modal, and plugin tabs.
- `ui/core/api.js`: fetch helpers for backend endpoints.
- `ui/core/events.js`: local event bus fed by the WebSocket stream.
- `ui/core/state.js`: persisted Studio state and DOM sync.
- `ui/core/utils.js`: DOM, toast, async, and form-data helpers used across the app.
- `ui/modules/GalleryManager.js`: gallery paging, search, filters, and thumbnail sizing.
- `ui/modules/LightboxManager.js`: full-screen viewer and metadata display.
- `ui/modules/LoraManager.js`: LoRA list, weights, and compatibility filtering.
- `ui/modules/EmbeddingManager.js`: embedding selection and token editing.
- `ui/modules/MaskEditor.js`: inpaint mask interactions.
- `ui/modules/ProgressManager.js`: progress card and cancel handling.

## Realtime Flow

WebSocket events from `/ws` are translated into local frontend events:

- `state` -> `Events.STATUS_UPDATE`
- `queue` -> `Events.QUEUE_UPDATE`
- `catalog` -> `Events.CATALOG_UPDATE`

This keeps queue and catalog state fresh without extra polling loops in the browser.

## Persisted State

`ui/core/state.js` stores Studio settings in `localStorage` under `webbduck_state_v2`.

Persisted state includes prompt fields, dimensions, scheduler, second-pass options, img2img settings, smart-extend options, selected LoRAs, and selected embeddings.

## Editing Recipes

### Add a new control

1. Add the markup in `ui/index.html`.
2. Read and write the value in `ui/app.js`.
3. Persist it in `ui/core/state.js` if it should survive reloads.
4. Send it through `ui/core/api.js` if the backend needs it.
5. Update related docs and tests.

### Add a new screen-level behavior

- Put page orchestration in `ui/app.js`.
- Move feature-specific logic into a module in `ui/modules/` when the behavior grows beyond a small helper.

### Change styles

- Update `ui/styles/design-tokens.css` for reusable colors, spacing, or typography tokens.
- Update `ui/styles/theme-nova.css` for WebbDuck-specific theming and component presentation.
- Update files in `ui/styles/components/` or `ui/styles/layouts/` for reusable CSS organization.
- `ui/planning/` is available for tracked planning notes, but it is currently empty.

### Change API contracts

- Keep `ui/core/api.js`, `ui/app.js`, and `server/app.py` aligned.
- If the change affects persisted inputs, update `ui/core/state.js` too.

## Current UI Features

- Studio and Gallery views.
- Queue modal with queued and running job controls.
- Prompt token counter and warning states.
- Resolution presets and custom size support.
- Seed randomization.
- LoRA and embedding management.
- Inpaint mask editor and smart extend controls.
- Lightbox metadata panel and image actions.
- Settings modal, in-app help modal, and remote plugin connection UI.
- Queue detail cards, compare view, and plugin iframe tabs.

## Verification

When you touch the frontend, prefer a focused check first:

```bash
pytest tests/test_ui_sanity.py -v
pytest tests/test_server.py -v
```

`tests/test_ui_sanity.py` expects a running server and browser automation support. If you changed data contracts, add the relevant backend or integration tests too.
