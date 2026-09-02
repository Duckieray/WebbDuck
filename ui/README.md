# WebbDuck UI Guide

WebbDuck's frontend is a zero-build web app built with plain HTML, ES modules, and CSS.

## Folder Map

```text
ui/
|- index.html
|- app.js
|- app_main.js
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
|  |- ProgressManager.js
|  `- ProviderCredentialsSettings.js
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
- `ui/app.js`: stable browser composition entrypoint; imports the main app plus small host-level extensions.
- `ui/app_main.js`: Studio, Queue, Gallery, Settings, help modal, and plugin-tab orchestration.
- `ui/modules/ProviderCredentialsSettings.js`: optional Hugging Face/Civitai credential controls injected into the existing Settings modal. Token values are never persisted in browser state.
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
- `ui/modules/PersonaIdentityUI.js`: Identity / Persona presentation over the shared reference + preset manager, including the FLUX.2-only persona tuning controls (reference face crop, anchor boost, face-focus framing).

## Realtime Flow

WebSocket events from `/ws` are translated into local frontend events:

- `state` -> `Events.STATUS_UPDATE`
- `queue` -> `Events.QUEUE_UPDATE`
- `catalog` -> `Events.CATALOG_UPDATE`

This keeps queue and catalog state fresh without extra polling loops in the browser.

## Persisted State

`ui/core/state.js` stores Studio settings in `localStorage` under `webbduck_state_v2`.

Persisted state includes prompt fields, dimensions, scheduler, second-pass options, img2img settings, smart-extend options, selected LoRAs, and selected embeddings. Identity persona state adds reference selection plus the FLUX.2 persona tuning keys (`face_crop`, `flux2_anchor_dup`, `face_focus`).

Provider credentials are deliberately excluded from this state. They are stored by the WebbDuck host under `~/.webbduck/provider_credentials.json`; the browser receives configuration status only and never reads a saved token back.

## Editing Recipes

### Add a new control

1. Add the markup in `ui/index.html`, or inject a tightly scoped host-level Settings extension from `ui/modules/` when the control should remain isolated.
2. Read and write ordinary Studio values in `ui/app_main.js`.
3. Persist it in `ui/core/state.js` if it should survive reloads and is not secret.
4. Send it through `ui/core/api.js` or a focused feature module if the backend needs it.
5. Update related docs and tests.

### Add a new screen-level behavior

- Keep the stable composition entry in `ui/app.js` small.
- Put page orchestration in `ui/app_main.js`.
- Move feature-specific logic into a module in `ui/modules/` when the behavior grows beyond a small helper.

### Change styles

- Update `ui/styles/design-tokens.css` for reusable colors, spacing, or typography tokens.
- Update `ui/styles/theme-nova.css` for WebbDuck-specific theming and component presentation.
- Update files in `ui/styles/components/` or `ui/styles/layouts/` for reusable CSS organization.
- `ui/planning/` is available for tracked planning notes, but it is currently empty.

### Change API contracts

- Keep the relevant UI module/API helper and server router aligned.
- If the change affects persisted non-secret inputs, update `ui/core/state.js` too.

## Current UI Features

- Studio and Gallery views.
- Queue modal with queued and running job controls.
- Prompt token counter and warning states.
- Resolution presets and custom size support.
- Seed randomization.
- LoRA and embedding management.
- Inpaint mask editor and smart extend controls.
- Lightbox metadata panel and image actions.
- Settings modal, optional model-provider credentials, in-app help modal, and remote plugin connection UI.
- Queue detail cards, compare view, and plugin iframe tabs.

## Verification

When you touch the frontend, prefer a focused check first:

```bash
pytest tests/test_ui_sanity.py -v
pytest tests/test_server.py -v
```

`tests/test_ui_sanity.py` expects a running server and browser automation support. If you changed data contracts, add the relevant backend or integration tests too.
