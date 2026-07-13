# Frontend Reference

WebbDuck's frontend is a zero-build web app built from plain HTML, ES modules, and CSS.

## Main Files

- `ui/index.html`: app shell, Studio controls, Gallery, modals, lightbox mounts, and plugin view containers.
- `ui/app.js`: top-level orchestration for generation, queue UI, settings, remote plugins, smart extend, uploads, preview state, and gallery/lightbox interactions.

## Core Modules

- `ui/core/api.js`: fetch wrappers for server routes.
- `ui/core/events.js`: local event bus and WebSocket client for `/ws`.
- `ui/core/state.js`: persisted Studio state in `localStorage` under `webbduck_state_v2`.
- `ui/core/utils.js`: DOM helpers, form-data helpers, throttling/debouncing, downloads, and toast UI.

## Feature Modules

- `ui/modules/ProgressManager.js`: progress ring, status bar, and VRAM display.
- `ui/modules/MaskEditor.js`: full-screen inpaint mask editor.
- `ui/modules/LoraManager.js`: LoRA loading, selection, weights, and restore-from-state behavior.
- `ui/modules/EmbeddingManager.js`: embedding loading, selection, token editing, and restore-from-state behavior.
- `ui/modules/GalleryManager.js`: gallery paging, search, filters, favorites, selection mode, and batch delete.
- `ui/modules/LightboxManager.js`: PhotoSwipe integration, metadata panel, compare mode, HD toggle, and image actions.

## Styles

- `ui/styles/main.css`: master import list.
- `ui/styles/design-tokens.css`: shared colors, typography, spacing, and effects.
- `ui/styles/reset.css`: CSS reset and browser normalization.
- `ui/styles/base.css`: global typography and utility classes.
- `ui/styles/theme-nova.css`: main visual theme.
- `ui/styles/components/`: buttons, forms, panels, modals, overlays, animations.
- `ui/styles/layouts/`: app shell, studio, gallery, and responsive layout files.

## UI Data Flow

1. `ui/index.html` provides the DOM structure.
2. `ui/app.js` reads from and writes to `ui/core/state.js`.
3. Requests go through `ui/core/api.js`.
4. Realtime updates arrive through `ui/core/events.js` from `/ws`.
5. Feature managers subscribe to events and update focused regions of the page.

## Frontend Change Rules

- When adding a control, update `ui/index.html`, `ui/app.js`, and `ui/core/state.js` together.
- If the backend consumes the value, also update `ui/core/api.js` and `server/app.py`.
- Keep mobile behavior working; `ui/styles/theme-nova.css` and `ui/index.html` contain responsive UI assumptions.
- Prefer existing modal patterns over browser dialogs.
