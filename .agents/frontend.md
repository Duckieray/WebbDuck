# Frontend Reference

WebbDuck's frontend is a zero-build web app built from plain HTML, ES modules, and CSS.

## Main Files

- `ui/index.html`: app shell, Studio controls, Gallery, modals, lightbox mounts, and plugin view containers.
- `ui/app.js`: stable browser composition entrypoint.
- `ui/app_main.js`: top-level orchestration for generation, queue UI, settings, remote plugins, smart extend, uploads, preview state, and gallery/lightbox interactions.

## Core Modules

- `ui/core/api.js`: fetch wrappers for server routes.
- `ui/core/events.js`: local event bus and WebSocket client for `/ws`.
- `ui/core/state.js`: persisted Studio state in `localStorage` under `webbduck_state_v2`.
- `ui/core/utils.js`: DOM helpers, form-data helpers, throttling/debouncing, downloads, and toast UI.
- `ui/core/modelCapabilities.js`: capability-driven model profile layer; gates sections/controls from `/model-catalog` capabilities and never dispatches on architecture names.

## Feature Modules

- `ui/modules/ProgressManager.js`: progress ring, status bar, and VRAM display.
- `ui/modules/MaskEditor.js`: full-screen inpaint mask editor.
- `ui/modules/LoraManager.js`: LoRA loading, selection, weights, and restore-from-state behavior.
- `ui/modules/EmbeddingManager.js`: embedding loading, selection, token editing, and restore-from-state behavior.
- `ui/modules/GalleryManager.js`: gallery paging, search, filters, favorites, selection mode, and batch delete.
- `ui/modules/LightboxManager.js`: PhotoSwipe integration, metadata panel, compare mode, HD toggle, and image actions. `Edit` and `Input Image` are only shown for `/outputs/` sources; its global Delete/Backspace handler is suppressed while `body.image-editor-open` is set.
- `ui/modules/ImageEditor.js`: self-mounting full-screen editor overlay. Crop/rotate/flip happen on
  an offscreen `<canvas>` at full resolution. The preview always paints the *whole* oriented frame
  with the crop drawn as an overlay, so the excluded area stays visible while dragging; the crop
  rectangle is stored normalized (0..1) in the oriented frame so a rotation carries it across, and
  any pixel floor is applied *after* converting back to pixels. Brightness/contrast/saturation use `ctx.filter`
  (feature-detected; sliders disable if unsupported). `Save Copy` posts the rendered PNG to
  `/image/edit`; `Use as Input Image` hands the same blob to `handleImageUpload()` so it lands in
  the Studio through the normal funnel.
- `ui/modules/PersonaIdentityUI.js`: Identity / Persona presentation; provider switch (`faceid_sdxl` / `flux2_native` / `krea2_identity_edit`), persona hints, Krea-only controls, and per-provider slider defaults for the identity section (IP-Adapter FaceID is the SDXL `faceid_sdxl` provider).
- `ui/modules/PersonaIdentityUI_impl.js`: implementation details behind `PersonaIdentityUI.js`.
- `ui/modules/ProviderCredentialsSettings.js`: optional Hugging Face/Civitai credential controls injected into Settings; token values are never persisted in browser state.

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
2. `ui/app_main.js` (via `ui/app.js`) reads from and writes to `ui/core/state.js`.
3. Requests go through `ui/core/api.js`.
4. Realtime updates arrive through `ui/core/events.js` from `/ws`.
5. Feature managers subscribe to events and update focused regions of the page.

## Frontend Change Rules

- When adding a control, update `ui/index.html`, `ui/app_main.js`, and `ui/core/state.js` together.
- If the backend consumes the value, also update `ui/core/api.js` and `server/app.py`.
- Keep `ui/app.js` small; put page orchestration in `ui/app_main.js` and feature logic in `ui/modules/`.
- Keep mobile behavior working; `ui/styles/layouts/responsive.css`, `ui/styles/theme-nova.css`, and `ui/index.html` contain responsive UI assumptions.
- Prefer existing modal patterns over browser dialogs.
