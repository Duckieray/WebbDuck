# WebbDuck UI

WebbDuck UI is a zero-build frontend using ES modules and vanilla CSS.

## Structure

```text
ui/
|- index.html
|- app.js
|- core/
|  |- api.js
|  |- events.js
|  |- state.js
|  |- utils.js
|- modules/
|  |- LightboxManager.js
|  |- LoraManager.js
|  |- MaskEditor.js
|  |- ProgressManager.js
|- styles/
|  |- main.css
|  |- theme-nova.css
|  |- design-tokens.css
|  |- base.css
|  |- reset.css
|  |- components/
|  |- layouts/
```

## Key UI Capabilities

- Studio + Gallery tabbed layout (desktop and mobile).
- Mobile Studio pane toggle (`Preview` / `Settings`) for full-height usability.
- Large Studio preview with action toolbar (zoom/upscale/inpaint/download).
- Docked generation progress panel in the Studio status bar with cancel control.
- Prompt token counter with over-limit warning styling.
- Resolution preset chips with active `Custom` state.
- Seed randomize icon sets true random mode (clears seed input) with persisted seed handling.
- Dedicated Queue modal with job metadata, cancel action for queued/running jobs, and img2img thumbnail previews.
- Settings modal for long-run warning threshold and CLIP_SKIP2 toggle.
- Smart Extend has a compact default mode plus an optional `Advanced` toggle for manual seam/pyramid tuning.
- Lightbox info panel at the bottom with toggle and full metadata.
- Gallery flat-grid view with keyword search, filter chips (`All`/`HD`/`Favorites`), thumbnail-size slider, and infinite scroll.

## Event Flow

WebSocket events from `/ws` are translated to local events:
- `state` -> `Events.STATUS_UPDATE`
- `queue` -> `Events.QUEUE_UPDATE`
- `catalog` -> `Events.CATALOG_UPDATE`

This avoids frontend polling loops for queue/catalog freshness.

## LoRA UX Notes

- LoRA selector options are loaded from `/models/{base_model:path}/loras`.
- Slider range is `0.00` to `2.00` in `0.05` increments.
- Default slider value uses backend `weight` from `loras.json`.
- Selected LoRAs are persisted in local state and restored when compatible.

## Editing Guidance

- Add API wrappers in `ui/core/api.js`.
- Prefer module-local logic in `ui/modules/*` for feature-specific behavior.
- Use event bus (`ui/core/events.js`) for cross-module communication.
- Keep style changes in `ui/styles/theme-nova.css` unless the change is a reusable primitive.
