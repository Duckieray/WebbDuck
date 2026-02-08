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
- Large Studio preview with action toolbar (zoom/upscale/inpaint/download).
- Prompt token counter with over-limit warning styling.
- Resolution preset chips with active `Custom` state.
- Seed randomize icon button and persisted seed handling.
- Dedicated Queue modal with job metadata, cancel action, and img2img thumbnail previews.
- Lightbox info panel at the bottom with toggle and full metadata.
- Gallery search and lazy thumbnail loading.

## Event Flow

WebSocket events from `/ws` are translated to local events:
- `state` -> `Events.STATUS_UPDATE`
- `queue` -> `Events.QUEUE_UPDATE`
- `catalog` -> `Events.CATALOG_UPDATE`

This avoids frontend polling loops for queue/catalog freshness.

## LoRA UX Notes

- LoRA selector options are loaded from `/models/{base_model}/loras`.
- Slider range is `0.00` to `2.00` in `0.05` increments.
- Default slider value uses backend `weight` from `loras.json`.
- Selected LoRAs are persisted in local state and restored when compatible.

## Editing Guidance

- Add API wrappers in `ui/core/api.js`.
- Prefer module-local logic in `ui/modules/*` for feature-specific behavior.
- Use event bus (`ui/core/events.js`) for cross-module communication.
- Keep style changes in `ui/styles/theme-nova.css` unless the change is a reusable primitive.
