# WebbDuck User Guide

This guide describes the current UI and workflows in WebbDuck.

## Studio

The Studio view is where generation jobs are configured and submitted.

### Core Inputs

- `Model`: Base SDXL checkpoint.
- `Prompt`: Main prompt text.
- `Negative Prompt`: Undesired features.
- `Parameters`: Width, height, steps, CFG, scheduler, seed.
- `Batch Size`: Always docked near the bottom with quick access.

### Prompt Token Counter

- Prompt token count updates from the backend tokenizer.
- Over 77 tokens triggers warning/danger styling and a tooltip warning.
- Prompts longer than 77 tokens are chunk-encoded for SDXL instead of being hard-truncated.

### Resolution Presets

- Presets include `1:1`, `4:3`, `3:2`, `16:9`, `2:3`, `9:16`.
- `Custom` is highlighted when width/height does not match a preset.

### Seed Behavior

- Seed input can be blank for random seed.
- Randomize button sets seed mode to random by clearing the seed input.
- Last used seed is shown in the Studio status bar after generation.

### LoRA Stack

- LoRAs are filtered by selected model architecture.
- Each LoRA has weight slider range `0.00` to `2.00` with `0.05` steps.
- Default LoRA weight is loaded from `lora/loras.json` when available.
- Selected LoRAs persist across refresh (when still compatible with model).
- Trigger phrases defined in `loras.json` are injected into the prompt automatically at generation time.

### Embedding Stack

- Embeddings are filtered by selected model architecture.
- Local embedding files are discovered from the embeddings folder and synced into `embeddings/embeddings.json`.
- Each selected embedding can set/edit the active token used in prompt text.
- Selected embeddings persist across refresh (when still compatible with model).

### Input Image / Inpaint

- Drag/drop or click upload for img2img.
- Optional caption generation if a captioner plugin is installed.
- Mask editor supports draw/erase/invert, blur, and replace/keep inpaint mode.

### Preview Area

- Placeholder appears until an image is generated.
- Toolbar actions: zoom, upscale, send to inpaint, download.
- Progress card appears in the bottom status area during generation with cancel action.

## Queue

Queue is shown in a dedicated modal (`Queue` button in top bar).

Each queued/running job can show:
- status and queue position,
- model/scheduler/steps/cfg/batch,
- truncated prompt,
- seed/negative/LoRA summary in expandable details,
- embedding summary in expandable details,
- img2img/inpaint input thumbnail when available.

Queued jobs can be canceled from the modal.
Running jobs can also be cancellation-requested from the modal or the progress card.

## Gallery

- Images are shown in a flat newest-first grid (not grouped into run rows).
- Search matches prompt/metadata keywords globally using manifest-backed search.
  - All typed terms must be present, but can appear in any order.
- Thumbnails are loaded through `/thumbs/...` for lighter browsing.
- Filters include `All`, `HD`, and `Favorites`.
- Thumbnail size can be adjusted with the gallery `Thumb` slider.
- More images auto-load as you scroll (infinite scroll).

## Mobile Studio

- On mobile, Studio includes a `Preview` / `Settings` toggle in the top bar.
- `Settings` view keeps controls, batch, and action buttons accessible.
- `Preview` view shows the image workspace full-height.

## Lightbox

- Open any image to inspect at full size.
- Bottom metadata/info panel includes prompt, negative, model, seed, settings, inpaint/outpaint settings, LoRAs, embeddings, and timing info.
- `Info` toggle shows/hides metadata panel.
- Actions: regenerate, upscale, inpaint, download, compare (when variant exists), delete.

## Settings Modal

Open `Settings` from the top bar to configure:
- `Coffee Warning (min)` threshold for long-run confirmation prompts.
- `Enable CLIP_SKIP2` toggle (sends `clip_skip=2` to generation endpoints).

## State Persistence

Most Studio settings are persisted in `localStorage` and restored on refresh, including prompt fields, dimensions, scheduler, second-pass settings, denoise strength, selected LoRAs, and selected embeddings.

## GPU Idle Unload

The model is automatically unloaded from VRAM after 5 minutes of inactivity to free resources. Generation will reload it on the next job (adds a few seconds of startup time). This is configurable via the `WEBBDUCK_IDLE_UNLOAD_SECONDS` environment variable.
