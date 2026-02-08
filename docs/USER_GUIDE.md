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

### Resolution Presets

- Presets include `1:1`, `4:3`, `3:2`, `16:9`, `2:3`, `9:16`.
- `Custom` is highlighted when width/height does not match a preset.

### Seed Behavior

- Seed input can be blank for random seed.
- Randomize button fills the seed input with a new generated seed value.
- Last used seed is shown in the Studio status bar after generation.

### LoRA Stack

- LoRAs are filtered by selected model architecture.
- Each LoRA has weight slider range `0.00` to `2.00` with `0.05` steps.
- Default LoRA weight is loaded from `lora/loras.json` when available.
- Selected LoRAs persist across refresh (when still compatible with model).
- Trigger phrases defined in `loras.json` are injected into the prompt automatically at generation time.

### Input Image / Inpaint

- Drag/drop or click upload for img2img.
- Optional caption generation if a captioner plugin is installed.
- Mask editor supports draw/erase/invert, blur, and replace/keep inpaint mode.

### Preview Area

- Placeholder appears until an image is generated.
- Toolbar actions: zoom, upscale, send to inpaint, download.
- Progress card appears during generation with cancel action.

## Queue

Queue is shown in a dedicated modal (`Queue` button in top bar).

Each queued/running job can show:
- status and queue position,
- model/scheduler/steps/cfg/batch,
- truncated prompt,
- seed/negative/LoRA summary in expandable details,
- img2img/inpaint input thumbnail when available.

Queued jobs can be canceled from the modal (running jobs cannot).

## Gallery

- Sessions are listed newest-first.
- Search filters by prompt/metadata text.
- Thumbnails are loaded through `/thumbs/...` for lighter browsing.

## Lightbox

- Open any image to inspect at full size.
- Bottom metadata/info panel includes prompt, negative, model, seed, settings, and LoRAs.
- `Info` toggle shows/hides metadata panel.
- Actions: regenerate, upscale, inpaint, download, compare (when variant exists), delete.

## State Persistence

Most Studio settings are persisted in `localStorage` and restored on refresh, including prompt fields, dimensions, scheduler, second-pass settings, denoise strength, and selected LoRAs.
