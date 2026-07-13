# WebbDuck User Guide

This guide explains the current user-facing workflows in WebbDuck. It is the longer reference guide; `docs/SIMPLE_GUIDE.md` is the shorter version shown in the in-app help modal.

## Main Screens

### Studio

Studio is where you configure and submit generation jobs.

- Pick a base model, write a prompt, and optionally add a negative prompt.
- Adjust width, height, steps, CFG, scheduler, seed, and batch size.
- Upload an input image for img2img or inpaint workflows.
- Review progress and the latest result in the preview area.

### Queue

Open `Queue` in the top bar to inspect pending and running work.

- Queued jobs show prompt and settings summaries.
- Queued jobs can be canceled.
- Running jobs can receive a cancellation request.
- Queue state updates live over WebSocket instead of polling.

### Gallery

Gallery shows saved outputs in a flat newest-first grid.

- Search matches prompt and metadata keywords through the manifest index.
- Filters include `All`, `HD`, and `Favorites`.
- The `Thumb` slider changes thumbnail size.
- More results load automatically as you scroll.

### Lightbox

Click an image to inspect it full size.

- Metadata includes prompt, negative prompt, model, seed, dimensions, timing, LoRAs, embeddings, and inpaint/outpaint settings.
- Actions include regenerate, upscale, send to inpaint, download, compare, and delete.

## Core Generation Controls

### Prompt and Token Count

- Prompt length is checked by the backend tokenizer.
- WebbDuck warns when a prompt exceeds the 77-token CLIP window.
- Long SDXL prompts are chunk-encoded instead of being hard-truncated.

### Resolution and Aspect Ratio

- Presets include common aspect ratios like `1:1`, `4:3`, `3:2`, `16:9`, `2:3`, and `9:16`.
- `Custom` becomes active when the size does not match a preset.

### Seed

- Leave the seed blank for a random result.
- The randomize button clears the field so the next run uses a fresh seed.
- The last used seed appears in the Studio status area after generation.

### Batch Size

- Batch size controls how many images are generated in one queued job.
- Larger batches use more time and VRAM.

## LoRAs and Embeddings

### LoRAs

- LoRAs are filtered to match the selected base model architecture.
- Each selected LoRA has a weight slider from `0.00` to `2.00` in `0.05` steps.
- Default weights and trigger phrases come from `lora/loras.json` when present.
- Trigger phrases are injected automatically during generation.

### Embeddings

- Embeddings are filtered to match the selected base model architecture.
- Local embedding files are discovered and synced into the embedding catalog.
- Each selected embedding lets you review or edit the token used in prompts.

## Image Editing Workflows

### Img2img

- Upload or drop an image into the Studio input area.
- Lower denoising strength keeps the source closer to the original.
- Higher denoising strength gives WebbDuck more freedom to change the image.

### Inpaint

- Use the mask editor to paint where changes are allowed.
- `Replace` repaints the masked area.
- `Keep` protects the mask and edits the surrounding region.
- Blur softens mask edges for smoother blending.

### Smart Extend / Outpaint

- Extend the canvas to add more scene around an image.
- The default mode handles the seam settings for you.
- Advanced controls can override seam and pyramid tuning when needed.

### Upscale

- You can upscale from the Studio preview or from the lightbox.
- Upscaling runs through the job queue like other GPU-heavy actions.

## Settings and Persistence

Open `Settings` from the top bar to manage app-level options.

- Long-run warning threshold controls the coffee warning prompt.
- `Enable CLIP_SKIP2` sends `clip_skip=2` to generation endpoints.
- Remote web-app plugins can be connected from the same settings area.

Most Studio settings persist in `localStorage`, including prompt fields, size, scheduler, second-pass options, denoising strength, selected LoRAs, and selected embeddings.

## GPU Idle Unload

The model is automatically unloaded from VRAM after 5 minutes of inactivity to free resources. Generation will reload it on the next job (adds a few seconds of startup time). This is configurable via the `WEBBDUCK_IDLE_UNLOAD_SECONDS` environment variable.

## Optional Captioning

- If a captioner plugin is installed, the input-image area exposes a `Caption` action.
- Captioning is optional and may use substantial VRAM.
- WebbDuck unloads generation pipelines before captioning to free memory.

## Troubleshooting Tips

- Slow generation: reduce steps, resolution, or batch size.
- Memory issues: close other GPU apps or lower the requested size/batch.
- Prompt not matching output: simplify the prompt or lower CFG slightly.
- Search returning nothing: try fewer keywords first, then narrow it down.
- Missing caption button: verify plugin installation and restart WebbDuck.
