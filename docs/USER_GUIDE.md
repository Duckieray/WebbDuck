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
- `Pause Queue` lets the in-flight job finish, then releases the GPU and holds
  before starting the next queued job. `Resume Queue` continues where it left off.
  Jobs added while paused wait until you resume.
- Queue state updates live over WebSocket instead of polling.

### Gallery

Gallery shows saved outputs in a flat newest-first grid.

- Search matches prompt and metadata keywords through the manifest index.
- Filters include `All`, `HD`, and `Favorites`.
- The `Thumb` slider changes thumbnail size.
- More results load automatically as you scroll.
- `Select` enters selection mode, where `Delete (N)` removes the selected images.

### Editing Images

Click `Edit` in the lightbox to open the editor. Edits are non-destructive: the original is always
kept and the result is saved as a new image next to it in the same run folder, so nothing you
already generated is lost.

- `Crop` — the whole image is shown with your crop drawn on top, so you can always see what you are
  cutting away. Drag on the image to draw a crop; once one exists, drag inside it to move it, or
  drag a handle to resize. Picking a ratio (`1:1` / `4:3` / `3:2` / `16:9` / `2:3` / `9:16` / `Custom`) starts a centred
  crop of that shape immediately. `Full Frame` clears the crop.
- `Rotate & Flip` turns the image in 90-degree steps and mirrors it horizontally or vertically.
- `Resize` sets an explicit output size. `Lock aspect ratio` keeps the other dimension in step, and
  `Snap to multiples of 8` rounds to a size every backend accepts. `Reset Size` returns to the cropped size.
- `Adjust` applies brightness, contrast, and saturation. Browsers without canvas filter support disable
  these sliders automatically.
- The readout shows the source size on the left and the resulting size on the right.

Editor actions:

- `Use as Input Image` loads the edited pixels straight into the Studio Input Image and switches to the
  Studio view, so you can immediately img2img, inpaint, or outpaint from your edit.
- `Save Copy` writes the edit into the gallery as a new image.
- `Reset All` clears every adjustment; `Cancel` closes without saving.

The lightbox `Inpaint` button does the same import without opening the editor, sending the full-size
original image unchanged. It loads the image into the Studio input slot; the operation starts as img2img
and becomes inpaint once you draw a mask. (This button was previously duplicated as `Input Image`; the
duplicate ran the exact same code path and was removed.)

### Lightbox

Click an image to inspect it full size.

- Metadata includes prompt, negative prompt, model, seed, dimensions, timing, LoRAs, embeddings, and inpaint/outpaint settings.
- Actions include regenerate, upscale, edit, send to inpaint, download, compare, and delete.
- `Edit` only appears for saved gallery outputs.

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

### Realism handling (Krea 2)

You do not need to configure this. For Krea 2 models WebbDuck reads the prompt and
the model's own metadata, and when the request looks photographic it automatically
uses the photo-trained 2x decoder and re-renders detected faces at higher effective
detail. Photographic prompts (`portrait photograph`, `skin pores`, `85mm`,
`natural lighting`, `bokeh`) get this treatment; anime, manga, cel shading and line
art do not. An ambiguous prompt is left alone rather than guessed at.

Parameters shows a one-line status such as `Realistic: photographic terms: 85mm,
natural lighting, photograph` so you can see what was chosen and why.

### Advanced Settings (Krea 2)

Collapsed by default, at the bottom of Parameters. Everything here overrides the
automatic behaviour.

- **Realism mode** is `auto` by default. Set it to `always` to force the
  photographic decoder and face refinement on, or `never` to turn both off - useful
  for a stylized prompt that still contains photographic words.
- **Refine faces**, **Detail Strength**, **Min Face px** and **Max Faces** control
  face refinement directly. Unticking **Refine faces** stops the automatic pass while
  leaving the decoder choice alone.
- Lower **Detail Strength** keeps the original likeness; higher gives cleaner skin.
  The default of 0.30 is deliberately conservative.
- Faces smaller than **Min Face px** (default 32) are skipped, as are faces that
  already fill more than 40% of the frame - those are already sharp, and
  re-rendering them only risks changing the likeness. The largest faces are refined
  first.
- Each refined face is composited back only inside the facial region, so hair,
  clothing and background are left exactly as generated.
- Face refinement costs roughly one extra generation per detected face, so it is
  most valuable for wide shots and group photos.
- If no face is found, or the detector is unavailable, the image is saved normally
  and the reason is recorded in the image metadata (`face_detailer`).
- The decoder and detector download their models on first use and are cached.

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

## Identity and Personas

- Upload reference photos once, save them as a named persona, and reuse that identity in later generations.
- SDXL uses the saved persona through IP-Adapter FaceID; FLUX.2 uses the same photo set as native multi-reference identity conditioning (up to 5 references, ordered by importance — the first reference is conditioned strongest).
- SDXL FaceID auto-loads its companion `ip-adapter-faceid_sdxl_lora.safetensors` at the **LoRA Scale** value (default `0.60`). Keep **Adapter Scale** around `0.6-0.8`. For likeness, use a normal SDXL schedule (~20-30 steps, CFG 3-7); low-step LCM/CFG 1 recipes weaken identity and can also break the adapter's CFG embeds.
- SDXL FaceID is experimental and less consistent than the Krea 2 and FLUX.2 identity paths. Keep identity prompts short and neutral, and avoid stylized character tags that paint over the face. Do not put negations such as "no tattoos" or "no makeup" in the positive prompt; SDXL often renders the thing you are negating. Put those terms in the negative prompt instead.
- FLUX.2 personas get three behind-the-scenes likeness upgrades, all validated by identity A/B sweeps:
  - **Ref Face Crop (Auto):** references are re-framed around the strongest detected face as a tight square portrait. Photos with no detectable face are used unchanged.
  - **Anchor boost:** duplicates your strongest (first) reference so it dominates identity conditioning.
  - **Close-up face-focus framing:** adds a portrait-framing instruction so generated scenes keep the face large, centered, and sharp.
- Load a saved persona from the dropdown; its reference order and tuning options restore automatically and can be overridden per request.
- **Krea Identity** (Krea 2 checkpoints): the same persona photos drive the Krea dual-conditioning recipe — a single anchor reference plus the `krea2-identity-edit` LoRA. When a Krea model is selected the provider switches automatically, the same **Identity Strength** slider controls likeness (mapped to the `ref_boost` dial, where 0.1 is the balanced default), and Grounding Res + Identity LoRA Rank appear. Reference count is capped at one, so the last image you select becomes the anchor.

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
- SDXL and FLUX.2 Klein support this workflow. Krea 2 masked editing is not available yet.

### Smart Extend / Outpaint

- Extend the canvas to add more scene around an image.
- The default mode handles the seam settings for you.
- `Auto Step Outpaint` is on by default, in the UI, in the backend, and for existing saved sessions. It grows the canvas in stages so each new strip is generated next to real detail from the previous stage. Turning it off switches to a single pass over the whole new area, which is faster but noticeably weaker on large expansions.
- Advanced controls can override seam and pyramid tuning when needed.
- FLUX.2 Klein uses its native masked-inpaint runtime for a single extension pass and then locks the original placed source pixels. SDXL retains its staged seam-refinement workflow.
- Krea 2 outpainting remains unavailable until a model-trained masked-edit runtime is released.

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
