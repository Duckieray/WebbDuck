# WebbDuck Simple Guide

This is the short help guide shown inside WebbDuck. Use `docs/USER_GUIDE.md` for the full version.

## 1. What WebbDuck Does

- You type what you want to see.
- WebbDuck makes images on your own machine.
- You can edit with img2img, inpaint, smart extend, and upscale tools.
- Saved images show up later in Gallery.

## 2. Start WebbDuck

Run:

```bash
python run.py
```

Then open `http://localhost:8010`.

## 3. Make Your First Image

1. Open `Studio`.
2. Pick a `Model`.
3. Type a short prompt.
4. Optional: add a negative prompt.
5. Click `Generate`.

Your newest result appears in the preview area when the job finishes.

## 4. Quick Prompt Tips

- Start simple.
- Put the main subject first.
- Add style and lighting after the main idea.
- Example prompt: `portrait photo of a woman with glasses, soft window light`
- Example negative prompt: `blurry, bad hands, watermark, text`

## 5. Main Settings

- `Width` and `Height`: image size.
- `Steps`: how long the model works.
- `CFG`: how strongly the prompt is followed.
- `Scheduler`: the generation method. `LCM` is for LCM LoRAs and low-step models (about 4-8 steps, CFG 1-2).
- `Seed`: repeatable starting point; blank means random.
- `Batch Size`: how many images to make in one job.

## 6. Edit an Existing Image

Upload an image to unlock edit tools.

- `Denoising Strength`: low keeps the original closer; high changes it more.
- `Mask`: paint where edits can happen.
- `Inpaint Mode`:
  - `Replace` changes the masked area.
  - `Keep` protects the mask and edits around it.
- `Smart Extend`: makes the canvas bigger and fills the new space.
  - `Auto Step Outpaint` is on by default and should stay on. It grows the canvas
    in small steps, so each new strip of picture is generated next to real detail
    from the previous step. With it off, WebbDuck instead does a single pass over
    the whole new area, which has to invent the entire continuation in one go and
    often comes back looking soft, stretched, or like a repeat of the original.
    This setting is now enabled in every layer, including sessions saved by older
    builds, so a stale saved value can no longer keep the single-pass mode on.
  - `Step Growth` controls how much each step adds. Lower is safer for big expansions.

## 7. Queue and Gallery

- Open `Queue` to see pending and running jobs.
- Use `Pause Queue` to let the current job finish, free the GPU, and hold the rest.
  Use `Resume Queue` to keep going.
- Open `Gallery` to browse saved images.
- Search Gallery with simple keywords like `forest night`.
- Use the `Thumb` slider to change gallery thumbnail size.
- Use `Select` to pick several images, then `Delete (N)` to remove them.

## 8. Lightbox

Click any gallery image to open the full viewer.

You can:

- regenerate
- upscale
- edit
- send to inpaint
- download
- delete

Use `Info` to see prompt, model, seed, LoRAs, embeddings, and other settings.

## 9. Editing an Image

Click `Edit` in the lightbox to crop, rotate, flip, resize, or adjust an image. The preview updates
as you work, and the size readout shows what you started with next to what you will get.

- The whole picture stays visible with your crop drawn on top, so you can see what you are cutting away.
- Drag on the picture to draw a crop, drag inside it to move it, or drag a handle to resize it. Aspect
  buttons start a centred crop in a common shape like `1:1` or `16:9`.
- `Save Copy` adds the edited version to the Gallery next to the original.
- `Use as Input Image` puts the edited image into the Studio so you can keep working from it.

Your original image is never overwritten.

## 10. Good Starter Preset

- Size: `1024 x 1024`
- Steps: `30`
- CFG: `6` to `8`
- Batch: `1`
- Denoising for img2img: about `0.75`

Change one setting at a time while learning.

## 11. If Something Goes Wrong

- Slow generation: lower size, steps, or batch size.
- Memory errors: close other GPU apps and lower the requested size.
- Bad prompt match: simplify the prompt and lower CFG slightly.
- Search finds nothing: try fewer words first.
