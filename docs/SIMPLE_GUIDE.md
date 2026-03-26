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
- `Scheduler`: the generation method.
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

## 7. Queue and Gallery

- Open `Queue` to see pending and running jobs.
- Open `Gallery` to browse saved images.
- Search Gallery with simple keywords like `forest night`.
- Use the `Thumb` slider to change gallery thumbnail size.

## 8. Lightbox

Click any gallery image to open the full viewer.

You can:

- regenerate
- upscale
- send to inpaint
- download
- delete

Use `Info` to see prompt, model, seed, LoRAs, embeddings, and other settings.

## 9. Good Starter Preset

- Size: `1024 x 1024`
- Steps: `30`
- CFG: `6` to `8`
- Batch: `1`
- Denoising for img2img: about `0.75`

Change one setting at a time while learning.

## 10. If Something Goes Wrong

- Slow generation: lower size, steps, or batch size.
- Memory errors: close other GPU apps and lower the requested size.
- Bad prompt match: simplify the prompt and lower CFG slightly.
- Search finds nothing: try fewer words first.
