# WebbDuck Simple Guide

This guide is for first-time users.
WebbDuck makes images from text on your own machine.

## 1. What WebbDuck Does

- You type what you want to see.
- WebbDuck creates one or more images.
- You can fix parts of an image, extend an image, or upscale an image.
- Your images and settings are saved so you can find them later.

## 2. Start the Server

Run:

```bash
python run.py
```

Then open:

`http://localhost:8000`

## 3. First Image (Quick Start)

1. Open the `Studio` tab.
2. Pick a `Model`.
3. In `Prompt`, describe your image.
4. Optional: add a `Negative Prompt` (things to avoid).
5. Click `Generate`.

When done, your newest image appears in the big preview.

## 4. Prompt Tips (Simple)

- Start short and clear.
- Add details after the main idea.
- Example prompt: `portrait photo of a woman with glasses, soft light, bedroom background`
- Example negative prompt: `blurry, bad hands, watermark, text`

## 5. Main Settings (What They Mean)

- `Width` and `Height`: image size.
  Bigger size can look better, but is slower and uses more memory.
- `Aspect Ratio` buttons: quick shape presets (1:1, 9:16, etc.).
- `Steps`: how long the model thinks.
  More steps can improve detail, but takes longer.
- `CFG`: how strongly the model follows your prompt.
- `Scheduler`: the generation method.
  Different schedulers can give slightly different look/speed.
- `Seed`: random starting number.
  Same seed + same settings gives a similar result.
- `Batch Size`: number of images made in one request.

## 6. Input Image, Inpaint, and Smart Extend

If you upload an image:

- `Denoising Strength`:
  Lower = keep image close to original.
  Higher = change image more.
- `Mask`: paint where edits are allowed.
- `Inpaint Mode`:
  - `Replace`: repaint masked area.
  - `Keep`: protect masked area and change outside it.
- `Smart Extend`: make canvas bigger and fill new space.
  Use drag handles in preview to choose where to extend.

## 7. Queue (Many Jobs)

- If you click `Generate` many times, jobs go to queue.
- Open `Queue` in the top bar to see pending/running jobs.
- You can cancel queued jobs.

## 8. Gallery (Find Old Images)

- Open `Gallery` to see saved runs.
- Use search to find old images by keywords.
- Search uses keyword matching.
  Example: `blonde glasses` finds images that contain both words.

## 9. Lightbox (Image Viewer)

Click any image to open viewer.

You can:

- Regenerate with same settings
- Upscale
- Send to Inpaint
- Download
- Delete image or run

Use `Info` to show metadata like prompt, model, seed, LoRAs, and inpaint/outpaint settings.

## 10. If Something Goes Wrong

- Generation is very slow:
  lower size, steps, or batch size.
- You get memory errors:
  close other heavy apps and reduce batch/size.
- Result does not match prompt:
  simplify prompt and lower CFG a little.
- Search finds nothing:
  try fewer words first, then add more.

## 11. Good Starter Preset

Try this first:

- Size: `1024 x 1024`
- Steps: `30`
- CFG: `6` to `8`
- Batch: `1`
- Denoising (img2img): `0.75`

Then adjust one setting at a time.
