---
id: TASK-008
title: Add basic image editing from the gallery
status: Done
assignee: []
created_date: '2026-09-25 00:00'
updated_date: '2026-09-25 00:00'
labels:
  - ui
  - gallery
  - server
dependencies: []
references:
  - server/image_edit.py
  - server/app.py
  - server/storage.py
  - ui/modules/ImageEditor.js
  - ui/modules/LightboxManager.js
  - ui/modules/GalleryManager.js
  - ui/core/api.js
documentation:
  - docs/ARCHITECTURE.md
  - docs/USER_GUIDE.md
  - .agents/backend-runtime.md
  - .agents/frontend.md
  - ui/README.md
  - tests/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The gallery currently supports only whole-image actions (delete, favorite, upscale,
inpaint, regenerate). Add a basic editing workflow so a user can crop, rotate/flip,
resize, and make tonal adjustments to an existing output, then either save the result
as a new image in the same run folder or push it straight into the Studio Input Image.

Editing is reached from the lightbox `Edit` action, next to `Upscale`. The work runs
entirely in the browser on a `<canvas>` (no GPU, no upload round-trip for previews); only
the final PNG is sent to the server, which writes it next to the source image and
registers it in the manifest. Originals are never overwritten.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `server/image_edit.py` resolves output web paths with an explicit containment check (rejecting `..` traversal and paths outside the output root) and allocates non-colliding `<stem>_edit[_N].png` targets.
- [x] #2 `POST /image/edit` (multipart `path` + `image` file) writes the uploaded result as a new PNG in the source run folder, registers it in the manifest with an `edited_from` pointer, and returns the new web path.
- [x] #4 Edited images appear in `/gallery`, gallery search, and the HD/favorites filters; their thumbnails are generated on demand like any other output.
- [x] #5 An editor overlay provides crop (the full frame stays visible with the crop drawn as an overlay; drag to draw, drag inside to move, handles to resize, plus free/1:1/3:4/4:5/16:9/9:16 presets that seed a centred crop), 90-degree rotate, horizontal/vertical flip, resize with aspect lock and optional multiple-of-8 snapping, and brightness/contrast/saturation sliders with a live preview.
- [x] #6 The editor offers "Save Copy" (writes a new gallery image) and "Use as Input Image" (feeds the edited pixels into the Studio Input Image through the existing upload funnel) plus Reset and Cancel.
- [x] #7 The lightbox exposes `Edit` (between `Upscale` and `DuckMotion`) and `Input Image` actions alongside the existing buttons, hidden for non-`/outputs/` sources. Gallery select mode keeps only `Delete (N)`; editing is not reachable from selection mode.
- [x] #8 `POST /image/edit` broadcasts a `gallery` websocket event so other open tabs refresh; the frontend event map handles it.
- [x] #9 Focused tests cover path containment, edit-path allocation, manifest lineage, the `/image/edit` endpoint, and the frontend contract (entry point, button order, crop render/geometry rules).
- [x] #10 Docs updated: `docs/ARCHITECTURE.md`, `docs/USER_GUIDE.md`, `.agents/backend-runtime.md`, `.agents/frontend.md`, `ui/README.md`, `tests/README.md`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `server/image_edit.py` with path containment, edit-path allocation, browser-PNG persistence, and manifest registration (lineage) for new outputs. No server-side pixel work: the browser already rendered it.
2. Wire `POST /image/edit` in `server/app.py`, reusing the thumbnail semaphore/executor pattern so the write stays off the event loop and the GPU worker.
3. Add a `ui/core/api.js` wrapper for the endpoint.
4. Add `ui/modules/ImageEditor.js` (self-mounting overlay, canvas crop handles, transform stack, live preview) and its styles in `ui/styles/components/overlays.css`. The preview paints the whole oriented frame with the crop drawn as an overlay.
5. Wire `Edit` / `Input Image` in `LightboxManager` + `app_main.js`. Editing stays lightbox-only; gallery select mode keeps only `Delete (N)`.
6. Add the `gallery` websocket event type and frontend dispatch.
7. Add `tests/test_image_edit.py` + `tests/test_image_edit_ui_contract.py` and run focused verification.
8. Update the documentation set listed in the frontmatter.
<!-- SECTION:PLAN:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] #1 Crop/rotate/flip/resize/adjust work in the browser with a live preview and no console errors (verified in headless Chromium against a harness page)
- [x] #2 Saved edits land in the gallery as new images; originals untouched (verified over HTTP + on disk)
- [ ] #3 "Use as Input Image" loads the edited pixels into the Studio and switches views (wiring covered by static contract test; the Studio hand-off itself is not browser-verified)
- [x] #4 `Save Copy` and `Use as Input Image` both export the cropped/rotated/resized result (export verified to match the crop rectangle in-browser)
- [x] #5 Traversal attempts against the new endpoints are rejected (verified over HTTP and by unit test)
- [x] #6 Targeted tests pass
- [x] #7 Docs updated per project conventions
<!-- DOD:END -->

## Verification Notes

<!-- SECTION:VERIFICATION:BEGIN -->
Backend, HTTP, WebSocket, and the crop/transform canvas path are verified. The Studio
Input Image hand-off is not browser-verified (see below).

Ran and passing:
- `pytest tests/test_image_edit.py tests/test_image_edit_ui_contract.py tests/test_thumbnails.py`
- `pytest tests/test_server.py -k "not GenerationEndpoints and not generate"` - 14 passed
- `python -m compileall` on the touched Python files, `node --check` on every touched JS file
- Live HTTP smoke on a temp output root: `/image/edit` wrote `<stem>_edit.png`; `..` traversal
  returned 400; the edited image appeared in `/gallery` and `/gallery/search`; its thumbnail
  served 200; the original was left untouched.
- WebSocket smoke: a subscriber received
  `{"type":"gallery","payload":{"reason":"edit","images":[...]}}` after an edit.
- Headless Chromium against a harness page that loads the real `ui/modules/ImageEditor.js`:
  the whole frame paints (96000/96000 px); a drag produces exactly the dragged crop (260x170
  at offset 40,30); the excluded corner stays visible underneath; dragging inside moves the crop
  without resizing; `renderExport()` returns the crop rectangle; rotating 90 degrees swaps the
  oriented frame; console clean.

Two real bugs were found and fixed this way, and both are now locked down by
`tests/test_image_edit_ui_contract.py`:
- The preview drew only the crop region scaled to fill the canvas, so there was no surrounding
  image left to aim at.
- `resolveCrop()` floored the *normalized* crop width with `Math.max(1, crop.w)`, which inflated
  every crop smaller than the full frame back to the whole image on mouseup.

Not verified:
- "Use as Input Image" into the live Studio. The full app page crashes headless Chromium in this
  sandbox with `FATAL: SkFontMgr_FontConfigInterface.cpp: Not implemented` - a font-stack defect,
  reproducible with JS disabled, all subresources blocked, and under swiftshader/angle/old-headless
  flags, while plain HTML and `/health` load fine. The editor was therefore verified against a
  harness page with the text-heavy chrome hidden (which is what triggers the font crash). Please
  click through the editor once in a real browser.

Pre-existing unrelated failures, not caused by this task:
- `tests/test_embedding_loading.py`, `tests/test_krea2_runtime_policy.py` - collection errors
- `tests/test_capability_driven_ui.py::test_identity_provider_presentation_lives_only_in_persona_module`
  - reads only `ui/core/modelCapabilities.js` and `ui/modules/PersonaIdentityUI.js`, neither of which
  this task modifies
- `tests/test_server.py::TestGenerationEndpoints` - hangs because the host already has a WebbDuck
  server holding 13.7/15.5 GB VRAM
<!-- SECTION:VERIFICATION:END -->
