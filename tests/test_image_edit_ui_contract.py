"""Static UI contract checks for gallery image editing.

These are headless contract assertions (no running server, no browser), matching
the style of ``tests/test_capability_driven_ui.py``. They guard the wiring that a
browser test would otherwise be needed for: module registration, API wrappers,
the lightbox/gallery entry points, and the markup the JS binds to by id.
"""

import json
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
EDITOR = (ROOT / "ui" / "modules" / "ImageEditor.js").read_text(encoding="utf-8")
GALLERY = (ROOT / "ui" / "modules" / "GalleryManager.js").read_text(encoding="utf-8")
LIGHTBOX = (ROOT / "ui" / "modules" / "LightboxManager.js").read_text(encoding="utf-8")
API = (ROOT / "ui" / "core" / "api.js").read_text(encoding="utf-8")
EVENTS = (ROOT / "ui" / "core" / "events.js").read_text(encoding="utf-8")
MAIN = (ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
APP = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
CSS = (ROOT / "ui" / "styles" / "components" / "overlays.css").read_text(encoding="utf-8")


VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def _tag_of(source, element_id):
    match = re.search(r"<[a-zA-Z][^>]*\bid=\"" + re.escape(element_id) + r"\"[^>]*>", source)
    assert match, f"no element with id {element_id!r} in ui/index.html"
    return match.group(0)


def test_index_html_is_balanced():
    class Balance(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []
            self.errors = []

        def handle_starttag(self, tag, attrs):
            if tag not in VOID_TAGS:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag):
            if tag in VOID_TAGS:
                return
            assert self.stack, f"stray </{tag}> at {self.getpos()}"
            open_tag, pos = self.stack[-1]
            assert open_tag == tag, (
                f"</{tag}> at {self.getpos()} closes <{open_tag}> opened at {pos}"
            )
            self.stack.pop()

    parser = Balance()
    parser.feed(INDEX)
    assert not parser.stack, f"unclosed tags: {parser.stack}"
    assert not parser.errors


def test_index_html_has_no_nested_labels():
    """Nested <label> is invalid and makes the inner control ambiguous to AT."""

    class LabelDepth(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.depth = 0
            self.nested = []

        def handle_starttag(self, tag, attrs):
            if tag == "label":
                if self.depth:
                    self.nested.append(self.getpos())
                self.depth += 1

        def handle_endtag(self, tag):
            if tag == "label":
                self.depth -= 1

    parser = LabelDepth()
    parser.feed(INDEX)
    assert not parser.nested, f"labels nested inside labels at {parser.nested}"


def test_selection_controls_use_class_visibility_not_the_hidden_attribute():
    """A native [hidden] attribute cannot be undone by toggling the .hidden class."""
    tag = _tag_of(INDEX, "gallery-delete-selected")
    assert "hidden" in tag, "gallery-delete-selected should start hidden"
    assert not re.search(r"\shidden(\s|=|>|$)", tag), (
        "gallery-delete-selected must use class=\"hidden\", not the hidden attribute, "
        "because updateSelectionUI() toggles the class"
    )
    assert "deleteBtn.classList.toggle('hidden', !this.selectionMode)" in GALLERY


def test_editing_is_reached_from_the_lightbox_only():
    """Regression: an earlier revision put Edit behind gallery selection mode."""
    assert "gallery-edit-selected" not in INDEX
    assert "gallery-batch-edit" not in INDEX
    assert "setupBatchEditControls" not in GALLERY
    assert "BatchImageEditRequest" not in INDEX
    assert "batchEditImages" not in API
    assert "/image/batch_edit" not in API
    assert "/image/batch_edit" not in APP


def test_lightbox_edit_sits_between_upscale_and_duckmotion():
    order = [
        INDEX.index('id="lightbox-upscale"'),
        INDEX.index('id="lightbox-edit"'),
        INDEX.index('id="lightbox-duckmotion"'),
    ]
    assert order == sorted(order), (
        "lightbox Edit should appear after Upscale and before DuckMotion"
    )


def test_lightbox_actions_are_present_and_output_scoped():
    assert "lightbox-edit" in INDEX
    # Edit starts hidden and is revealed per-slide, since it only makes sense for
    # images that actually live under the outputs directory.
    assert "display:none" in _tag_of(INDEX, "lightbox-edit")
    assert "onEdit" in LIGHTBOX
    assert "includes('/outputs/')" in LIGHTBOX


def test_duplicate_input_image_button_was_removed():
    """Regression: "Inpaint" and "Input Image" ran the same funnel.

    ``sendToInpaint`` was a one-line wrapper over ``sendToInputImage`` and the upload
    handler clears ``_maskBlob``, so both landed in img2img. Only the toast differed.
    The duplicate "Input Image" button is gone; the surviving "Inpaint" button is what
    loads a gallery image into the Studio input slot.
    """
    assert "lightbox-input-image" not in INDEX
    assert "lightbox-input-image" not in LIGHTBOX
    assert "lightbox-inpaint" in INDEX
    # The dead callback plumbing goes with it.
    assert "onUseAsInput" not in LIGHTBOX, (
        "the removed Input Image handler was the only consumer of this callback"
    )


def test_editor_module_is_registered_and_imported():
    assert "ImageEditor.js" in MAIN
    assert "import { ImageEditor }" in MAIN
    assert "new ImageEditor(" in MAIN


def test_editor_exposes_the_documented_controls():
    for control in (
        "image-editor-canvas",
        "image-editor-crop-reset",
        "image-editor-rot-ccw",
        "image-editor-rot-cw",
        "image-editor-flip-h",
        "image-editor-flip-v",
        "image-editor-width",
        "image-editor-height",
        "image-editor-lock-aspect",
        "image-editor-snap-eight",
        "image-editor-resize-reset",
        "image-editor-brightness",
        "image-editor-contrast",
        "image-editor-saturation",
        "image-editor-save",
        "image-editor-use-input",
        "image-editor-reset",
        "image-editor-cancel",
    ):
        assert control in EDITOR, f"editor is missing {control!r}"


def test_editor_aspect_presets_are_offered():
    """The crop presets, in the advertised order, with Custom as the free option."""
    table = EDITOR[EDITOR.index("const ASPECT_PRESETS = ["):]
    table = table[:table.index("];") + 2]
    order = re.findall(r"label: '([^']+)'", table)
    assert order == ["1:1", "4:3", "3:2", "16:9", "2:3", "9:16", "Custom"], (
        f"crop presets are {order}"
    )
    # Each named preset's ratio must actually match its label.
    for key, ratio in re.findall(r"key: '([^']+)', label: '[^']+', ratio: ([\d./ ]+)", table):
        # The table stores arithmetic expressions (e.g. "4 / 3"), so evaluate rather
        # than trying to parse them as literals.
        value = eval(ratio, {"__builtins__": {}}, {})  # noqa: S307 - test-only arithmetic
        if key == "free":
            assert value == 0, "Custom must be the unconstrained option"
            continue
        a, b = (int(x) for x in key.split(":"))
        assert abs(value - a / b) < 1e-12, f"{key} has ratio {ratio}"
    # Custom stays the default so opening the editor never imposes a ratio.
    assert "aspect: 'free'" in EDITOR


def test_editor_state_round_trips_through_the_backend_contract():
    assert "saveImageEdit(" in EDITOR
    assert "postForm('/image/edit'" in API


def test_editor_renders_on_canvas_and_exports_the_rendered_blob():
    assert "createElement('canvas')" in EDITOR or "getContext('2d')" in EDITOR
    assert "toBlob(" in EDITOR


def test_crop_preview_paints_the_whole_frame_not_just_the_selection():
    """Regression: the preview used to draw only the crop, zoomed to fill the canvas.

    That destroyed the surrounding context, so there was nothing left to aim a crop at.
    """
    render = EDITOR[EDITOR.index("_renderPreview() {"):EDITOR.index("_drawCropOverlay(")]
    assert "ctx.drawImage(this.oriented, 0, 0, viewW, viewH)" in render, (
        "the preview must paint the full oriented frame"
    )
    # The crop must not be used as the drawImage source rectangle.
    assert "crop.x, crop.y, crop.width, crop.height," not in render
    # Export still crops, independently of the preview.
    export = EDITOR[EDITOR.index("async renderExport()"):EDITOR.index("opsSummary()")]
    assert "crop.x, crop.y, crop.width, crop.height," in export
    assert "0, 0, out.width, out.height," in export


def test_crop_pixel_floor_is_applied_after_normalisation():
    """Regression: resolveCrop() floored the *normalized* width to 1.

    ``Math.max(1, crop.w) * width`` inflated every crop smaller than the full frame back
    up to the whole image, so a dragged crop silently snapped to full-frame on mouseup.
    """
    body = EDITOR[EDITOR.index("    resolveCrop() {"):EDITOR.index("    outputSize() {")]
    assert body.strip(), "failed to slice resolveCrop() out of ImageEditor.js"
    assert "Math.max(1, crop.w * width)" in body
    assert "Math.max(1, crop.h * height)" in body
    assert "Math.max(1, crop.w)" not in body, (
        "crop.w is normalized 0..1; flooring it to 1 makes every crop the full frame"
    )
    assert "Math.max(1, crop.h)" not in body


def test_choosing_an_aspect_ratio_seeds_a_visible_crop():
    """Regression: aspect chips did nothing until the user dragged a rect by hand."""
    assert "_centeredCropForAspect(" in EDITOR
    seed = EDITOR[EDITOR.index("_centeredCropForAspect(ratio) {"):]
    assert "return null" in seed.split("    }")[0], "free-form ratio has nothing to seed"
    assert "this.state.crop = seed" in EDITOR


def test_gallery_refreshes_on_the_server_gallery_event():
    # Frontend: a `gallery` websocket frame becomes a local refresh event.
    assert "GALLERY_REFRESH: 'gallery:refresh'" in EVENTS
    assert "data?.type === 'gallery'" in EVENTS
    assert "emit(Events.GALLERY_REFRESH, data.payload || {})" in EVENTS
    # A subscriber exists so the open gallery actually reloads.
    assert "Events.GALLERY_REFRESH" in MAIN


def test_editor_locks_page_scroll_only_while_open():
    assert "image-editor-open" in EDITOR
    assert "body.image-editor-open" in CSS


def _output_src_predicate():
    """Extract the shipped regex from isOutputImageSrc() and compile it here.

    The JS and Python regex semantics agree for this pattern, so we can execute
    the real shipped expression instead of only grepping for it.
    """
    match = re.search(
        r"export function isOutputImageSrc\(value\)\s*\{\s*return typeof value === 'string' && /(.*)/\.test\(value\);",
        LIGHTBOX,
        re.S,
    )
    assert match, "isOutputImageSrc() no longer uses a single inline regex literal"
    return re.compile(match.group(1))


def test_output_source_gate_accepts_the_real_gallery_path_shape():
    """Regression: the gate used includes('/outputs/'), which never matches.

    to_web_path() emits "outputs/<run>/<file>" with no leading slash, so every
    gallery item failed the check and Edit / Input Image / Favorite stayed
    hidden. Assert against the server's actual path format, not a guess.
    """
    from server.storage import to_web_path

    gallery_path = to_web_path(ROOT / "outputs" / "run_20260925" / "img.png")
    assert not gallery_path.startswith("/"), (
        f"to_web_path() unexpectedly grew a leading slash: {gallery_path!r}; "
        "update isOutputImageSrc() and this test together"
    )

    predicate = _output_src_predicate()
    assert predicate.search(gallery_path), (
        f"gate rejects the real gallery path {gallery_path!r}; Edit/Favorite would stay hidden"
    )
    # The rooted form returned by POST /image/edit must match too.
    assert predicate.search("/" + gallery_path)
    # And non-output sources must not.
    for bad in ("/thumbs/run/img_thumb.png", "http://example.com/x.png", "outputs.png", ""):
        assert not predicate.search(bad), f"gate wrongly accepts non-output source {bad!r}"

    # The typeof guard is what makes non-string input safe in JS; mirror it here.
    body = re.search(
        r"return typeof value === 'string' && /(.*)/\.test\(value\);", LIGHTBOX, re.S
    )
    assert body, "the typeof-string guard is gone; isOutputImageSrc(undefined) would throw"


def test_output_source_gate_drives_edit_visibility():
    # The button must be wired to the shared helper, not an inline substring test.
    assert "const isOutput = isOutputImageSrc(" in LIGHTBOX
    assert "editBtn.style.display = isOutput" in LIGHTBOX
    # Ignore comments: the helper's docblock names the old bug it replaced.
    code = re.sub(r"/\*.*?\*/", "", LIGHTBOX, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    assert "includes('/outputs/')" not in code, (
        "the old substring gate is back and will hide the buttons for gallery images"
    )
    # The Inpaint handler reads the full-size source, not the thumbnail, so removing
    # the old "Input Image" button did not regress the resolution it preferred.
    inpaint_handler = LIGHTBOX.split("listen(byId('lightbox-inpaint')")[1].split("listen(")[0]
    assert "originalSrc || curr?.src" in inpaint_handler


def _rule_body(css, selector):
    """Return the declaration block for a selector, with comments stripped.

    Comments are removed first: they legitimately mention z-index values while
    explaining the stacking tiers, and would otherwise be matched instead of the
    real declaration.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    idx = css.index(selector)
    start = css.index("{", idx)
    depth, i = 1, start + 1
    while depth:
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return css[start + 1:i - 1]


def test_editing_dismisses_the_lightbox_before_opening_the_editor():
    """Regression: the editor opened behind the still-open lightbox."""
    handler = LIGHTBOX[
        LIGHTBOX.index("listen(byId('lightbox-edit')"):LIGHTBOX.index("listen(byId('lightbox-duckmotion')")
    ]
    assert "pswp.close()" in handler, (
        "the Edit handler must close PhotoSwipe first, or the editor renders behind the lightbox"
    )
    # Mirror the Inpaint handler, which also dismisses the lightbox first.
    assert "this.currentPswpInstance.pswp.close();" in LIGHTBOX.split(
        "listen(byId('lightbox-inpaint')"
    )[1].split("listen(")[0]


def test_editor_overlay_stacks_above_photoswipe_but_below_modals():
    """Regression: the overlay was z-index 10060, under PhotoSwipe's 100000."""
    pswp_css = (ROOT / "ui" / "lib" / "photoswipe.css").read_text(encoding="utf-8")
    tokens = (ROOT / "ui" / "styles" / "design-tokens.css").read_text(encoding="utf-8")

    pswp_z = int(re.search(r"--pswp-root-z-index:\s*(\d+)", pswp_css).group(1))
    modal_backdrop_z = int(re.search(r"--z-modal-backdrop:\s*(\d+)", tokens).group(1))
    editor_z = int(re.search(r"z-index:\s*(\d+)", _rule_body(CSS, ".image-editor-overlay")).group(1))

    assert editor_z > pswp_z, (
        f"editor z-index {editor_z} is under PhotoSwipe's {pswp_z}; the editor would open behind the lightbox"
    )
    assert editor_z < modal_backdrop_z, (
        f"editor z-index {editor_z} collides with the modal tier at {modal_backdrop_z}"
    )
    # The docked lightbox info panel is hard-coded above the PhotoSwipe root.
    modals = (ROOT / "ui" / "styles" / "components" / "modals.css").read_text(encoding="utf-8")
    docked = [int(m) for m in re.findall(r"z-index:\s*(\d{6,});", modals)]
    assert docked, "expected a hard-coded high z-index for the docked lightbox info panel"
    assert editor_z > max(docked), (
        f"editor z-index {editor_z} is under the docked lightbox info panel {max(docked)}"
    )


def _sized_functions_source() -> str:
    """Slice the real rounding helpers out of ImageEditor.js.

    ``roundTo`` and ``snapPairPreservingAspect`` are contiguous module-level
    functions, so the shipped source can be executed as-is rather than
    reimplemented (a reimplementation would only assert the test's own logic).
    """
    start = EDITOR.index("function roundTo(value, step) {")
    end = EDITOR.index("function readFileAsImage(url) {")
    return EDITOR[start:end]


def test_snap_to_eight_no_longer_rounds_each_axis_independently():
    """Regression: independent rounding stretched the crop.

    ``width`` and ``height`` were each rounded to the nearest multiple of 8, so a
    260x170 crop exported as 264x168 and ``drawImage`` stretched it 2.75% off its
    true aspect ratio.
    """
    body = EDITOR[EDITOR.index("    outputSize() {"):EDITOR.index("    async renderExport(")]
    assert "snapPairPreservingAspect(width, height, crop.width / crop.height)" in body
    assert "width = Math.max(8, roundTo(width, 8))" not in body, (
        "rounding each axis independently distorts the aspect ratio"
    )
    assert "height = Math.max(8, roundTo(height, 8))" not in body
    # The aspect handed to the helper must be the crop's, not the output's.
    assert "crop.width / crop.height" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_snap_pair_keeps_the_aspect_ratio_and_never_worsens_it():
    """Execute the shipped helper and assert the properties the UI depends on."""
    script = f"""
{_sized_functions_source()}
const out = [];
for (let w = 8; w <= 512; w++) {{
  for (let h = 8; h <= 512; h++) {{
    const aspect = w / h;
    const naive = {{ width: Math.max(8, roundTo(w, 8)), height: Math.max(8, roundTo(h, 8)) }};
    const s = snapPairPreservingAspect(w, h, aspect);
    out.push([w, h, s.width, s.height, naive.width, naive.height]);
  }}
}}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert rows, "helper produced no samples"

    for w, h, sw, sh, nw, nh in rows:
        aspect = w / h
        assert sw % 8 == 0 and sh % 8 == 0, f"{w}x{h} -> {sw}x{sh} is not divisible by 8"
        new_err = abs((sw / sh) / aspect - 1)
        naive_err = abs((nw / nh) / aspect - 1)
        assert new_err <= naive_err + 1e-12, (
            f"{w}x{h}: {sw}x{sh} distorts {new_err:.4%}, worse than naive {naive_err:.4%}"
        )

    by_crop = {(w, h): (sw, sh) for w, h, sw, sh, _nw, _nh in rows}
    # Already-valid sizes are left exactly alone (no needless 1024 -> 1016).
    assert by_crop[(512, 512)] == (512, 512)
    # The reported case: 2.75% off before, 0.37% off now.
    assert by_crop[(264, 168)] == (264, 168)
    assert by_crop[(256, 168)] == (256, 168)


def test_flatten_sessions_prefers_per_image_dims_over_run_meta():
    """Regression: every image in a run inherited the run's dimensions.

    ``/gallery`` returns one ``meta.json`` per run, and the edit is stored as an extra
    file in that same run. Flattening stamped ``meta.width``/``meta.height`` onto all of
    them, so the edited pixels were laid out at the *generated* image's aspect ratio and
    rendered stretched in both the tile and the lightbox.
    """
    body = GALLERY[GALLERY.index("    flattenSessions(sessions) {"):]
    body = body[:body.index("\n    }")]
    assert "session.dims || {}" in body, "the per-image dims map is ignored"
    assert "width: Number(own?.[0] || meta.width || 1024)" in body
    assert "height: Number(own?.[1] || meta.height || 1024)" in body
    assert "width: Number(meta.width || 1024)" not in body, (
        "images are back to inheriting the run's dimensions"
    )


def test_save_copy_reports_the_exported_size_to_on_saved():
    """The lightbox must know the saved copy's real size, not the source's."""
    save = EDITOR[EDITOR.index("    async save() {"):]
    save = save[:save.index("\n    }")]
    assert "const exported = this.outputSize();" in save
    assert "this.callbacks.onSaved({ path: newPath, width: exported.width," in save
    # The size has to be read before close() resets the editor state.
    assert save.index("this.outputSize()") < save.index("this.close()")


def test_saved_copy_opens_in_the_lightbox():
    APP = (ROOT / "ui" / "app_main.js").read_text(encoding="utf-8")
    handler = APP[APP.index("function handleEditedImageSaved(saved) {"):]
    handler = handler[:handler.index("\n}\n")]
    # The gallery must be refreshed (and awaited) before the lightbox opens, or the
    # new copy would not be in the grid yet.
    assert "refreshLatest()" in handler
    assert "Promise.resolve(refreshed).then(() => openSavedEditInLightbox(info))" in handler
    assert "function openSavedEditInLightbox(info = {}) {" in APP

    opener = APP[APP.index("function openSavedEditInLightbox(info = {}) {"):]
    opener = opener[:opener.index("\n}\n")]
    # Reuse the gallery tile so the lightbox gets full session metadata and index.
    assert '.image-item[data-src=' in opener
    assert "tile.click();" in opener
    # Normalise the server's rooted path to the web path the tiles use.
    assert "replace(/^\\/+/, '')" in opener
    # Fallback slide must carry the exported size, not a guess.
    assert "width: Number(width) || 1024" in opener
    assert "height: Number(height) || 1024" in opener
    assert "msrc: `/thumbs/${webPath}`" in opener
