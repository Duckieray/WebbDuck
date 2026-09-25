"""Static UI contract checks for gallery image editing.

These are headless contract assertions (no running server, no browser), matching
the style of ``tests/test_capability_driven_ui.py``. They guard the wiring that a
browser test would otherwise be needed for: module registration, API wrappers,
the lightbox/gallery entry points, and the markup the JS binds to by id.
"""

from html.parser import HTMLParser
from pathlib import Path
import re


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
    for element_id in ("lightbox-edit", "lightbox-input-image"):
        assert element_id in INDEX
    # Both start hidden and are revealed per-slide, since they only make sense for
    # images that actually live under the outputs directory.
    for element_id in ("lightbox-edit", "lightbox-input-image"):
        assert "display:none" in _tag_of(INDEX, element_id)
    assert "onEdit" in LIGHTBOX
    assert "onUseAsInput" in LIGHTBOX
    assert "includes('/outputs/')" in LIGHTBOX


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
    """Free crop plus the common ratios the user guide advertises."""
    for ratio in ("1:1", "3:4", "4:5", "16:9", "9:16"):
        assert ratio in EDITOR, f"editor is missing the {ratio} aspect preset"


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
