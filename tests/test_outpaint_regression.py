"""Behavioural regression tests for the SDXL smart-extend / outpaint path.

These run :meth:`InpaintMode.run` against a stubbed inpaint pipeline, so they
exercise the real staging, mask and composite code instead of grepping source.

Two regressions are covered:

* ``1bcde16`` wrapped ``base_inpaint`` in a closure to inject IP-Adapter FaceID
  kwargs, but every smart-extend conditioning call kept passing that closure as
  ``active_pipe``. The SDXL conditioning builder needs a real pipeline
  (``pipe.device`` / ``pipe.tokenizer``), so each outpaint pass died with
  ``AttributeError: 'function' object has no attribute 'device'``.
* ``modes/inpaint.py`` had a function-local ``from PIL import ImageFilter``,
  which shadowed the module-level import for the whole of ``run()`` and raised
  ``UnboundLocalError`` on the seam-mask blur whenever ``mask_blur`` was 0.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from modes.inpaint import InpaintMode

GENERATED_RGB = (255, 0, 255)


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakeTokenizer:
    """Minimal tokenizer so the native (non-chunked) prompt path is taken."""

    model_max_length = 77
    bos_token_id = 49406
    eos_token_id = 49407

    def __call__(self, text, **kwargs):
        ids = list(range(1, max(2, len(str(text).split()) * 2 + 1)))
        return {"input_ids": [ids]}


class _StubInpaint:
    """Ideal-model stub: every repainted pixel comes back as solid magenta."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = _FakeTokenizer()
        self.tokenizer_2 = _FakeTokenizer()
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.calls = []

    def __call__(self, **kwargs):
        image = kwargs["image"].convert("RGB")
        mask = kwargs.get("mask_image")
        mask = mask.convert("L") if mask is not None else Image.new("L", image.size, 255)
        if mask.size != image.size:
            mask = mask.resize(image.size)
        mask_np = np.array(mask, dtype=np.uint8)
        out = np.array(image, dtype=np.uint8).copy()
        out[mask_np > 127] = np.array(GENERATED_RGB, dtype=np.uint8)
        self.calls.append(
            {
                "size": image.size,
                "white_fraction": float((mask_np > 127).mean()),
                "strength": kwargs.get("strength"),
                "guidance_scale": kwargs.get("guidance_scale"),
                "num_inference_steps": kwargs.get("num_inference_steps"),
            }
        )
        return _FakeResult(Image.fromarray(out, mode="RGB"))


def _source_image(width, height):
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    band = max(8, height // 16)
    for y in range(0, height, band):
        draw.rectangle((0, y, width, min(height, y + band)), fill=((y * 5) % 255, (y * 3) % 255, (200 + y) % 255))
    return image


def _smart_extend_settings(source_size, target_size, auto_step, **overrides):
    from core.backends.sdxl_runtime import _build_smart_extend_inputs

    source = _source_image(*source_size)
    expanded, smart_mask, source_box = _build_smart_extend_inputs(
        source, None, target_size[0], target_size[1], "center", 8, None, None
    )
    settings = {
        "prompt": "a test scene",
        "negative_prompt": "",
        "input_image": expanded,
        "mask_image": smart_mask,
        "width": expanded.size[0],
        "height": expanded.size[1],
        "steps": 8,
        "cfg": 7.5,
        "strength": 0.8,
        "num_images": 1,
        "mask_blur": 8,
        "smart_extend": True,
        "smart_extend_auto_step": auto_step,
        "smart_extend_anchor": "center",
        "smart_extend_feather": 8,
        "smart_extend_refine": True,
        "smart_extend_source_box": source_box,
    }
    settings.update(overrides)
    return settings, source_box


def _run(settings):
    pipe = _StubInpaint()
    images, _seed = InpaintMode().run(
        settings=settings,
        pipe=None,
        img2img=None,
        base_img2img=None,
        base_inpaint=pipe,
        generator=torch.Generator("cpu").manual_seed(1234),
        callback=None,
        faceid_kwargs=None,
    )
    return images, pipe


def _generated_map(image):
    arr = np.array(image.convert("RGB"), dtype=np.int16)
    return (
        (np.abs(arr[:, :, 0] - 255) < 12)
        & (arr[:, :, 1] < 12)
        & (np.abs(arr[:, :, 2] - 255) < 12)
    )


def _assert_band_generated(image, source_box):
    """Every outpainted band must contain model output, not the blurry initializer."""
    generated = _generated_map(image)
    x, y, w, h = source_box["x"], source_box["y"], source_box["w"], source_box["h"]
    bands = []
    if x > 0:
        bands.append(("left", generated[y:y + h, 0:x]))
    if x + w < image.size[0]:
        bands.append(("right", generated[y:y + h, x + w:]))
    if y > 0:
        bands.append(("top", generated[0:y, x:x + w]))
    if y + h < image.size[1]:
        bands.append(("bottom", generated[y + h:, x:x + w]))
    assert bands, "test case did not actually outpaint anything"
    for name, band in bands:
        assert band.mean() > 0.90, f"{name} band is not generated (mean={band.mean():.3f})"


@pytest.mark.parametrize(
    "name,source_size,target_size,auto_step",
    [
        ("repeat_1024_to_1536", (1024, 1024), (1536, 1024), False),
        ("auto_step_1024_to_1536", (1024, 1024), (1536, 1024), True),
        ("repeat_chunked_512_to_1536", (512, 512), (1536, 1536), False),
    ],
)
def test_outpaint_actually_generates_new_pixels(name, source_size, target_size, auto_step):
    settings, source_box = _smart_extend_settings(source_size, target_size, auto_step)
    images, pipe = _run(settings)

    assert len(images) == 1
    assert images[0].size == (target_size[0], target_size[1])
    assert len(pipe.calls) >= 1
    _assert_band_generated(images[0], source_box)


def test_outpaint_survives_zero_mask_blur():
    """``mask_blur=0`` used to raise UnboundLocalError on the seam-mask blur."""
    settings, source_box = _smart_extend_settings((1024, 1024), (1536, 1024), False, mask_blur=0)
    images, _pipe = _run(settings)
    _assert_band_generated(images[0], source_box)


def test_outpaint_conditions_against_the_pipeline_not_the_faceid_wrapper():
    """The conditioning builder must never receive the faceid wrapper closure."""
    settings, _source_box = _smart_extend_settings((1024, 1024), (1536, 1024), False)
    images, _pipe = _run(settings)
    assert len(images) == 1

    # If the wrapper had been passed, build_sdxl_conditioning would have raised
    # before any pass ran. Reaching a finished image is the contract; assert the
    # guard explicitly so the failure mode stays obvious.
    from prompt.conditioning import inject_prompt_conditioning_kwargs

    def _explode(*args, **kwargs):
        raise AssertionError("conditioning builder must not run for short prompts")

    kwargs = inject_prompt_conditioning_kwargs(
        {"prompt": "a test scene", "negative_prompt": ""},
        active_pipe=_StubInpaint(),
        embeddings_active=False,
        builder=_explode,
        cache={},
    )
    assert "prompt" in kwargs


def test_outpaint_debug_log_records_the_step_count_each_pass_used(monkeypatch):
    """A blurry outpaint once went completely untraceable because the debug log
    recorded strength and guidance but never the denoising step count, so a
    checkpoint that inherited ``steps=4`` from another architecture produced mush
    with nothing in the log to explain it. ``num_inference_steps`` must now be
    visible both in the log and on the pipeline calls themselves.
    """
    settings, _source_box = _smart_extend_settings(
        (1024, 1024), (1536, 1024), False, steps=27
    )
    debug_root = Path(tempfile.mkdtemp(prefix="webbduck-debug-"))
    monkeypatch.setenv("WEBBDUCK_SMART_EXTEND_DEBUG_DIR", str(debug_root))
    try:
        _run(settings)
        metas = sorted(debug_root.glob("*/meta.json"))
        assert metas, "smart extend wrote no debug session"
        events = json.loads(metas[0].read_text(encoding="utf-8"))["events"]
        composites = [e for e in events if e.get("type") == "repeat_full_composite"]
        strategy = [e for e in events if e.get("type") == "repeat_mode_strategy"]
        assert composites, "no repeat composite events were logged"
        assert strategy, "no repeat strategy event was logged"
        assert strategy[0]["num_inference_steps"] == 27
        for event in composites:
            assert event["num_inference_steps"] == 27
    finally:
        shutil.rmtree(debug_root, ignore_errors=True)


def test_every_outpaint_pass_uses_the_configured_step_count():
    settings, _source_box = _smart_extend_settings(
        (1024, 1024), (1536, 1024), False, steps=23
    )
    _images, pipe = _run(settings)
    assert pipe.calls, "the pipeline was never invoked"
    for call in pipe.calls:
        assert call["num_inference_steps"] == 23


def test_ui_default_for_auto_step_matches_the_backend_default():
    """The UI always posts ``smart_extend_auto_step`` explicitly, so the backend's
    ``True`` default never applies. The checkbox used to ship unchecked, which
    silently forced every outpaint onto the single-pass repeat path: one inpaint
    over a mask covering most of the canvas, starting from a blurred magnified
    copy of the input. That reads as "blurry" and "it just made the same image
    again". The staged path grows the canvas in small increments at a higher
    guidance scale, so the two defaults must agree.
    """
    import re

    html = (Path(__file__).resolve().parents[1] / "ui" / "index.html").read_text(
        encoding="utf-8"
    )
    state = (Path(__file__).resolve().parents[1] / "ui" / "core" / "state.js").read_text(
        encoding="utf-8"
    )
    mode = (Path(__file__).resolve().parents[1] / "modes" / "inpaint.py").read_text(
        encoding="utf-8"
    )

    backend_default = re.search(
        r'smart_extend_auto_step",\s*(True|False)\)', mode
    )
    assert backend_default, "backend auto_step default not found"
    assert backend_default.group(1) == "True", (
        "this test encodes that the staged path is the intended default"
    )

    checkbox = re.search(r'<input[^>]*id="smart-extend-auto-step"[^>]*>', html)
    assert checkbox, "auto-step checkbox not found in ui/index.html"
    assert "checked" in checkbox.group(0), (
        "ui/index.html ships the auto-step checkbox unchecked, which overrides "
        "the backend default and forces the repeat path"
    )

    persisted = re.search(r"smartExtendAutoStep:\s*(true|false)", state)
    assert persisted, "smartExtendAutoStep default not found in ui/core/state.js"
    assert persisted.group(1) == "true", (
        "ui/core/state.js must persist auto-step as enabled to match the backend"
    )


def test_auto_step_is_enabled_at_every_layer_of_the_outpaint_chain():
    """Auto-step must default to on at every layer that can override it.

    The toggle appeared to do nothing because three separate layers each held their
    own ``false``: the server Form default, the UI's non-advanced "fixed" block, and
    the persisted localStorage value. Any one of them silently forces the single-pass
    repeat path regardless of what the checkbox shows.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    server = (root / "server" / "app.py").read_text(encoding="utf-8")
    main = (root / "ui" / "app_main.js").read_text(encoding="utf-8")
    state = (root / "ui" / "core" / "state.js").read_text(encoding="utf-8")

    server_default = re.search(
        r"SMART_EXTEND_AUTO_STEP_DEFAULT\s*=\s*(True|False)", server
    )
    assert server_default, "server auto-step default constant not found"
    assert server_default.group(1) == "True", (
        "server SMART_EXTEND_AUTO_STEP_DEFAULT must be True; this is the Form "
        "default used whenever the client omits the field"
    )

    fixed = re.search(
        r"const SMART_EXTEND_FIXED = Object\.freeze\(\{(.*?)\}\)", main, re.S
    )
    assert fixed, "SMART_EXTEND_FIXED block not found in ui/app_main.js"
    fixed_auto = re.search(r"autoStep:\s*(true|false)", fixed.group(1))
    assert fixed_auto, "autoStep not found in SMART_EXTEND_FIXED"
    assert fixed_auto.group(1) == "true", (
        "ui/app_main.js SMART_EXTEND_FIXED.autoStep is used for the whole "
        "non-advanced path, so it must default to true"
    )

    # A stale persisted false must be migrated, otherwise the default never
    # reaches an existing browser session.
    assert re.search(r"smartExtendDefaultsVersion:\s*\d+", state), (
        "state.js needs a smartExtendDefaultsVersion so changed defaults can be "
        "migrated into existing sessions"
    )
    assert re.search(r"parsed\.smartExtendDefaultsVersion", state), (
        "state.js must read smartExtendDefaultsVersion from persisted state"
    )
    assert re.search(
        r"state\.smartExtendAutoStep\s*=\s*DEFAULT_STATE\.smartExtendAutoStep", state
    ), "state.js must migrate a stale persisted smartExtendAutoStep to the new default"


def _gradient_source(w=1024, h=1024):
    src = Image.new("RGB", (w, h))
    px = src.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    return src


def _band_stats(band):
    """Row-to-row difference (texture) and column-correlation of a band."""
    b = band.astype(np.float32)
    row_step = float(np.abs(np.diff(b, axis=0)).mean()) if b.shape[0] > 1 else 0.0
    col_prof = b.mean(axis=(0, 2))
    col_step = float(np.abs(np.diff(col_prof)).mean()) if col_prof.size > 1 else 0.0
    return row_step, col_step


def test_smart_extend_canvas_default_initializer_is_structureless():
    """New canvas area must be seeded without fabricated structure by default.

    Inpainting treats everything outside the mask as ground-truth context, so the
    model faithfully continues whatever we put there. Observed failure sequence:

    * ``soft_context`` (scale-to-cover ``ImageOps.fit``) left a blurred,
      multi-magnified phantom beside the real source -> unrelated scene.
    * ``edge_strips`` (stretch a 1px border) left a constant row repeat ->
      vertical smearing that reads as "the image is stretched".

    The default must therefore be structureless noise fill.
    """
    from core.image_operations import build_smart_extend_inputs

    src = _gradient_source()
    canvas, mask, box = build_smart_extend_inputs(src, 1328, 2048, "center", 12, 139, 0)

    assert canvas.size == (1328, 2048)
    assert box == {"x": 139, "y": 0, "w": 1024, "h": 1024}

    arr = np.asarray(canvas.convert("RGB"), dtype=np.int16)
    x0, x1 = box["x"], box["x"] + box["w"]
    src_arr = np.asarray(src, dtype=np.int16)

    # Source pixels must survive bit-exact.
    assert np.array_equal(arr[0:1024, x0:x1], src_arr)

    # The source interior must stay unmasked. A small feather bleed of 255 at the very
# seam edges is intentional, so assert on the core rather than the full row.
    mask_arr = np.asarray(mask)
    assert mask_arr[512, x0 + 32 : x1 - 32].max() == 0
    # The added area below the source must be fully masked.
    assert mask_arr[1200, x0:x1].min() == 255

    band = arr[1024:, x0:x1]
    assert band.shape[0] > 0

    # A stretched 1px border yields near-identical adjacent rows (row_step ~0).
    # Noise fill must keep real per-row variation.
    row_step, _ = _band_stats(band)
    assert row_step > 3.0, (
        f"new area row_step={row_step:.2f} suggests a smeared/stretched context"
    )

    # Noise fill must not correlate with a mirrored copy of the source either.
    mirrored = np.asarray(src.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.int16)
    mirrored_band = np.concatenate([mirrored, mirrored], axis=0)[: band.shape[0], :, :]
    assert np.abs(band - mirrored_band).mean() > 5.0

    # Nor with the old magnified phantom: the band must not be a 2x crop of the source.
    phantom = np.asarray(src.resize((2048, 2048), Image.LANCZOS), dtype=np.int16)
    phantom_band = phantom[1024:, (x0 - 139):(x1 - 139)]
    assert np.abs(band - phantom_band).mean() > 5.0


def test_smart_extend_initializers_are_scale_consistent_and_distinct():
    """Every selectable initializer must stay 1:1 with the pasted source.

    A scale-to-cover context reintroduces a magnified phantom, so it must not be
    selectable as a normal option, and the variants must differ from each other so
    the choice is meaningful.
    """
    from core.image_operations import _seed_canvas

    src = _gradient_source(256, 256)
    box = (100, 0)

    canvases = {}
    for name in ("noise", "mirror", "edge_strips"):
        canvases[name] = _seed_canvas(src, (512, 768), box, name)

    src_arr = np.asarray(src, dtype=np.int16)
    for name, canvas in canvases.items():
        arr = np.asarray(canvas.convert("RGB"), dtype=np.int16)
        assert canvas.size == (512, 768), name
        # Source region bit-exact for all initializers.
        assert np.array_equal(arr[0:256, 100:356], src_arr), name
        # Squeezed whole-image context would correlate strongly with a resized source.
        squeezed = np.asarray(src.resize((512, 768), Image.LANCZOS), dtype=np.int16)
        band = arr[256:, 100:356]
        assert np.abs(band - squeezed[256:, 100:356]).mean() > 5.0, name

    # mirror must actually mirror (differs from edge_strips) and edge_strips must smear.
    a = np.asarray(canvases["mirror"].convert("RGB"), dtype=np.int16)
    b = np.asarray(canvases["edge_strips"].convert("RGB"), dtype=np.int16)
    n = np.asarray(canvases["noise"].convert("RGB"), dtype=np.int16)
    assert np.abs(a[256:, 100:356] - b[256:, 100:356]).mean() > 5.0

    mirror_row, _ = _band_stats(a[256:, 100:356])
    strip_row, _ = _band_stats(b[256:, 100:356])
    noise_row, _ = _band_stats(n[256:, 100:356])
    assert strip_row < mirror_row, "edge_strips should smear more than mirror"
    assert strip_row < noise_row, "edge_strips should smear more than noise fill"


def test_smart_extend_source_box_is_never_stretched_to_target():
    """Regression guard for the original report: expanding must not rescale the source."""
    from core.image_operations import build_smart_extend_inputs

    src = _gradient_source(1024, 1024)
    canvas, _mask, box = build_smart_extend_inputs(src, 1256, 1696, "center", 12, 140, 0)

    # Source keeps its native size; only the canvas grows.
    assert box["w"] == 1024 and box["h"] == 1024
    arr = np.asarray(canvas.convert("RGB"), dtype=np.int16)
    x0 = box["x"]
    assert np.array_equal(arr[box["y"]:box["y"] + 1024, x0:x0 + 1024],
                          np.asarray(src, dtype=np.int16))


def test_effective_refine_steps_never_floors_to_zero():
    """Diffusers runs ``int(num_inference_steps * strength)`` steps.

    Seam-refine passes deliberately use low step counts and low strengths, so the
    product can floor to zero and diffusers raises:
    ``ValueError: After adjusting the num_inference_steps by strength parameter:
    0.16, the number of pipeline steps is 0 which is < 1``.

    Real report: 18 base steps gave 6 refine steps at strength 0.16 -> int(0.96) == 0.
    """
    from modes.inpaint import _effective_refine_steps

    # Reproduce the crashing combination from the bug report.
    assert int(6 * 0.16) == 0, "precondition: this combination floors to zero"
    assert _effective_refine_steps(6, 0.16) >= 7

    # Across the realistic range, the effective step count must stay usable.
    for requested in (4, 6, 8, 9, 10, 12, 18, 20, 30, 40):
        for strength in (0.08, 0.12, 0.16, 0.2, 0.23, 0.28, 0.32, 0.5, 0.75):
            steps = _effective_refine_steps(requested, strength)
            effective = int(steps * strength)
            assert effective >= 1, (
                f"steps={steps} strength={strength} -> effective {effective}"
            )
            assert steps >= requested, "must never reduce the requested step count"

    # Degenerate inputs must not raise or return zero.
    assert _effective_refine_steps(0, 0.16) >= 1
    assert _effective_refine_steps(6, 0.0) >= 1


def test_seam_refine_steps_match_progress_estimator():
    """The estimator must predict the same refine steps the mode actually runs.

    If they disagree, the progress bar drifts from reality, which hides failures
    like the zero-step crash instead of surfacing them.
    """
    from core.backends.sdxl_runtime import estimate_total_steps
    from modes.inpaint import _build_step_fractions, _effective_refine_steps

    settings = {
        "steps": 18,
        "width": 1328,
        "height": 2048,
        "num_images": 1,
        "smart_extend": True,
        "smart_extend_auto_step": True,
        "smart_extend_refine": True,
        "smart_extend_refine_each_step": True,
        "smart_extend_refine_strength": 0.28,
        "smart_extend_step_growth": 1.25,
        "smart_extend_source_box": {"x": 139, "y": 0, "w": 1024, "h": 1024},
    }

    fractions = _build_step_fractions(1024, 1024, 1328, 2048, 1.25)
    passes = len(fractions)
    expected = 0
    for idx, p_next in enumerate(fractions):
        is_final = idx == (passes - 1)
        stage_steps = 18 if is_final else max(12, int(18 * (0.55 + 0.45 * float(p_next))))
        expected += stage_steps
        strength = max(
            0.16 if is_final else 0.08,
            min(0.24 if is_final else 0.16, 0.28 * (0.72 if is_final else 0.58)),
        )
        expected += _effective_refine_steps(
            max(8 if is_final else 6, int(stage_steps * (0.45 if is_final else 0.30))),
            strength,
        )

    assert estimate_total_steps(dict(settings)) == expected
