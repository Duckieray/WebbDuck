from pathlib import Path


def test_model_switch_preserves_existing_dimensions_but_keeps_model_defaults_for_sampling():
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "core"
        / "modelCapabilities.js"
    ).read_text(encoding="utf-8")

    assert "function snapDimensionInput(input, multiple)" in source
    assert "Resolution is user intent" in source
    assert "(field === 'width' || field === 'height') && Number(input.value) > 0" in source

    # Model changes may still force checkpoint-specific sampling defaults. This
    # is intentional for contracts such as Krea Raw (28/CFG) vs Turbo (8/no CFG).
    assert "refreshSelectedProfile({ forceDefaults: true })" in source
    assert "['steps', 'steps', 'steps-value']" in source
    assert "['cfg', 'cfg', 'cfg-value']" in source


def test_unsupported_krea_features_remain_capability_gated():
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "core"
        / "modelCapabilities.js"
    ).read_text(encoding="utf-8")

    assert "setSectionVisible('section-lora', capabilityAllowed(caps, 'lora'))" in source
    assert "setHidden(byId('smart-extend-group'), !capabilityAllowed(caps, 'outpaint'))" in source
