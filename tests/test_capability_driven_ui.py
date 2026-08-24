from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API = (ROOT / "ui" / "core" / "api.js").read_text(encoding="utf-8")
CAPS = (ROOT / "ui" / "core" / "modelCapabilities.js").read_text(encoding="utf-8")


def test_model_catalog_is_the_only_model_loading_contract():
    assert "get('/model-catalog')" in API
    assert "get('/models')" not in API
    assert "webbduck:model-catalog" in API
    assert "webbduck:model-profile" in API


def test_capability_controller_never_dispatches_on_architecture_names():
    lowered = CAPS.lower()
    for family in ("sdxl", "flux", "krea", "qwen"):
        assert family not in lowered
    assert "profile.capabilities" in CAPS
    assert "profile.constraints" in CAPS


def test_model_profile_controls_architecture_specific_features():
    for capability in (
        "negative_prompt",
        "second_pass",
        "lora",
        "embeddings",
        "identity_adapter",
        "prompt_2",
        "clip_skip",
        "img2img",
        "inpaint",
        "outpaint",
    ):
        assert capability in CAPS


def test_selected_model_defaults_and_dimension_constraints_drive_studio():
    assert "profile?.constraints?.dimension_multiple" in CAPS
    assert "defaults[field]" in CAPS
    assert "forceDefaults" in CAPS
    assert "input.step = String(multiple)" in CAPS


def test_unsupported_models_and_operations_are_blocked_in_browser():
    assert "option.disabled = profile.supported === false" in CAPS
    assert "validateCurrentOperation" in CAPS
    assert "event.stopImmediatePropagation()" in CAPS
    assert "Selected model runtime is unavailable" in CAPS


def test_visible_product_copy_is_neutralized_by_capability_layer():
    assert "Local Model-Driven Image Studio" in CAPS
    assert "Checkpoint-driven local image generation" in CAPS


def test_studio_only_hides_mature_controls_when_capability_is_explicitly_false():
    assert "function capabilityAllowed(caps, key)" in CAPS
    assert "return caps?.[key] !== false" in CAPS
    assert "restoreLegacyStudioControls" in CAPS
    for section in (
        "section-refiner",
        "section-lora",
        "section-embeddings",
        "section-ip-adapter",
    ):
        assert f"setSectionVisible('{section}', capabilityAllowed" in CAPS


def test_model_selection_refreshes_its_profile_instead_of_relying_on_catalog_timing():
    assert "async function refreshSelectedProfile" in CAPS
    assert "fetch(`/model-catalog/${encodeURIComponent(name)}`" in CAPS
    assert "cache: 'no-store'" in CAPS
    assert "select.addEventListener('change'" in CAPS
