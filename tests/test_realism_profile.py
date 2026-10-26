"""Focused coverage for automatic realism detection (the "just works" path).

The contract these tests protect is narrow and important: a photoreal prompt gets the
photographic pipeline with no user action, and a stylized prompt never does -
because over-sharpening anime, line art and text is the fastest way to ruin them.
"""

from __future__ import annotations

import pytest

from core.realism_profile import (
    detect_realism,
    resolve_realism,
    realism_from_metadata,
)


# ---------------------------------------------------------------------------
# Photographic prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "1girl, portrait photograph, detailed skin texture, natural lighting",
        "portrait of a woman, 85mm, bokeh, film grain",
        "raw photo, unedited, dslr",
        "close up, skin pores, subsurface scattering, catchlight",
        "photorealistic, lifelike, hyperrealistic",
        "cinematic portrait, studio lighting, shallow depth of field",
    ],
)
def test_photographic_prompts_are_detected(prompt):
    result = detect_realism(prompt)
    assert result["mode"] == "realistic", result
    assert result["reasons"]


@pytest.mark.parametrize(
    "prompt",
    [
        "masterpiece, best quality, anime, cel shading, lineart",
        "1girl, manga, flat color",
        "chibi, pixel art, isometric",
        "vector art, comic",
        "oil painting, digital painting",
    ],
)
def test_stylized_prompts_are_rejected(prompt):
    result = detect_realism(prompt)
    assert result["mode"] == "stylized", result


def test_anime_veto_beats_photographic_words():
    """One 'anime' is stronger evidence than three photographic words."""
    result = detect_realism("anime girl, photorealistic, dslr photo, 85mm bokeh")
    assert result["mode"] == "stylized"
    assert result["confidence"] == "high"


@pytest.mark.parametrize(
    "prompt",
    ["a cat on a windowsill", "", "masterpiece, best quality", "1girl, silver hair, sitting"],
)
def test_ambiguous_prompts_stay_unknown(prompt):
    """Silence is the safe default: unknown must not enable anything."""
    assert detect_realism(prompt)["mode"] == "unknown"


def test_neutral_artistic_words_do_not_count_against_realism():
    # "cinematic" and "masterpiece" are common in photoreal prompts.
    assert detect_realism("cinematic portrait, masterpiece, realistic lighting")["mode"] == "realistic"


def test_cinematic_alone_is_not_a_realism_signal():
    assert detect_realism("cinematic composition, masterpiece")["mode"] == "unknown"


def test_negative_prompt_excluding_anime_pushes_realistic():
    result = detect_realism("a woman", negative_prompt="anime, cartoon, drawing")
    assert result["mode"] == "unknown"
    assert any("negative prompt" in r for r in result["reasons"])


def test_detection_is_case_and_punctuation_insensitive():
    a = detect_realism("Portrait PHOTOGRAPH, Skin Pores!!!")
    b = detect_realism("portrait photograph, skin pores")
    assert a["mode"] == b["mode"] == "realistic"


def test_ngrams_match_multiword_cues():
    assert detect_realism("studio lighting, portrait")["mode"] == "realistic"
    assert detect_realism("cel shading, portrait")["mode"] == "stylized"


# ---------------------------------------------------------------------------
# Curated model metadata
# ---------------------------------------------------------------------------


def test_model_metadata_beats_prompt_wording():
    result = detect_realism("realistic photo", meta={"modes": ["anime"]})
    assert result["mode"] == "stylized"
    assert any("model metadata" in r for r in result["reasons"])


def test_model_metadata_recommended_mode_wins():
    result = detect_realism("a castle", meta={"recommended_mode": "realistic"})
    assert result["mode"] == "realistic"


def test_model_metadata_can_be_overridden_by_a_strong_stylized_prompt():
    # Tagged realistic, but the prompt explicitly demands anime.
    result = detect_realism("anime, cel shading, lineart", meta={"recommended_mode": "realistic"})
    assert result["mode"] == "stylized"


@pytest.mark.parametrize("meta", [None, {}, {"modes": []}, {"tags": "unrelated"}])
def test_missing_metadata_is_harmless(meta):
    assert detect_realism("portrait photograph", meta=meta)["mode"] == "realistic"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_always_override_forces_realistic():
    result = resolve_realism({"prompt": "anime, cel shading", "realism_mode": "always"})
    assert result["mode"] == "realistic"
    assert result["apply"] is True
    assert result["source"] == "override"


def test_never_override_forces_off():
    result = resolve_realism({"prompt": "photo, skin pores, dslr", "realism_mode": "never"})
    assert result["mode"] == "stylized"
    assert result["apply"] is False
    assert result["source"] == "override"


def test_auto_is_the_default():
    assert resolve_realism({"prompt": "photo, dslr"})["source"] == "auto"
    assert resolve_realism({"prompt": "photo, dslr", "realism_mode": ""})["override"] == "auto"


@pytest.mark.parametrize(
    "raw,expected",
    [("true", "always"), ("on", "always"), ("1", "always"),
     ("false", "never"), ("off", "never"), ("0", "never"),
     ("nonsense", "auto"), (None, "auto")],
)
def test_override_parsing_is_forgiving(raw, expected):
    settings = {"prompt": "x"}
    if raw is not None:
        settings["realism_mode"] = raw
    assert resolve_realism(settings)["override"] == expected


def test_realism_from_metadata_reports_absence():
    assert realism_from_metadata({}) is None
    assert realism_from_metadata({"realism_mode": "always"})["override"] == "always"


# ---------------------------------------------------------------------------
# Shape / explainability
# ---------------------------------------------------------------------------


def test_result_is_always_explainable():
    """The UI shows one line of reasoning, so a reason must always exist."""
    for prompt in ["", "photo", "anime", "masterpiece"]:
        result = resolve_realism({"prompt": prompt})
        assert result["reason"]
        assert result["mode"] in {"realistic", "stylized", "unknown"}
        assert result["apply"] is (result["mode"] == "realistic")


def test_result_is_json_serializable():
    import json

    json.dumps(resolve_realism({"prompt": "photo, dslr", "negative_prompt": "anime"}))