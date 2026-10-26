"""Automatic realism detection so good output is the default, not a setting.

WebbDuck's promise is that it "just works". Photorealistic prompts should get the
photographic pipeline - sharper skin, natural micro-contrast, a refined face -
without the user knowing any of those switches exist. Stylized prompts must not,
because over-sharpening anime, line art and text is the fastest way to ruin them.

Detection is deliberately conservative and explainable. Every decision returns the
signals that produced it, so the UI can show a one-line reason instead of a pile of
toggles, and so a surprising result can be diagnosed rather than guessed at.

Signal priority:

1. An explicit user override (Advanced Settings): ``auto`` | ``always`` | ``never``.
2. Curated model metadata from ``webbduck_meta/`` (``recommended_mode`` / tags).
   This is the authoritative source when a model has been tagged.
3. Prompt scoring against photographic and stylized vocabularies.

The prompt vocabulary reuses ``models.metastore.MODE_SYNONYMS`` so that "photo",
"realistic" and friends mean the same thing here as they do in the gallery filters,
with a photographic-only tier layered on top for terms the filters do not carry
("skin pores", "bokeh", "85mm").
"""

from __future__ import annotations

import re
from typing import Any

# Modes that must never receive the photographic pipeline.
_STYLIZED_MODES = {"anime", "cartoon", "manga", "comic", "furry", "stylized", "pony"}
_REALISTIC_MODES = {"realistic", "photo", "photorealistic", "realism"}

# Photographic cues that carry real signal on their own. Weighted higher than the
# generic taxonomy words because they describe the *medium*, not the subject.
_PHOTO_CUES: dict[str, float] = {
    "photo": 1.0,
    "photograph": 1.0,
    "photographic": 1.0,
    "photorealistic": 1.2,
    "realistic": 0.8,
    "realism": 0.8,
    "dslr": 0.8,
    "dslr photo": 1.2,
    "shot on": 0.7,
    "50mm": 0.6,
    "85mm": 0.6,
    "35mm": 0.6,
    "bokeh": 0.9,
    "depth of field": 0.7,
    "natural lighting": 0.9,
    "golden hour": 0.8,
    "soft lighting": 0.8,
    "studio lighting": 0.8,
    "cinematic lighting": 0.8,
    "film grain": 0.7,
    "grain": 0.4,
    "skin pores": 1.2,
    "pores": 0.9,
    "skin texture": 1.2,
    "subsurface scattering": 1.1,
    "freckles": 0.7,
    "eyelashes": 0.6,
    "catchlight": 0.8,
    "microcontrast": 0.6,
    "hyperrealistic": 1.2,
    "lifelike": 0.9,
    "unedited photo": 1.0,
    "raw photo": 1.0,
    "documentary": 0.5,
}

# Cues that mean "not a photograph". These veto rather than merely outvote, because
# one anime tag is a stronger signal than three photographic words.
_STYLIZED_CUES: dict[str, float] = {
    "anime": 1.4,
    "manga": 1.4,
    "cartoon": 1.2,
    "comic": 1.0,
    "cel shading": 1.2,
    "cel-shaded": 1.2,
    "lineart": 1.2,
    "line art": 1.2,
    "flat color": 0.9,
    "flat colour": 0.9,
    "vector art": 1.1,
    "digital painting": 0.8,
    "illustration": 0.7,
    "waifu": 1.0,
    "hentai": 1.0,
    "pony": 0.8,
    "chibi": 1.0,
    "isometric": 0.8,
    "pixel art": 1.3,
    "lowpoly": 1.1,
    "3d render": 0.9,
    "watercolor": 0.8,
    "oil painting": 0.8,
}

# Neutral artistic terms that must NOT count against realism: plenty of
# photorealistic prompts say "digital painting" or "cinematic" while meaning a photo.
_ARTISTIC_NEUTRAL = {
    "cinematic",
    "portrait",
    "detailed skin",
    "8k",
    "masterpiece",
    "best quality",
    "render",
    "art",
    "artistic",
}

# A negative prompt actively asking for the opposite style is strong evidence.
_NEGATIVE_STYLIZED = {"anime", "cartoon", "illustration", "drawing", "painting", "3d render", "cel shading"}
_NEGATIVE_PHOTO = {"anime", "cartoon", "illustration", "drawing", "painting", "3d", "cel"}

_AUTO_MODE = "auto"
_ALWAYS = "always"
_NEVER = "never"
_VALID_OVERRIDES = {_AUTO_MODE, _ALWAYS, _NEVER}


def _normalize_override(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "on", "yes"}:
        return _ALWAYS
    if raw in {"0", "false", "off", "no", "none"}:
        return _NEVER
    return raw if raw in _VALID_OVERRIDES else _AUTO_MODE


def _normalize(text: Any) -> str:
    """Lowercase a prompt and flatten punctuation to spaces."""
    return re.sub(r"[^a-z0-9']+", " ", str(text or "").lower()).strip()


def _tokens(text: Any) -> set[str]:
    """Word bag, used for the negative-prompt exclusion check."""
    return set(_normalize(text).split())


def _score_cues(text: str, cues: dict[str, float]) -> tuple[float, list[str]]:
    """Score cue matches with word boundaries.

    Matching against the whole normalized string rather than an n-gram bag means a
    cue of any word length works ("depth of field", "studio lighting") without
    generating combinatorial n-grams. Word boundaries stop "art" matching "heart".
    """
    score = 0.0
    hits: list[str] = []
    for cue, weight in cues.items():
        if re.search(rf"\b{re.escape(cue)}\b", text):
            score += weight
            hits.append(cue)
    return score, sorted(hits)


# Cues that settle the question on their own. A prompt asking for anime is an anime
# prompt even if it also says "photorealistic", because applying the photographic
# pipeline to line art is destructive.
_HARD_STYLIZED = {
    "anime",
    "manga",
    "cel shading",
    "cel-shaded",
    "lineart",
    "line art",
    "pixel art",
    "chibi",
}


def _meta_mode(meta: dict[str, Any] | None) -> str:
    """Pull a curated mode out of metastore-shaped metadata, if present."""
    if not isinstance(meta, dict):
        return ""
    candidates: list[str] = []
    recommended = meta.get("recommended_mode")
    if recommended:
        candidates.append(str(recommended))
    for key in ("modes", "families", "tags"):
        value = meta.get(key)
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, (list, tuple, set)):
            candidates.extend(str(v) for v in value)

    for raw in candidates:
        token = str(raw).strip().lower()
        if token in _STYLIZED_MODES:
            return "stylized"
        if token in _REALISTIC_MODES:
            return "realistic"
    return ""


def detect_realism(
    prompt: Any = "",
    negative_prompt: Any = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a request as ``realistic``, ``stylized`` or ``unknown``.

    Returns a dict with ``mode``, ``score``, ``reasons`` and ``confidence`` so the
    decision is inspectable from both the API and the UI.
    """
    override_meta = _meta_mode(meta)
    text = _normalize(prompt)
    for neutral in _ARTISTIC_NEUTRAL:
        text = re.sub(rf"\b{re.escape(neutral)}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    photo_score, photo_hits = _score_cues(text, _PHOTO_CUES)
    style_score, style_hits = _score_cues(text, _STYLIZED_CUES)
    hard_stylized = sorted(h for h in _HARD_STYLIZED if re.search(rf"\b{re.escape(h)}\b", text))

    negative_bag = _tokens(negative_prompt)
    neg_style = len(negative_bag & _NEGATIVE_STYLIZED)
    neg_photo = len(negative_bag & _NEGATIVE_PHOTO)

    reasons: list[str] = []
    if override_meta:
        reasons.append(f"model metadata: {override_meta}")
    if photo_hits:
        reasons.append("photographic terms: " + ", ".join(photo_hits[:6]))
    if style_hits:
        reasons.append("stylized terms: " + ", ".join(style_hits[:6]))
    if hard_stylized:
        reasons.append("explicitly stylized: " + ", ".join(hard_stylized[:4]))
    if neg_style:
        reasons.append(f"negative prompt excludes {neg_style} stylized tag(s)")
    if neg_photo:
        reasons.append(f"negative prompt excludes {neg_photo} photographic tag(s)")

    # Curated metadata wins outright: a model tagged "anime" is an anime model even
    # when the prompt happens to contain the word "realistic".
    if override_meta == "stylized":
        return {
            "mode": "stylized",
            "score": 1.0,
            "confidence": "high",
            "reasons": reasons,
        }
    if override_meta == "realistic" and style_score < 1.0:
        return {
            "mode": "realistic",
            "score": 1.0,
            "confidence": "high",
            "reasons": reasons,
        }

    # An explicit stylized style settles it, whatever else the prompt says.
    if hard_stylized:
        return {
            "mode": "stylized",
            "score": round(style_score, 3),
            "confidence": "high",
            "reasons": reasons,
        }

    # Otherwise a clear stylized majority outweighs photographic wording.
    if style_score >= 1.0 and style_score >= photo_score:
        return {
            "mode": "stylized",
            "score": round(style_score, 3),
            "confidence": "high" if style_score >= 1.4 else "medium",
            "reasons": reasons,
        }

    if photo_score >= 0.8:
        margin = photo_score - style_score
        return {
            "mode": "realistic",
            "score": round(photo_score, 3),
            "confidence": "high" if margin >= 1.0 else "medium",
            "reasons": reasons,
        }

    if photo_score > 0.0:
        return {
            "mode": "unknown",
            "score": round(photo_score, 3),
            "confidence": "low",
            "reasons": reasons,
        }

    return {"mode": "unknown", "score": 0.0, "confidence": "low", "reasons": reasons}


def realism_from_metadata(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Read the optional Advanced Settings override out of a request.

    Returns ``None`` when absent so callers can distinguish "user said nothing"
    from "user said auto".
    """
    for key in ("realism_mode", "realism_override", "realism"):
        if key in settings and settings.get(key) is not None:
            return {"override": _normalize_override(settings.get(key)), "source": key}
    return None


def resolve_realism(
    settings: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the effective realism decision for one request.

    An explicit override always wins. Otherwise the decision is automatic, and the
    result always explains itself via ``reason``/``reasons``.
    """
    override_info = realism_from_metadata(settings)
    override = override_info["override"] if override_info else _AUTO_MODE

    detection = detect_realism(
        settings.get("prompt"),
        settings.get("negative_prompt"),
        meta,
    )

    if override == _ALWAYS:
        mode, source = "realistic", "override"
    elif override == _NEVER:
        mode, source = "stylized", "override"
    else:
        mode, source = detection["mode"], "auto"

    apply_realism = mode == "realistic"

    # Explain the effective outcome in one line, for the UI status line.
    if source == "override":
        reason = f"{mode} (set in Advanced Settings)"
    elif detection["reasons"]:
        reason = f"{mode} - " + detection["reasons"][0]
    else:
        reason = f"{mode} - no strong realism signals in the prompt"

    return {
        "mode": mode,
        "apply": apply_realism,
        "source": source,
        "override": override,
        "reason": reason,
        "reasons": detection["reasons"],
        "score": detection["score"],
        "confidence": detection["confidence"],
    }