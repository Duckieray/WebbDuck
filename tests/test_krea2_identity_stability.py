from __future__ import annotations

from core.backends.krea2_identity_stability import _stable_identity_request


def test_persona_stability_keeps_single_reference_and_global_likeness():
    request = {
        "identity": {
            "ref_boost": 4.0,
            "grounding_px": 768,
        }
    }
    tuned = _stable_identity_request(request)
    identity = tuned["identity"]

    assert identity["multi_reference_identity"] is False
    assert identity["background_ref_boost"] == 4.0
    assert identity["face_ref_boost"] == 6.0
    assert identity["auto_face_crop"] is False
    assert identity["grounding_px"] == 768


def test_persona_stability_preserves_explicit_face_and_grounding_overrides():
    request = {
        "identity": {
            "ref_boost": 5.0,
            "background_ref_boost": 5.5,
            "face_ref_boost": 7.25,
            "auto_face_crop": True,
            "grounding_px": 640,
        }
    }
    tuned = _stable_identity_request(request)
    identity = tuned["identity"]

    assert identity["multi_reference_identity"] is False
    assert identity["background_ref_boost"] == 5.5
    assert identity["face_ref_boost"] == 7.25
    assert identity["auto_face_crop"] is True
    assert identity["grounding_px"] == 640
