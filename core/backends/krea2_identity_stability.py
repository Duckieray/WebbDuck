"""Stability policy for single-Persona Krea2 identity runs.

The v1.2 LoRA supports two references, but its trained two-input contract is
``[scene, subject]``.  Expanding one Persona photo into ``[full person, cropped
person]`` is *not* that contract and can occasionally drive the denoise off
manifold (posterized / black / saturated blob outputs).

For ordinary WebbDuck Persona generation, keep one appearance reference and use
face-localized attention on that same source instead:

* full reference retains the normal global ref_boost (normally 4x);
* detected face tokens may receive a modest extra boost (normally 6x);
* the primary reference is not destructively face-cropped by default;
* grounding remains at the trained/default 768px unless the user explicitly
  requests something else.

True scene+subject multi-reference can be reintroduced later when WebbDuck has a
real second reference input rather than deriving one from the Persona itself.
"""
from __future__ import annotations

import copy
from typing import Any


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _stable_identity_request(request: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(request)
    raw = tuned.get("identity")
    if not isinstance(raw, dict):
        return tuned

    identity = copy.deepcopy(raw)
    global_boost = max(1.0, min(10.0, _as_float(identity.get("ref_boost"), 4.0)))

    # A single Persona image is a single-reference restaging task.  Do not turn
    # it into the model's scene+subject training layout by duplicating the same
    # person as a second source block.
    identity["multi_reference_identity"] = False

    # Preserve whole-person identity at the ordinary ref_boost while allowing a
    # soft face-region lift.  The previous 2x background / 6x face split weakened
    # the rest of the source too much and encouraged identity drift.
    identity.setdefault("background_ref_boost", global_boost)
    identity.setdefault("face_ref_boost", min(10.0, global_boost + 2.0))

    # Keep the full prepared Persona as the source.  Face detection is still
    # used to build the token-space emphasis mask, but cropping the primary ref
    # changes the appearance/context distribution and is unnecessary here.
    identity.setdefault("auto_face_crop", False)

    # The current LoRA's grounded encoder was trained with 384-768px jitter.
    # Leave explicit user values alone, but never auto-promote Persona grounding
    # above the normal 768px default.
    if identity.get("grounding_px") is None:
        identity["grounding_px"] = 768

    identity["stability_recipe"] = "single-ref-global4-faceplus2"
    tuned["identity"] = identity
    return tuned


def install_identity_stability_patch(base: Any) -> None:
    """Install single-reference Persona stability as the outermost run wrapper."""
    if bool(getattr(base, "_krea_identity_stability_patch", False)):
        return

    # Upgrade the reference detector to the already-shipped multi-view detector
    # (frontal/profile/mirrored-profile) without enabling multi-reference latent
    # conditioning itself.  This helps 3/4 Persona shots receive face emphasis.
    try:
        from core.backends import krea2_identity_reference as reference
        from core.backends.krea2_identity_multiref import _detect_face as detect_multiview

        reference.detect_face = detect_multiview
    except Exception:
        pass

    original_run = base._run_identity

    def _run_stable(request: dict[str, Any], *args: Any, **kwargs: Any):
        tuned = _stable_identity_request(request)
        result = original_run(tuned, *args, **kwargs)
        if isinstance(result, dict):
            runtime = result.setdefault("runtime", {})
            if isinstance(runtime, dict):
                identity_runtime = runtime.setdefault("identity", {})
                if isinstance(identity_runtime, dict):
                    cfg = tuned.get("identity") or {}
                    identity_runtime["stability"] = {
                        "mode": "single-reference-persona",
                        "global_ref_boost": _as_float(cfg.get("background_ref_boost"), 4.0),
                        "face_ref_boost": _as_float(cfg.get("face_ref_boost"), 6.0),
                        "auto_face_crop": bool(cfg.get("auto_face_crop", False)),
                        "grounding_px": int(cfg.get("grounding_px") or 768),
                        "derived_multiref": False,
                    }
        return result

    base._run_identity = _run_stable
    base._stable_krea_identity_request = _stable_identity_request
    base._krea_identity_stability_patch = True
