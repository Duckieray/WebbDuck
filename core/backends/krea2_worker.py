"""Krea worker entry module with v1.2 identity quality fixes layered in."""
from __future__ import annotations

import sys

from core.backends import krea2_worker_impl as _impl
from core.backends.krea2_identity_face import install_identity_face_patch
from core.backends.krea2_identity_quality import install_worker_quality_patch
from core.backends.krea2_identity_reference import install_identity_reference_patch
from core.backends.krea2_identity_stability import install_identity_stability_patch

# Order matters:
# 1) quality patch installs corrected v1.2.4 encode/denoise math;
# 2) face patch adds a soft face-region lift while retaining single-source math;
# 3) reference patch canonicalizes oversized references and restores requested
#    artifact size;
# 4) stability patch is outermost: ordinary Persona runs remain SINGLE-reference,
#    preserve the normal global likeness boost, and use the multiview detector
#    without fabricating the model's trained [scene, subject] two-input layout.
install_worker_quality_patch(_impl)
install_identity_face_patch(_impl)
install_identity_reference_patch(_impl)
install_identity_stability_patch(_impl)

# Preserve the original module object for monkeypatch-heavy tests and for code
# whose function globals intentionally refer to the worker module itself.
sys.modules[__name__] = _impl
