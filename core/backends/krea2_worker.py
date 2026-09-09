"""Krea worker entry module with v1.2 identity quality fixes layered in."""
from __future__ import annotations

import sys

from core.backends import krea2_worker_impl as _impl
from core.backends.krea2_identity_quality import install_worker_quality_patch
from core.backends.krea2_identity_face import install_identity_face_patch
from core.backends.krea2_identity_reference import install_identity_reference_patch

# Order matters:
# 1) quality patch installs corrected v1.2.4 encode/denoise math;
# 2) face patch wraps that encoder and patches ref_boost attention;
# 3) reference patch sits outermost so it can prepare/crop the reference and
#    pass the prepared face box into the face-aware run wrapper.
install_worker_quality_patch(_impl)
install_identity_face_patch(_impl)
install_identity_reference_patch(_impl)

# Preserve the original module object for monkeypatch-heavy tests and for code
# whose function globals intentionally refer to the worker module itself.
sys.modules[__name__] = _impl
