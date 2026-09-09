"""Krea worker entry module with v1.2 identity quality fixes layered in."""
from __future__ import annotations

import sys

from core.backends import krea2_worker_impl as _impl
from core.backends.krea2_identity_multiref import install_identity_multiref_patch
from core.backends.krea2_identity_quality import install_worker_quality_patch
from core.backends.krea2_identity_reference import install_identity_reference_patch

# Order matters:
# 1) quality patch installs corrected v1.2.4 encode/denoise math;
# 2) reference patch canonicalizes oversized primary references and restores the
#    requested artifact size without changing native denoise safety;
# 3) multi-reference patch sits outermost.  It derives a dedicated subject/face
#    anchor from the ORIGINAL Persona image, disables the older destructive
#    primary face crop, and feeds [full reference | subject anchor | target]
#    through independent source frames.
install_worker_quality_patch(_impl)
install_identity_reference_patch(_impl)
install_identity_multiref_patch(_impl)

# Preserve the original module object for monkeypatch-heavy tests and for code
# whose function globals intentionally refer to the worker module itself.
sys.modules[__name__] = _impl
