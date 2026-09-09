"""Krea worker entry module with v1.2 identity quality fixes layered in."""
from __future__ import annotations

import sys

from core.backends import krea2_worker_impl as _impl
from core.backends.krea2_identity_quality import install_worker_quality_patch
from core.backends.krea2_identity_reference import install_identity_reference_patch

install_worker_quality_patch(_impl)
install_identity_reference_patch(_impl)

# Preserve the original module object for monkeypatch-heavy tests and for code
# whose function globals intentionally refer to the worker module itself.
sys.modules[__name__] = _impl
