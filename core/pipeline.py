"""Internal forwarding seam for SDXL pipeline code.

The implementation now lives under ``core.backends.sdxl_pipeline``.  This module
exists only until the final server/API rewrite removes the remaining direct
imports from health, idle-unload, and captioning housekeeping.
"""

from core.backends.sdxl_pipeline import *  # noqa: F401,F403
