"""Internal forwarding seam for SDXL generation code.

The implementation now lives under ``core.backends.sdxl_runtime``. Generic
model routing must not import this module.
"""

from core.backends.sdxl_runtime import *  # noqa: F401,F403
