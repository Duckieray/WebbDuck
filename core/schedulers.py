"""Scheduler names and lazy Diffusers construction.

The WebbDuck host needs scheduler names for the public UI but must not import
Diffusers merely to render that list. SDXL workers call ``create_scheduler`` and
load the implementation classes only inside the isolated runtime.
"""

from __future__ import annotations


SCHEDULERS = {
    "Euler a": "EulerAncestralDiscreteScheduler",
    "Euler": "EulerDiscreteScheduler",
    "DPM++ 2M Karras": "DPMSolverMultistepScheduler",
    "DPM++ SDE Karras": "DPMSolverSinglestepScheduler",
    "DPM++ 3M SDE": "DPMSolverMultistepScheduler",
    "DDIM": "DDIMScheduler",
    "UniPC": "UniPCMultistepScheduler",
}


def _scheduler_classes():
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    return {
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
        "DDIMScheduler": DDIMScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
    }


def get_scheduler_class(name):
    classes = _scheduler_classes()
    class_name = SCHEDULERS.get(name, SCHEDULERS["UniPC"])
    return classes[class_name]


def create_scheduler(name, config):
    """Create scheduler instance with proper configuration."""
    cls = get_scheduler_class(name)

    kwargs = {}
    if "Karras" in name:
        kwargs["use_karras_sigmas"] = True
    if "DPM++ SDE" in name:
        kwargs["algorithm_type"] = "sde-dpmsolver++"
    if name == "DPM++ 3M SDE":
        kwargs["solver_order"] = 3

    try:
        return cls.from_config(config, **kwargs)
    except Exception:
        return cls.from_config(config)
