"""Scheduler configurations."""

from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)

SCHEDULERS = {
    "Euler a": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
    "DDIM": DDIMScheduler,
    # "UniPC": UniPCMultistepScheduler, # Use UniPC via custom config if needed, often default
}

def get_scheduler_class(name):
    return SCHEDULERS.get(name, UniPCMultistepScheduler)

def create_scheduler(name, config):
    """Create scheduler instance with proper configuration."""
    cls = get_scheduler_class(name)
    
    # Common config adjustments
    kwargs = {}
    
    if "Karras" in name:
        kwargs["use_karras_sigmas"] = True
        
    if "DPM++ SDE" in name:
        kwargs["algorithm_type"] = "sde-dpmsolver++"
        
    try:
        scheduler = cls.from_config(config, **kwargs)
    except Exception:
        # Fallback without kwargs if fails (e.g. some schedulers might not support all args)
        scheduler = cls.from_config(config)
        
    return scheduler
