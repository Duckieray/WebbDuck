"""Core generation logic and pipeline management."""

from .pipeline import pipeline_manager

# Note: run_generation imported lazily to avoid circular import with modes
__all__ = ["run_generation", "pipeline_manager"]


def __getattr__(name):
    """Lazy import for run_generation to avoid circular import."""
    if name == "run_generation":
        from .generation import run_generation
        return run_generation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")