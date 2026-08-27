"""Core generation logic and pipeline management.

The package root intentionally stays lightweight. Importing a backend helper or
model-routing module must not construct the mature SDXL/storage stack as a side
effect; those objects are resolved lazily only when explicitly requested.
"""

__all__ = ["run_generation", "pipeline_manager"]


def __getattr__(name):
    """Lazy imports for legacy package-level convenience attributes."""
    if name == "run_generation":
        from .generation import run_generation

        return run_generation
    if name == "pipeline_manager":
        from .pipeline import pipeline_manager

        return pipeline_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
