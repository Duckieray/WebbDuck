"""Core generation logic and pipeline management."""

from .generation import run_generation
from .pipeline import pipeline_manager

__all__ = ["run_generation", "pipeline_manager"]