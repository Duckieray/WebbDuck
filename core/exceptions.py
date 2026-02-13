"""Shared runtime exceptions for generation flow."""


class GenerationCancelledError(RuntimeError):
    """Raised when a running generation job is cancelled by user request."""

