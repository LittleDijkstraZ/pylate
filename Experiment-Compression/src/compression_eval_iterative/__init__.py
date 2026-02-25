"""Iterative compression evaluation package."""

__all__ = ["main"]


def __getattr__(name):
    """Lazy import to avoid requiring hydra at import time."""
    if name == "main":
        from .cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
