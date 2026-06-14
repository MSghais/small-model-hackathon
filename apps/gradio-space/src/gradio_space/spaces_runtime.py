"""Hugging Face Spaces ZeroGPU helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def is_hf_gradio_runtime() -> bool:
    """True on Hugging Face Gradio SDK Spaces (skip startup model preload)."""
    try:
        import spaces  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("SPACE_ID"))


def gpu_task(
    *,
    duration: int = 180,
    size: str = "large",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply @spaces.GPU when the HF spaces runtime is present (no-op elsewhere)."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        try:
            import spaces

            return spaces.GPU(duration=duration, size=size)(fn)
        except ImportError:
            return fn

    return decorator
