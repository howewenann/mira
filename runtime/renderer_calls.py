"""Small helpers for optional renderer callbacks."""

from __future__ import annotations

from typing import Any


def call_renderer(renderer: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """Call an optional renderer method."""
    callback = getattr(renderer, method, None)
    if callback is None:
        return False
    callback(*args, **kwargs)
    return True


__all__ = ["call_renderer"]
