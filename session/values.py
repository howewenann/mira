"""Small normalization helpers shared by durable session artifacts."""

from __future__ import annotations

from typing import Any


def bounded_iterations(value: Any) -> int:
    """Return a persisted rubric iteration cap within the supported range."""
    if isinstance(value, bool):
        return 3
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 3
    return parsed if 1 <= parsed <= 20 else 3
