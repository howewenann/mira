"""Small normalization helpers shared by durable session artifacts."""

from __future__ import annotations

from typing import Any

SESSION_SCHEMA_VERSION = 1


def strict_text(
    value: Any,
    *,
    compact: bool = False,
    allow_empty: bool = False,
) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()) if compact else value.strip()
    return text if text or allow_empty else None


def strict_items(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return []
    items = []
    for item in value:
        text = strict_text(item, compact=True)
        if text is None:
            return []
        items.append(text)
    return items


def valid_iterations(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 20


def valid_attempts(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
