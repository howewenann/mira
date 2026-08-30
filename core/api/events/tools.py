"""Native-shaped tool-call lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.api.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolEvent(EventIdentity):
    """One native tool-call lifecycle update with structured arguments intact."""

    phase: Literal[
        "delegation",
        "arguments_delta",
        "start",
        "update",
        "approval_resolved",
        "result",
        "error",
        "completed_result",
        "completed_error",
        "recovered_start",
        "recovered_result",
        "recovered_error",
        "stop",
    ]
    name: str = ""
    tool_call_id: str = ""
    arguments: Any = None
    result: Any = None
    calls: tuple[Any, ...] = ()
    status: str = ""
    duration_ms: int | None = None


__all__ = ["ToolEvent"]
