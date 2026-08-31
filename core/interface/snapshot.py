"""Consumer-readable projection of authoritative MIRA session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Current frontend state without exposing live agents or graph objects.

    ``session_id`` is MIRA's durable session identity; ``workspace`` is its
    absolute project root; ``mode`` is ``action`` or ``planning``;
    ``runtime_state`` is ``ready``, ``running``, ``cancelling``, or ``closed``;
    ``title`` and ``turns`` describe transcript history; ``current_goal`` and
    ``current_plan`` contain at most one retained formal artifact; ``transcript``
    is the tuple of persisted event mappings; and the remaining mappings project
    dashboard usage, model metadata, tools, resources, Rubric settings, and MCP
    state. Nested mappings intentionally remain lightweight projections.
    """

    session_id: str
    workspace: str
    mode: str
    runtime_state: str
    title: str
    turns: int
    current_goal: Mapping[str, Any] | None
    current_plan: Mapping[str, Any] | None
    transcript: tuple[Mapping[str, Any], ...]
    dashboard: Mapping[str, Any]
    model: Mapping[str, Any]
    tools: tuple[Mapping[str, Any], ...]
    resources: Mapping[str, tuple[Mapping[str, Any], ...]]
    rubric: Mapping[str, Any]
    mcp: Mapping[str, Any]


__all__ = ["SessionSnapshot"]
