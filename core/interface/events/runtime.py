"""Application, usage, compaction, MCP, and information events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from core.interface.events.base import EventIdentity

RuntimeKind = Literal["startup", "waiting", "message_group", "session"]
MCPPhase = Literal["initializing", "initialized", "error"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEvent(EventIdentity):
    """Application/session lifecycle notification.

    Known state values depend on ``kind``: waiting uses ``start``/``finish``;
    message groups use ``finish``; sessions use ``opened``, ``turn_started``,
    ``ready``, ``cancelling``, and ``closed``. Startup state is intentionally
    extensible human-readable progress text.
    """

    kind: RuntimeKind
    state: str = ""
    detail: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageEvent(EventIdentity):
    """Updated usage/context projection for the current session."""

    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionEvent(EventIdentity):
    """DeepAgents compaction lifecycle notification."""

    phase: Literal["start", "finish"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPEvent(EventIdentity):
    """MIRA MCP initialization lifecycle notification."""

    phase: MCPPhase
    server: str = ""
    detail: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InformationEvent(EventIdentity):
    """User-visible informational, warning, error, or correction event.

    Known ``kind`` values are ``system``, ``startup``, ``info``, ``status``,
    ``warning``, ``error``, ``muted``, and ``correction``. This field remains
    extensible so consumers should render unknown values as ordinary system
    information rather than rejecting the event.
    """

    text: str = ""
    kind: str = "system"
    correction: Mapping[str, Any] | None = None


__all__ = [
    "CompactionEvent",
    "InformationEvent",
    "MCPEvent",
    "MCPPhase",
    "RuntimeEvent",
    "RuntimeKind",
    "UsageEvent",
]
