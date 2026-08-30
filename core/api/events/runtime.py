"""Application, usage, compaction, MCP, and information events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from core.api.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEvent(EventIdentity):
    """Application/session lifecycle notification."""

    kind: str
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
    """MIRA MCP discovery/connection lifecycle notification."""

    phase: str
    server: str = ""
    detail: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InformationEvent(EventIdentity):
    """User-visible informational, warning, error, or correction event."""

    text: str = ""
    kind: str = "system"
    correction: Mapping[str, Any] | None = None


__all__ = ["CompactionEvent", "InformationEvent", "MCPEvent", "RuntimeEvent", "UsageEvent"]
