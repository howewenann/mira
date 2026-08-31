"""Typed observations emitted through the MIRA Core Interface."""

from core.interface.events.artifacts import ArtifactEvent, ArtifactPhase, ArtifactType
from core.interface.events.base import EventIdentity, Namespace
from core.interface.events.messages import MessageEvent
from core.interface.events.rubric import RubricEvent, RubricPhase
from core.interface.events.runtime import (
    CompactionEvent,
    InformationEvent,
    MCPEvent,
    MCPPhase,
    RuntimeEvent,
    RuntimeKind,
    UsageEvent,
)
from core.interface.events.subagents import SubagentEvent
from core.interface.events.tools import ToolEvent


FrontendEvent = (
    MessageEvent
    | ToolEvent
    | SubagentEvent
    | RuntimeEvent
    | UsageEvent
    | CompactionEvent
    | ArtifactEvent
    | RubricEvent
    | MCPEvent
    | InformationEvent
)


__all__ = [
    "ArtifactEvent",
    "ArtifactPhase",
    "ArtifactType",
    "CompactionEvent",
    "EventIdentity",
    "FrontendEvent",
    "InformationEvent",
    "MCPEvent",
    "MCPPhase",
    "MessageEvent",
    "Namespace",
    "RubricEvent",
    "RubricPhase",
    "RuntimeEvent",
    "RuntimeKind",
    "SubagentEvent",
    "ToolEvent",
    "UsageEvent",
]
