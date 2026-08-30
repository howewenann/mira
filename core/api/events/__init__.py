"""Typed observations emitted through the MIRA Core consumer contract."""

from core.api.events.artifacts import ArtifactEvent
from core.api.events.base import EventIdentity, Namespace
from core.api.events.messages import MessageEvent
from core.api.events.rubric import RubricEvent
from core.api.events.runtime import CompactionEvent, InformationEvent, MCPEvent, RuntimeEvent, UsageEvent
from core.api.events.subagents import SubagentEvent
from core.api.events.tools import ToolEvent


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
    "CompactionEvent",
    "EventIdentity",
    "FrontendEvent",
    "InformationEvent",
    "MCPEvent",
    "MessageEvent",
    "Namespace",
    "RubricEvent",
    "RuntimeEvent",
    "SubagentEvent",
    "ToolEvent",
    "UsageEvent",
]
