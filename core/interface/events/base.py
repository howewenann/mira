"""Identity and provenance shared by all MIRA Core Interface events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


Namespace = tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EventIdentity:
    """Native/application provenance; zero durable IDs indicate transient events.

    ``event_id`` refers to an existing session transcript entry. Tool lifecycle
    updates use ``origin_event_id`` to retain the original tool-call relationship.
    """

    session_id: str = ""
    turn_id: str = ""
    message_id: str = ""
    event_id: int = 0
    origin_event_id: int = 0
    namespace: Namespace = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""


__all__ = ["EventIdentity", "Namespace"]
