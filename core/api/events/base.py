"""Identity and provenance shared by all MIRA Core API events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


Namespace = tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EventIdentity:
    """Stable native/application identity carried across the consumer boundary."""

    session_id: str = ""
    turn_id: str = ""
    message_id: str = ""
    namespace: Namespace = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""


__all__ = ["EventIdentity", "Namespace"]
