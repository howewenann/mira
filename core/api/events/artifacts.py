"""MIRA Goal and retained Plan lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from core.api.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactEvent(EventIdentity):
    """MIRA-only formal Goal or retained Plan lifecycle notification."""

    artifact_type: Literal["goal", "plan"]
    phase: str
    artifact: Mapping[str, Any] | None = None
    artifact_id: str = ""
    decision: Mapping[str, Any] | None = None


__all__ = ["ArtifactEvent"]
