"""MIRA Goal and retained Plan lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from core.interface.events.base import EventIdentity

ArtifactType = Literal["goal", "plan"]
ArtifactPhase = Literal[
    "proposed",
    "implement",
    "active",
    "revise",
    "close",
    "clear",
    "cancelled",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactEvent(EventIdentity):
    """MIRA-only formal Goal or retained Plan lifecycle notification.

    ``phase`` is ``proposed`` while awaiting review, ``implement`` when a
    proposal is accepted and started, ``active`` when a retained artifact is
    started directly, ``revise``/``close``/``clear`` for review or retained
    lifecycle decisions, and ``cancelled`` when its interrupted review ends.
    """

    artifact_type: ArtifactType
    phase: ArtifactPhase
    artifact: Mapping[str, Any] | None = None
    artifact_id: str = ""
    decision: Mapping[str, Any] | None = None


__all__ = ["ArtifactEvent", "ArtifactPhase", "ArtifactType"]
