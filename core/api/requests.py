"""Interactions for which MIRA Core waits on the active consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


# Presentation code reads this annotation; native HITL decisions remain
# LangGraph decision dictionaries and Commands inside core.execution.runner.
APPROVAL_CONSEQUENCE = "_mira_consequence"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Project native LangGraph HITL interrupts to a frontend."""

    interrupts: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class AskUserRequest:
    """Project MIRA's native ask_user tool interrupt to a frontend."""

    interrupt: Any


@dataclass(frozen=True, slots=True)
class ArtifactReviewRequest:
    """Request review of a MIRA-only formal Goal or Plan interrupt."""

    artifact_type: Literal["goal", "plan"]
    interrupt: Any
    artifact: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactDisplayRequest:
    """Request display of the retained formal MIRA artifact."""

    artifact_type: Literal["goal", "plan"]
    interrupt: Any = None


@dataclass(frozen=True, slots=True)
class MCPApprovalRequest:
    """Ask whether one configured MCP server may be connected."""

    server: Any
    preview: str = ""


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """Small application-level confirmation not owned by an agent interrupt."""

    kind: str
    message: str
    context: Mapping[str, Any] | None = None


FrontendRequest = (
    ApprovalRequest
    | AskUserRequest
    | ArtifactReviewRequest
    | ArtifactDisplayRequest
    | MCPApprovalRequest
    | ConfirmationRequest
)


__all__ = [
    "APPROVAL_CONSEQUENCE",
    "ApprovalRequest",
    "ArtifactDisplayRequest",
    "ArtifactReviewRequest",
    "AskUserRequest",
    "ConfirmationRequest",
    "FrontendRequest",
    "MCPApprovalRequest",
]
