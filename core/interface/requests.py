"""Interactions for which MIRA Core waits on the active consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


# Presentation code reads this annotation; native HITL decisions remain
# LangGraph decision dictionaries and Commands inside core.execution.runner.
APPROVAL_CONSEQUENCE = "_mira_consequence"
ArtifactReviewAction = Literal["implement", "close", "revise", "clear"]
MCPApprovalDecision = Literal["allow", "deny", "always_allow"]
ApprovalDecision = Literal["allow_once", "edit", "reject_once", "allow_always"]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Request decisions for native LangGraph HITL interrupts.

    Core sends this when one or more action requests require approval. The
    ``interrupts`` tuple preserves the native LangGraph values. Return one
    MIRA decision dictionary per action request, in encounter order, using
    ``allow_once``, ``edit``, ``reject_once``, or ``allow_always`` as its
    ``type``. Edit also supplies ``edited_action``. Core keeps this frontend
    contract flat and translates decisions to LangGraph's native vocabulary.
    It uses the ordinary decision payload for one interrupt and partitions
    decisions by native interrupt ID only when LangGraph reports multiple
    pending interrupts.
    """

    interrupts: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class AskUserRequest:
    """Request one answer for MIRA's native ``ask_user`` interrupt.

    ``interrupt`` is the original LangGraph interrupt containing the question
    and options. Return the selected option or free-form answer as ``str``.
    Core passes that string directly to ``Command(resume=answer)``.
    """

    interrupt: Any


@dataclass(frozen=True, slots=True)
class MCPElicitationRequest:
    """Request answers for one native LangChain MCP elicitation interrupt.

    The frontend returns the upstream resume shape unchanged:
    ``{"responses": {key: {"action": "accept", "content": {...}}}}``.
    Decline and cancel responses omit ``content``.
    """

    interrupt: Any


@dataclass(frozen=True, slots=True)
class ArtifactReviewRequest:
    """Request review of a provisional MIRA Goal or Plan.

    ``artifact_type`` identifies the formal artifact, ``interrupt`` is the
    native finalizer interrupt, and ``artifact`` is the complete provisional
    mapping. Return ``{"action": action}``, where action is ``implement``,
    ``close``, ``revise``, or ``clear``. A revision may also include
    ``{"feedback": str}``. Implement accepts and starts the artifact; close
    accepts and retains it without starting; revise/clear reject the proposal
    and preserve any previously retained formal work.
    """

    artifact_type: Literal["goal", "plan"]
    interrupt: Any
    artifact: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactDisplayRequest:
    """Request display of the retained formal Goal or Plan.

    Core sends this for ``show_goal`` or ``show_plan``. Read the retained value
    from ``session.snapshot()``, display it, and return a short string describing
    the outcome. Core stringifies the response as the control-tool result and
    ends that display-only turn.
    """

    artifact_type: Literal["goal", "plan"]
    interrupt: Any = None


@dataclass(frozen=True, slots=True)
class MCPApprovalRequest:
    """Ask whether one configured MCP server may be connected.

    ``server`` is the configured server state and ``preview`` is the safe
    human-readable launch/use summary. Return exactly ``"allow"`` for this
    process, ``"always_allow"`` to persist approval for the current server
    fingerprint, or ``"deny"``. Any other value is interpreted as denial.
    """

    server: Any
    preview: str = ""


FrontendRequest = (
    ApprovalRequest
    | AskUserRequest
    | MCPElicitationRequest
    | ArtifactReviewRequest
    | ArtifactDisplayRequest
    | MCPApprovalRequest
)


__all__ = [
    "APPROVAL_CONSEQUENCE",
    "ApprovalDecision",
    "ApprovalRequest",
    "ArtifactDisplayRequest",
    "ArtifactReviewAction",
    "ArtifactReviewRequest",
    "AskUserRequest",
    "FrontendRequest",
    "MCPApprovalDecision",
    "MCPApprovalRequest",
    "MCPElicitationRequest",
]
