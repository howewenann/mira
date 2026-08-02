"""Plan/Goal natural-stop response-status correction rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage

from agent.middleware.correction import CorrectionDecision
from agent.planning.policy import (
    FINALIZE_GOAL_TOOL,
    FINALIZE_PLAN_TOOL,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_RESEARCH,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
    SHOW_GOAL_TOOL,
    SHOW_PLAN_TOOL,
)

PLANNING_RESPONSE_STATUS_COMPLETE = "RESPONSE_STATUS: COMPLETE"
PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH = "RESPONSE_STATUS: NEEDS_RESEARCH"
PLANNING_RESPONSE_STATUS_NEEDS_USER_INPUT = "RESPONSE_STATUS: NEEDS_USER_INPUT"
PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN = "RESPONSE_STATUS: READY_TO_PREPARE_PLAN"
PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL = "RESPONSE_STATUS: READY_TO_PREPARE_GOAL"
PLANNING_RESPONSE_STATUSES = (
    PLANNING_RESPONSE_STATUS_COMPLETE,
    PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH,
    PLANNING_RESPONSE_STATUS_NEEDS_USER_INPUT,
    PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN,
    PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL,
)
PLANNING_RESPONSE_STATUS_FAILURE = (
    "MIRA could not produce a valid Plan/Goal-mode response within the configured "
    "response-status retry limit. The requested work may be incomplete."
)

PLANNING_RESPONSE_STATUS_CONTRACT = """Work normally and use the available tools whenever needed.

After drafting a response, perform a final response-state check. Before ending
a response that contains no tool call, append exactly one of these lines as the
final non-empty line:

RESPONSE_STATUS: COMPLETE
RESPONSE_STATUS: NEEDS_RESEARCH
RESPONSE_STATUS: NEEDS_USER_INPUT
{prepare_status}

Use RESPONSE_STATUS: COMPLETE only when the response fully satisfies the
user's current request and nothing remains unresolved. A response that
announces future work, awaits agreement, lacks required information, requires
a tool call, or still needs formal {artifact_name} preparation is not COMPLETE.

Use RESPONSE_STATUS: NEEDS_RESEARCH when more information available through
the read-only tools is required. Call the relevant research tool instead of
ending the response.

Use RESPONSE_STATUS: NEEDS_USER_INPUT when a material decision or unavailable
fact must come from the user. Call ask_user instead of ending the response.

Use {prepare_status} when the {artifact_name} is decision-complete. Call
{prepare_tool} instead of ending the response.

To display the retained {artifact_name}, call {show_tool}. To construct a new or
revised {artifact_name}, call {prepare_tool}. {finalize_tool} is finalization-only.
Requests to show, reopen, review, or return to retained work require an immediate
{show_tool} call without research, prose reproduction, preparation, or finalization.

This is a post-draft classification of the response's current state, not a
prediction of a future next step. Responses containing a tool call omit the
RESPONSE_STATUS line. Keep the RESPONSE_STATUS line in exact ASCII even when
answering in another language. Do not announce research, questioning, or
{artifact_name} preparation without performing the corresponding tool call."""


@dataclass(frozen=True)
class PlanningResponseStatusRule:
    """Natural-stop response-status rule for one Plan or Goal workflow."""

    workflow: Literal["plan", "goal"]
    failure_text: str = PLANNING_RESPONSE_STATUS_FAILURE
    check_name: str = "Response"

    def __post_init__(self) -> None:
        if self.workflow not in {"plan", "goal"}:
            raise ValueError("PlanningResponseStatusRule workflow must be 'plan' or 'goal'")

    @property
    def protocol_id(self) -> str:
        return f"{self.workflow}_response_status"

    @property
    def workflow_label(self) -> str:
        return self.workflow.title()

    @property
    def active_stage(self) -> str:
        return (
            PLANNING_STAGE_GOAL_RESEARCH
            if self.workflow == "goal"
            else PLANNING_STAGE_PLAN_RESEARCH
        )

    @property
    def prepare_status(self) -> str:
        return (
            PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL
            if self.workflow == "goal"
            else PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN
        )

    @property
    def prepare_tool(self) -> str:
        return PREPARE_GOAL_TOOL if self.workflow == "goal" else PREPARE_PLAN_TOOL

    @property
    def show_tool(self) -> str:
        return SHOW_GOAL_TOOL if self.workflow == "goal" else SHOW_PLAN_TOOL

    @property
    def finalize_tool(self) -> str:
        return FINALIZE_GOAL_TOOL if self.workflow == "goal" else FINALIZE_PLAN_TOOL

    def applies(self, state: dict[str, Any]) -> bool:
        return str(state.get("planning_stage") or PLANNING_STAGE_PLAN_RESEARCH) == self.active_stage

    def reminder(self, state: dict[str, Any]) -> str:  # noqa: ARG002
        return planning_response_status_contract(self.workflow)

    def inspect(self, message: AIMessage, state: dict[str, Any]) -> CorrectionDecision:  # noqa: ARG002
        status = planning_response_status(message, self.active_stage)
        if status == PLANNING_RESPONSE_STATUS_COMPLETE:
            return CorrectionDecision(accepted=True)

        failed_check, retry_prompt = self._rejection(status, message)
        return CorrectionDecision(
            accepted=False,
            failed_check=failed_check,
            retry_prompt=retry_prompt,
        )

    def _rejection(self, status: str | None, message: AIMessage) -> tuple[str, str]:
        if status == PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH:
            return (
                "RESPONSE_STATUS: NEEDS_RESEARCH was declared, but no research tool was called.",
                "Your previous response classified itself as needing research, but ended "
                "without calling a research tool. Perform that research now.",
            )
        if status == PLANNING_RESPONSE_STATUS_NEEDS_USER_INPUT:
            return (
                "RESPONSE_STATUS: NEEDS_USER_INPUT was declared, but ask_user was not called.",
                "Your previous response classified itself as needing user input, but ended "
                "without calling ask_user. Call ask_user now.",
            )
        if status == self.prepare_status:
            return (
                f"{self.prepare_status} was declared, but {self.prepare_tool} was not called.",
                f"Your previous response classified the {self.workflow_label} as ready, but "
                f"ended without calling {self.prepare_tool}. Call {self.prepare_tool} now.",
            )

        terminal = terminal_planning_response_status(message)
        other_prepare = (
            PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN
            if self.workflow == "goal"
            else PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL
        )
        if terminal == other_prepare:
            failed_check = f"{terminal} is not valid during {self.workflow_label} research."
        else:
            failed_check = (
                "The response did not end with exactly one valid terminal RESPONSE_STATUS "
                f"line for {self.workflow_label} research."
            )
        return (
            failed_check,
            "Your previous response ended without a tool call or a valid terminal "
            "response-status classification. Perform the required tool action now, or "
            "return a fully resolved answer ending with `RESPONSE_STATUS: COMPLETE`.",
        )


def planning_response_status_contract(workflow: Literal["plan", "goal"]) -> str:
    """Return the workflow-specific transient response-status reminder."""
    if workflow == "goal":
        status = PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL
        tool = PREPARE_GOAL_TOOL
        artifact = "Goal"
        show_tool = SHOW_GOAL_TOOL
        finalize_tool = FINALIZE_GOAL_TOOL
    else:
        status = PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN
        tool = PREPARE_PLAN_TOOL
        artifact = "Plan"
        show_tool = SHOW_PLAN_TOOL
        finalize_tool = FINALIZE_PLAN_TOOL
    return PLANNING_RESPONSE_STATUS_CONTRACT.format(
        prepare_status=status,
        prepare_tool=tool,
        artifact_name=artifact,
        show_tool=show_tool,
        finalize_tool=finalize_tool,
    )


def terminal_planning_response_status(message: AIMessage) -> str | None:
    """Return an exact terminal status without checking workflow validity."""
    lines = str(message.text).splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty or non_empty[-1] not in PLANNING_RESPONSE_STATUSES:
        return None
    return non_empty[-1]


def planning_response_status(message: AIMessage, stage: str) -> str | None:
    """Return one exact, terminal, stage-valid response status."""
    lines = str(message.text).splitlines()
    non_empty = [line for line in lines if line.strip()]
    statuses = [line for line in lines if line in PLANNING_RESPONSE_STATUSES]
    if not non_empty or len(statuses) != 1 or non_empty[-1] != statuses[0]:
        return None
    allowed = {
        PLANNING_RESPONSE_STATUS_COMPLETE,
        PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH,
        PLANNING_RESPONSE_STATUS_NEEDS_USER_INPUT,
        (
            PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL
            if stage == PLANNING_STAGE_GOAL_RESEARCH
            else PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN
        ),
    }
    return statuses[0] if statuses[0] in allowed else None


__all__ = [
    "PLANNING_RESPONSE_STATUS_COMPLETE",
    "PLANNING_RESPONSE_STATUS_FAILURE",
    "PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH",
    "PLANNING_RESPONSE_STATUS_NEEDS_USER_INPUT",
    "PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL",
    "PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN",
    "PLANNING_RESPONSE_STATUSES",
    "PlanningResponseStatusRule",
    "planning_response_status",
    "planning_response_status_contract",
    "terminal_planning_response_status",
]
