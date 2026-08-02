"""Plan/Goal natural-stop correction rules and control-marker helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage

from agent.middleware.correction import CorrectionDecision
from agent.planning.policy import (
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_RESEARCH,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
)

PLANNING_NEXT_ACTION_ANSWER = "NEXT_ACTION: ANSWER"
PLANNING_NEXT_ACTION_RESEARCH = "NEXT_ACTION: RESEARCH"
PLANNING_NEXT_ACTION_ASK_USER = "NEXT_ACTION: ASK_USER"
PLANNING_NEXT_ACTION_PREPARE_PLAN = "NEXT_ACTION: PREPARE_PLAN"
PLANNING_NEXT_ACTION_PREPARE_GOAL = "NEXT_ACTION: PREPARE_GOAL"
PLANNING_NEXT_ACTION_MARKERS = (
    PLANNING_NEXT_ACTION_ANSWER,
    PLANNING_NEXT_ACTION_RESEARCH,
    PLANNING_NEXT_ACTION_ASK_USER,
    PLANNING_NEXT_ACTION_PREPARE_PLAN,
    PLANNING_NEXT_ACTION_PREPARE_GOAL,
)
PLANNING_NEXT_ACTION_FAILURE = (
    "MIRA could not produce a valid Plan/Goal-mode response within the configured "
    "next-action retry limit. The requested work may be incomplete."
)

PLANNING_NEXT_ACTION_CONTRACT = """Work normally and use the available tools whenever needed.

Before ending a response that contains no tool call, append exactly one final
line describing what must happen next to satisfy the user's current request:

NEXT_ACTION: ANSWER
NEXT_ACTION: RESEARCH
NEXT_ACTION: ASK_USER
{prepare_marker}

Use NEXT_ACTION: ANSWER only when the prose being returned completely answers
the user's current request using work already performed.

Use NEXT_ACTION: RESEARCH when more information available through the read-only
tools is needed. Call the relevant research tool instead of ending the response.

Use NEXT_ACTION: ASK_USER when a material decision or unavailable fact must come
from the user. Call ask_user instead of ending the response.

Use {prepare_marker} when the {artifact_name} is decision-complete. Call
{prepare_tool} instead of ending the response.

To display the retained {artifact_name}, call {show_tool}. To construct a new or
revised {artifact_name}, call {prepare_tool}. {present_tool} is finalization-only.

This is a post-response self-check, not an instruction to select an action
before reasoning. Responses containing a tool call do not need a NEXT_ACTION
line. Keep the NEXT_ACTION line in exact ASCII even when answering in another
language. Do not announce research, questioning, or {artifact_name}
preparation without performing the corresponding tool call."""


@dataclass(frozen=True)
class PlanningNextActionRule:
    """Natural-stop rule for one formal Plan or Goal research workflow."""

    workflow: Literal["plan", "goal"]
    failure_text: str = PLANNING_NEXT_ACTION_FAILURE

    def __post_init__(self) -> None:
        if self.workflow not in {"plan", "goal"}:
            raise ValueError("PlanningNextActionRule workflow must be 'plan' or 'goal'")

    @property
    def protocol_id(self) -> str:
        return f"{self.workflow}_next_action"

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
    def prepare_marker(self) -> str:
        return (
            PLANNING_NEXT_ACTION_PREPARE_GOAL
            if self.workflow == "goal"
            else PLANNING_NEXT_ACTION_PREPARE_PLAN
        )

    @property
    def prepare_tool(self) -> str:
        return PREPARE_GOAL_TOOL if self.workflow == "goal" else PREPARE_PLAN_TOOL

    @property
    def show_tool(self) -> str:
        return "goal_show" if self.workflow == "goal" else "plan_show"

    @property
    def present_tool(self) -> str:
        return "present_goal" if self.workflow == "goal" else "present_plan"

    def applies(self, state: dict[str, Any]) -> bool:
        return str(state.get("planning_stage") or PLANNING_STAGE_PLAN_RESEARCH) == self.active_stage

    def reminder(self, state: dict[str, Any]) -> str:  # noqa: ARG002
        return planning_next_action_contract(self.workflow)

    def inspect(self, message: AIMessage, state: dict[str, Any]) -> CorrectionDecision:  # noqa: ARG002
        action = planning_next_action(message, self.active_stage)
        if action == PLANNING_NEXT_ACTION_ANSWER:
            return CorrectionDecision(
                accepted=True,
                replacement=strip_planning_next_action(message),
            )

        failed_check, retry_prompt = self._rejection(action, message)
        return CorrectionDecision(
            accepted=False,
            failed_check=failed_check,
            retry_prompt=retry_prompt,
        )

    def _rejection(self, action: str | None, message: AIMessage) -> tuple[str, str]:
        if action == PLANNING_NEXT_ACTION_RESEARCH:
            return (
                "NEXT_ACTION: RESEARCH was declared, but no research tool was called.",
                "You concluded that further research is required, but ended without calling "
                "a research tool. Perform that research now.",
            )
        if action == PLANNING_NEXT_ACTION_ASK_USER:
            return (
                "NEXT_ACTION: ASK_USER was declared, but ask_user was not called.",
                "You concluded that user input is required, but ended without calling "
                "ask_user. Call ask_user now.",
            )
        if action == self.prepare_marker:
            return (
                f"{self.prepare_marker} was declared, but {self.prepare_tool} was not called.",
                f"You concluded that the {self.workflow_label} is ready, but ended without "
                f"calling {self.prepare_tool}. Call {self.prepare_tool} now.",
            )

        terminal = terminal_planning_next_action(message)
        other_prepare = (
            PLANNING_NEXT_ACTION_PREPARE_PLAN
            if self.workflow == "goal"
            else PLANNING_NEXT_ACTION_PREPARE_GOAL
        )
        if terminal == other_prepare:
            failed_check = f"{terminal} is not valid during {self.workflow_label} research."
        else:
            failed_check = (
                "The response did not end with exactly one valid terminal NEXT_ACTION line "
                f"for {self.workflow_label} research."
            )
        return (
            failed_check,
            "Your previous response ended without a tool call or a valid terminal "
            "next-action classification. Perform the required tool action now, or return "
            "a complete answer ending with `NEXT_ACTION: ANSWER`.",
        )


def planning_next_action_contract(workflow: Literal["plan", "goal"]) -> str:
    """Return the workflow-specific transient next-action reminder."""
    if workflow == "goal":
        marker = PLANNING_NEXT_ACTION_PREPARE_GOAL
        tool = PREPARE_GOAL_TOOL
        artifact = "Goal"
        show_tool = "goal_show"
        present_tool = "present_goal"
    else:
        marker = PLANNING_NEXT_ACTION_PREPARE_PLAN
        tool = PREPARE_PLAN_TOOL
        artifact = "Plan"
        show_tool = "plan_show"
        present_tool = "present_plan"
    return PLANNING_NEXT_ACTION_CONTRACT.format(
        prepare_marker=marker,
        prepare_tool=tool,
        artifact_name=artifact,
        show_tool=show_tool,
        present_tool=present_tool,
    )


def terminal_planning_next_action_span(text: str) -> tuple[int, int] | None:
    """Return the exact terminal control-line span, including its newline."""
    offset = 0
    candidate: tuple[int, int, str] | None = None
    for line in text.splitlines(keepends=True):
        value = line.rstrip("\r\n")
        if value.strip():
            candidate = (offset, offset + len(line), value)
        offset += len(line)
    if candidate is None and text:
        candidate = (0, len(text), text)
    if candidate is None or candidate[2] not in PLANNING_NEXT_ACTION_MARKERS:
        return None
    return candidate[0], candidate[1]


def terminal_planning_next_action(message: AIMessage) -> str | None:
    """Return an exact terminal marker without checking workflow validity."""
    text = str(message.text)
    span = terminal_planning_next_action_span(text)
    if span is None:
        return None
    return text[span[0] : span[1]].rstrip("\r\n")


def strip_terminal_planning_next_action_text(text: str) -> str:
    """Remove only an exact terminal planning next-action control line."""
    span = terminal_planning_next_action_span(text)
    if span is None:
        return text
    start, end = span
    return text[:start] + text[end:]


def planning_next_action(message: AIMessage, stage: str) -> str | None:
    """Return one exact, terminal, stage-valid next-action marker."""
    text = str(message.text)
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    markers = [line for line in lines if line in PLANNING_NEXT_ACTION_MARKERS]
    if not non_empty or len(markers) != 1 or non_empty[-1] != markers[0]:
        return None
    allowed = {
        PLANNING_NEXT_ACTION_ANSWER,
        PLANNING_NEXT_ACTION_RESEARCH,
        PLANNING_NEXT_ACTION_ASK_USER,
        (
            PLANNING_NEXT_ACTION_PREPARE_GOAL
            if stage == PLANNING_STAGE_GOAL_RESEARCH
            else PLANNING_NEXT_ACTION_PREPARE_PLAN
        ),
    }
    return markers[0] if markers[0] in allowed else None


def strip_planning_next_action(message: AIMessage) -> AIMessage:
    """Strip only the exact terminal next-action control line."""
    text = str(message.text)
    span = terminal_planning_next_action_span(text)
    if span is None:
        return message
    start, end = span
    content = message.content
    if isinstance(content, str):
        cleaned: Any = content[:start] + content[end:]
    else:
        cleaned = _strip_text_span_from_blocks(content, start, end)
    return message.model_copy(update={"content": cleaned})


def _strip_text_span_from_blocks(content: list[Any], start: int, end: int) -> list[Any]:
    cleaned = list(content)
    offset = 0
    for index, block in enumerate(content):
        if isinstance(block, str):
            value = block
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            value = block["text"]
        else:
            continue

        block_start = offset
        block_end = offset + len(value)
        overlap_start = max(start, block_start)
        overlap_end = min(end, block_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - block_start
            local_end = overlap_end - block_start
            value = value[:local_start] + value[local_end:]
            cleaned[index] = value if isinstance(block, str) else {**block, "text": value}
        offset = block_end
    return cleaned


__all__ = [
    "PLANNING_NEXT_ACTION_ANSWER",
    "PLANNING_NEXT_ACTION_ASK_USER",
    "PLANNING_NEXT_ACTION_FAILURE",
    "PLANNING_NEXT_ACTION_MARKERS",
    "PLANNING_NEXT_ACTION_PREPARE_GOAL",
    "PLANNING_NEXT_ACTION_PREPARE_PLAN",
    "PLANNING_NEXT_ACTION_RESEARCH",
    "PlanningNextActionRule",
    "planning_next_action",
    "planning_next_action_contract",
    "strip_planning_next_action",
    "strip_terminal_planning_next_action_text",
    "terminal_planning_next_action_span",
]
