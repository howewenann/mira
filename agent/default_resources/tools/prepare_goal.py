"""Criteria-first preparation for durable MIRA Goals."""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.planning.policy import GOAL_FINALIZATION_POLICY, PLANNING_STAGE_GOAL_FINALIZE
from agent.planning.tool_context import PlanningToolContext

GOAL_FIELD_MAX_CHARS = 4000


@tool(
    "prepare_goal",
    description=(
        "Begin formal Goal construction when the Goal objective is sufficiently understood and every "
        "material user decision is resolved. Pass a concise formal Objective that preserves the "
        "authoritative request's meaning without adding, removing, or materially changing its scope, "
        "deliverables, outcome, or constraints. Include concise relevant "
        "context, constraints, and bounded research evidence. MIRA will generate Success Criteria next "
        "and then require finalize_goal. "
        "Do not produce or include an implementation Plan, return the Goal in prose, or add unsupported "
        "scope. Use ask_user first when user input is still required."
    ),
)
async def prepare_goal(
    objective: str,
    runtime: ToolRuntime[PlanningToolContext, dict],
    context_and_constraints: str = "",
    research_evidence: str = "",
) -> Command:
    """Generate exact criteria and advance normally to Goal finalization."""
    if runtime.context is None:
        raise RuntimeError("prepare_goal requires MIRA planning context")
    state = runtime.state if isinstance(runtime.state, dict) else {}
    authoritative_request = str(state.get("planning_authoritative_request") or "")
    objective = str(objective or "").strip()[:GOAL_FIELD_MAX_CHARS] or authoritative_request
    context = str(context_and_constraints or "").strip()[:GOAL_FIELD_MAX_CHARS]
    evidence = str(research_evidence or "").strip()[:GOAL_FIELD_MAX_CHARS]
    if not objective:
        raise RuntimeError("prepare_goal requires the authoritative user objective")
    research_context = "\n\n".join(value for value in (context, evidence) if value)

    previous_criteria = str(state.get("planning_previous_criteria") or "")
    feedback = str(state.get("planning_revision_feedback") or "")
    planning_context = runtime.context
    if not isinstance(planning_context, PlanningToolContext):
        raise RuntimeError("prepare_goal received invalid MIRA planning context")
    service = planning_context.success_criteria
    if previous_criteria and feedback:
        criteria = await service.revise(objective, previous_criteria, feedback, research_context)
    else:
        criteria = await service.generate(
            objective,
            research_context,
            authoritative_request=authoritative_request,
        )

    revision_context = (
        f"\n\n<user_feedback>\n{feedback}\n</user_feedback>" if feedback else ""
    )
    result = (
        f"{GOAL_FINALIZATION_POLICY}\n\n"
        f"<authoritative_request>\n{authoritative_request}\n</authoritative_request>\n\n"
        f"<objective>\n{objective}\n</objective>\n\n"
        f"<success_criteria>\n{criteria}\n</success_criteria>"
        f"{revision_context}"
    )
    return Command(
        update={
            "planning_stage": PLANNING_STAGE_GOAL_FINALIZE,
            "planning_objective": objective,
            "planning_context_and_constraints": context,
            "planning_research_evidence": evidence,
            "planning_success_criteria": criteria,
            "messages": [
                ToolMessage(
                    content=result,
                    name="prepare_goal",
                    tool_call_id=str(runtime.tool_call_id or "prepare_goal"),
                )
            ],
        }
    )
