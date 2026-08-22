"""Criteria-first preparation for durable MIRA Plans."""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.planning.policy import PLAN_FINALIZATION_POLICY, PLANNING_STAGE_PLAN_FINALIZE
from agent.planning.tool_context import PlanningToolContext

PLAN_FIELD_MAX_CHARS = 4000


@tool(
    "prepare_plan",
    description=(
        "Begin formal Plan construction only when the proposal is decision-complete. "
        "Pass a concise formal Objective that may improve wording but preserves the authoritative "
        "user request's meaning, scope, deliverables, outcome, and constraints. Include relevant context, constraints, "
        "and bounded research findings. MIRA will generate Success Criteria before the "
        "final Plan and then require finalize_plan. Do not include a completed Plan, draft "
        "the final Plan in prose, or call this for ordinary discussion or read-only answers."
    ),
)
async def prepare_plan(
    objective: str,
    runtime: ToolRuntime[PlanningToolContext, dict],
    context_and_constraints: str = "",
) -> Command:
    """Generate exact criteria and advance normally to Plan finalization."""
    if runtime.context is None:
        raise RuntimeError("prepare_plan requires MIRA planning context")
    objective = str(objective or "").strip()[:PLAN_FIELD_MAX_CHARS]
    context = (
        str(context_and_constraints or "").strip()[:PLAN_FIELD_MAX_CHARS]
        or "No additional constraints."
    )
    if not objective:
        raise RuntimeError("prepare_plan requires the authoritative user objective")

    state = runtime.state if isinstance(runtime.state, dict) else {}
    previous_criteria = str(state.get("planning_previous_criteria") or "")
    feedback = str(state.get("planning_revision_feedback") or "")
    planning_context = runtime.context
    if not isinstance(planning_context, PlanningToolContext):
        raise RuntimeError("prepare_plan received invalid MIRA planning context")
    service = planning_context.success_criteria
    if previous_criteria and feedback:
        criteria = await service.revise(objective, previous_criteria, feedback, context)
    else:
        criteria = await service.generate(
            objective,
            context,
            authoritative_request=str(state.get("planning_authoritative_request") or ""),
        )

    revision_context = ""
    previous_artifact = str(state.get("planning_previous_artifact") or "")
    if previous_artifact and feedback:
        revision_context = (
            "\n\nThe revision must be a complete replacement.\n"
            f"<previous_plan>\n{previous_artifact}\n</previous_plan>\n"
            f"<user_feedback>\n{feedback}\n</user_feedback>"
        )
    result = (
        f"{PLAN_FINALIZATION_POLICY}\n\n"
        f"<objective>\n{objective}\n</objective>\n\n"
        f"<context_and_constraints>\n{context}\n</context_and_constraints>\n\n"
        f"<success_criteria>\n{criteria}\n</success_criteria>"
        f"{revision_context}"
    )
    return Command(
        update={
            "planning_stage": PLANNING_STAGE_PLAN_FINALIZE,
            "planning_objective": objective,
            "planning_context_and_constraints": context,
            "planning_research_evidence": "",
            "planning_success_criteria": criteria,
            "messages": [
                ToolMessage(
                    content=result,
                    name="prepare_plan",
                    tool_call_id=str(runtime.tool_call_id or "prepare_plan"),
                )
            ],
        }
    )
