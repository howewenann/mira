"""Read-only preparation boundary for durable MIRA Plans."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PREPARE_PLAN_INTERRUPT_TYPE = "prepare_plan"
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
def prepare_plan(objective: str, context_and_constraints: str = "") -> str:
    """Pause Plan construction while MIRA generates Success Criteria."""
    return str(
        interrupt(
            {
                "type": PREPARE_PLAN_INTERRUPT_TYPE,
                "objective": str(objective or "").strip()[:PLAN_FIELD_MAX_CHARS],
                "context_and_constraints": str(context_and_constraints or "").strip()[
                    :PLAN_FIELD_MAX_CHARS
                ],
            }
        )
    )
