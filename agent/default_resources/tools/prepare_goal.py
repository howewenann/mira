"""Read-only preparation boundary for durable MIRA Goals."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PREPARE_GOAL_INTERRUPT_TYPE = "prepare_goal"
GOAL_FIELD_MAX_CHARS = 4000


@tool(
    "prepare_goal",
    description=(
        "Begin formal Goal construction when the Goal objective is sufficiently understood and every "
        "material user decision is resolved. Pass the authoritative objective plus concise relevant "
        "context, constraints, and bounded research evidence. MIRA will generate Success Criteria next. "
        "Do not produce or include an implementation Plan, return the Goal in prose, or add unsupported "
        "scope. Use ask_user first when user input is still required."
    ),
)
def prepare_goal(objective: str, context_and_constraints: str = "", research_evidence: str = "") -> str:
    """Pause Goal construction while MIRA generates Success Criteria."""
    return str(
        interrupt(
            {
                "type": PREPARE_GOAL_INTERRUPT_TYPE,
                "objective": str(objective or "").strip()[:GOAL_FIELD_MAX_CHARS],
                "context_and_constraints": str(context_and_constraints or "").strip()[:GOAL_FIELD_MAX_CHARS],
                "research_evidence": str(research_evidence or "").strip()[:GOAL_FIELD_MAX_CHARS],
            }
        )
    )
