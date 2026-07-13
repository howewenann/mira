"""Default rubric-planning transition tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PREPARE_GOAL_INTERRUPT_TYPE = "prepare_goal"


@tool(
    "prepare_goal",
    description=(
        "Signal that read-only research is complete and every material user decision has been resolved, "
        "so MIRA can generate acceptance criteria before you produce the final plan. Call this instead of "
        "present_plan on the first pass when rubric planning is enabled. It takes no plan or criteria. After "
        "MIRA resumes the tool with approved criteria, create the final plan with present_plan."
    ),
)
def prepare_goal() -> str:
    """Pause planning so MIRA can generate acceptance criteria."""
    return str(interrupt({"type": PREPARE_GOAL_INTERRUPT_TYPE}))
