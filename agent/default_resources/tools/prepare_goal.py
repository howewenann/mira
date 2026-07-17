"""Default rubric-planning transition tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PREPARE_GOAL_INTERRUPT_TYPE = "prepare_goal"
RESEARCH_SUMMARY_MAX_CHARS = 4000


@tool(
    "prepare_goal",
    description=(
        "Signal that read-only research is complete and every material user decision has been resolved, "
        "so MIRA can generate acceptance criteria before you produce the final plan. Call this instead of "
        "present_plan on the first pass when rubric planning is enabled. Pass a concise research_summary of "
        "only material facts, constraints, existing behavior, and relevant context; omit plans, new requirements, "
        "and instructions, and use an empty string when no research was needed. After MIRA resumes the tool with "
        "approved criteria, create the final plan with present_plan."
    ),
)
def prepare_goal(research_summary: str = "") -> str:
    """Pause planning so MIRA can generate acceptance criteria."""
    summary = str(research_summary or "").strip()[:RESEARCH_SUMMARY_MAX_CHARS]
    return str(interrupt({"type": PREPARE_GOAL_INTERRUPT_TYPE, "research_summary": summary}))
