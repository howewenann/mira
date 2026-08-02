"""Required final presentation tool for durable MIRA Goals."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PRESENT_GOAL_INTERRUPT_TYPE = "present_goal"


@tool(
    "present_goal",
    description=(
        "Present the final Goal after MIRA has generated Success Criteria. This is the only tool "
        "available in Goal finalisation and a call is required. Supply only a concise user-facing "
        "title; MIRA builds the Goal from the staged authoritative Objective and Success Criteria. "
        "Use goal_show to display the retained Goal and prepare_goal to construct a new or "
        "revised Goal. Do not produce a Plan or return the Goal in prose."
    ),
)
def present_goal(title: str) -> str:
    """Pause and present one complete Goal."""
    return str(
        interrupt(
            {
                "type": PRESENT_GOAL_INTERRUPT_TYPE,
                "title": " ".join(str(title or "").split()) or "Goal",
            }
        )
    )
