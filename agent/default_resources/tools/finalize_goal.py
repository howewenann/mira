"""Required finalization tool for durable MIRA Goals."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

FINALIZE_GOAL_INTERRUPT_TYPE = "finalize_goal"


@tool(
    "finalize_goal",
    description=(
        "Finalize the Goal after MIRA has generated Success Criteria. This is the only tool "
        "available in Goal finalization and a call is required. Supply only a concise "
        "user-facing title; MIRA builds the Goal from the staged authoritative Objective "
        "and Success Criteria. Call show_goal immediately to display the retained Goal and "
        "prepare_goal only to construct a new or revised Goal. Do not produce a Plan or "
        "return the Goal in prose."
    ),
)
def finalize_goal(title: str) -> str:
    """Pause and present one complete Goal."""
    return str(
        interrupt(
            {
                "type": FINALIZE_GOAL_INTERRUPT_TYPE,
                "title": " ".join(str(title or "").split()) or "Goal",
            }
        )
    )
