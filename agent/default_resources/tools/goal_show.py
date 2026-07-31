"""Read-only control tool for reopening MIRA's authoritative current Goal."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

GOAL_SHOW_INTERRUPT_TYPE = "goal_show"


@tool(
    "goal_show",
    description=(
        "Render the exact current Goal using MIRA's normal Goal bubble. Use this when the user "
        "asks to show, reopen, review, revise, run, rerun, implement, or return to the current "
        "or previous Goal. Do not summarize or reproduce the Goal in prose. 'Previous Goal' "
        "means MIRA's retained current_goal, not an arbitrary historical Goal. This tool only "
        "displays the Goal and does not alter its status."
    ),
)
def goal_show() -> str:
    """Pause so MIRA can render the retained Goal without model paraphrasing."""
    return str(interrupt({"type": GOAL_SHOW_INTERRUPT_TYPE}))
