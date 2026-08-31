"""Read-only control tool for reopening MIRA's authoritative current Goal."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

SHOW_GOAL_INTERRUPT_TYPE = "show_goal"


@tool(
    "show_goal",
    description=(
        "Immediately render the exact retained Goal using MIRA's normal Goal bubble. "
        "Use this whenever the user asks to show, reopen, review, or return to the "
        "current or previous Goal. Do not research, summarize, reproduce the Goal in "
        "prose, prepare a replacement, or finalize it first. 'Previous Goal' means "
        "MIRA's retained current_goal, not an arbitrary historical Goal. This tool only "
        "displays the Goal and does not alter its status."
    ),
)
def show_goal() -> str:
    """Pause so MIRA can render the retained Goal without model paraphrasing."""
    return str(interrupt({"type": SHOW_GOAL_INTERRUPT_TYPE}))
