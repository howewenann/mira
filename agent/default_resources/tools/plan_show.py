"""Read-only control tool for reopening MIRA's authoritative current Plan."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PLAN_SHOW_INTERRUPT_TYPE = "plan_show"


@tool(
    "plan_show",
    description=(
        "Render the exact current Plan using MIRA's normal Plan bubble. Use this when "
        "the user asks to show, reopen, review, revise, run, rerun, implement, or return "
        "to the current or previous Plan. Do not summarize or reproduce the Plan in "
        "prose. 'Previous Plan' means MIRA's retained current_plan, not arbitrary "
        "historical plans. This tool only displays the Plan and does not alter its status."
    ),
)
def plan_show() -> str:
    """Pause so MIRA can render the retained Plan without model paraphrasing."""
    return str(interrupt({"type": PLAN_SHOW_INTERRUPT_TYPE}))
