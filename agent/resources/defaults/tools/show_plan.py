"""Read-only control tool for reopening MIRA's authoritative current Plan."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

SHOW_PLAN_INTERRUPT_TYPE = "show_plan"


@tool(
    "show_plan",
    description=(
        "Immediately render the exact retained Plan using MIRA's normal Plan bubble. "
        "Use this whenever the user asks to show, reopen, review, or return to the "
        "current or previous Plan. Do not research, summarize, reproduce the Plan in "
        "prose, prepare a replacement, or finalize it first. 'Previous Plan' means "
        "MIRA's retained current_plan, not arbitrary historical plans. This tool only "
        "displays the Plan and does not alter its status."
    ),
)
def show_plan() -> str:
    """Pause so MIRA can render the retained Plan without model paraphrasing."""
    return str(interrupt({"type": SHOW_PLAN_INTERRUPT_TYPE}))
