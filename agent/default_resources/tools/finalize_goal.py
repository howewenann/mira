"""Required finalization tool for durable MIRA Goals."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

FINALIZE_GOAL_INTERRUPT_TYPE = "finalize_goal"


@tool(
    "finalize_goal",
    return_direct=True,
    description=(
        "Finalize the Goal after MIRA has generated Success Criteria. This is the only tool "
        "available in Goal finalization and a call is required. Supply only a concise "
        "user-facing title; MIRA builds the Goal from the staged Objective, its binding "
        "request or revision context, "
        "and Success Criteria. Call show_goal immediately to display the retained Goal and "
        "prepare_goal only to construct a new or revised Goal. Do not produce a Plan or "
        "return the Goal in prose."
    ),
)
def finalize_goal(title: str, runtime: ToolRuntime[Any, dict]) -> str:
    """Pause and present one complete Goal."""
    state = runtime.state if isinstance(runtime.state, dict) else {}
    return str(
        interrupt(
            {
                "type": FINALIZE_GOAL_INTERRUPT_TYPE,
                "title": " ".join(str(title or "").split()) or "Goal",
                "objective": str(state.get("planning_objective") or ""),
                "success_criteria": str(state.get("planning_success_criteria") or ""),
            }
        )
    )
