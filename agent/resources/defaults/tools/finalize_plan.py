"""Required finalization tool for durable MIRA Plans."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

FINALIZE_PLAN_INTERRUPT_TYPE = "finalize_plan"


@tool(
    "finalize_plan",
    return_direct=True,
    description=(
        "Finalize the concise Plan after MIRA has generated Success Criteria. "
        "This tool is required in the formal finalisation stage and is unavailable during "
        "ordinary Plan discussion and research. Supply title, key_changes, test_plan, and "
        "assumptions as the complete Plan around the binding Objective, Context and "
        "Constraints, and Success Criteria supplied by MIRA. Call show_plan immediately "
        "to display the retained Plan and prepare_plan only to construct a new or revised "
        "Plan. Do not add a Summary section."
    ),
)
def finalize_plan(
    title: str,
    key_changes: list[str],
    test_plan: list[str],
    assumptions: list[str],
    runtime: ToolRuntime[Any, dict],
) -> str:
    """Pause and present one complete Plan."""
    state = runtime.state if isinstance(runtime.state, dict) else {}
    return str(
        interrupt(
            {
                "type": FINALIZE_PLAN_INTERRUPT_TYPE,
                "title": clean_text(title) or "Plan",
                "key_changes": clean_items(key_changes, fallback="List the key implementation changes."),
                "test_plan": clean_items(test_plan, fallback="Describe the tests or checks to create."),
                "assumptions": clean_items(assumptions, fallback="No additional assumptions."),
                "objective": str(state.get("planning_objective") or ""),
                "context_and_constraints": str(
                    state.get("planning_context_and_constraints") or ""
                ),
                "success_criteria": str(state.get("planning_success_criteria") or ""),
            }
        )
    )


def clean_text(value: str) -> str:
    """Return compact non-empty text."""
    return " ".join(str(value or "").split())


def clean_items(values: str | list[str], *, fallback: str) -> list[str]:
    """Return compact non-empty list items."""
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = []
    items = []
    for value in values:
        text = clean_text(str(value))
        if text:
            items.append(text)
    return items or ([fallback] if fallback else [])
