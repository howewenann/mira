"""Default structured planning tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PRESENT_PLAN_INTERRUPT_TYPE = "present_plan"


@tool(
    "present_plan",
    description=(
        "Present the final concise Plan after MIRA has generated Success Criteria. "
        "This tool is required in the formal finalisation stage and is unavailable during "
        "ordinary Plan discussion and research. Supply title, key_changes, test_plan, and "
        "assumptions as the complete Plan around the binding Objective, Context and "
        "Constraints, and Success Criteria supplied by MIRA. Use plan_show to display the "
        "retained Plan and prepare_plan to construct a new or revised Plan. Do not add a "
        "Summary section."
    ),
)
def present_plan(
    title: str,
    key_changes: list[str],
    test_plan: list[str],
    assumptions: list[str],
) -> str:
    """Pause and present one complete Plan."""
    return str(
        interrupt(
            {
                "type": PRESENT_PLAN_INTERRUPT_TYPE,
                "title": clean_text(title) or "Plan",
                "key_changes": clean_items(key_changes, fallback="List the key implementation changes."),
                "test_plan": clean_items(test_plan, fallback="Describe the tests or checks to create."),
                "assumptions": clean_items(assumptions, fallback="No additional assumptions."),
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
