"""Default structured planning tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

PRESENT_PLAN_INTERRUPT_TYPE = "present_plan"


@tool(
    "present_plan",
    description=(
        "Present a concrete implementation plan to the user for review. "
        "Use this only in planning mode when the user explicitly asks for a plan, "
        "final review, or implementation-ready proposal. Do not use it for ordinary "
        "planning discussion or brainstorming. Fill every section: title, summary, "
        "key_changes, test_plan, and assumptions."
    ),
)
def present_plan(
    title: str,
    summary: list[str],
    key_changes: list[str],
    test_plan: list[str],
    assumptions: list[str],
) -> str:
    """Pause and present one structured implementation plan."""
    return str(
        interrupt(
            {
                "type": PRESENT_PLAN_INTERRUPT_TYPE,
                "title": clean_text(title) or "Implementation Plan",
                "summary": clean_items(summary, fallback="Summarize the intended change before implementation."),
                "key_changes": clean_items(key_changes, fallback="List the key implementation changes."),
                "test_plan": clean_items(test_plan, fallback="Describe the tests or checks to create."),
                "assumptions": clean_items(assumptions, fallback="No additional assumptions identified."),
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
    return items or [fallback]
