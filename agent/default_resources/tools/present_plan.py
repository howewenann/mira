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
        "planning discussion or brainstorming."
    ),
)
def present_plan(
    title: str,
    summary: list[str],
    key_changes: list[str],
    assumptions: list[str] | None = None,
) -> str:
    """Pause and present one structured implementation plan."""
    return str(
        interrupt(
            {
                "type": PRESENT_PLAN_INTERRUPT_TYPE,
                "title": clean_text(title) or "Implementation Plan",
                "summary": clean_items(summary),
                "key_changes": clean_items(key_changes),
                "assumptions": clean_items(assumptions or []),
            }
        )
    )


def clean_text(value: str) -> str:
    """Return compact non-empty text."""
    return " ".join(str(value or "").split())


def clean_items(values: str | list[str]) -> list[str]:
    """Return compact non-empty list items."""
    if isinstance(values, str):
        values = [values]
    items = []
    for value in values:
        text = clean_text(str(value))
        if text:
            items.append(text)
    return items
