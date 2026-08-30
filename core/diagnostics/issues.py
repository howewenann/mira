"""Startup and configuration issue diagnostics exposed by Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IssueCategory = Literal["STARTUP", "MODEL", "MCP", "TOOL"]
CATEGORY_ORDER: dict[IssueCategory, int] = {
    "STARTUP": 0,
    "MODEL": 1,
    "MCP": 2,
    "TOOL": 3,
}


@dataclass(frozen=True, slots=True)
class Issue:
    """One user-actionable problem discovered without aborting startup."""

    category: IssueCategory
    summary: str
    location: str = ""
    details: str = ""
    guidance: str = ""


def sort_issues(issues: list[Issue]) -> list[Issue]:
    """Sort by stable category order while preserving discovery order."""
    return sorted(issues, key=lambda issue: CATEGORY_ORDER[issue.category])


__all__ = ["CATEGORY_ORDER", "Issue", "IssueCategory", "sort_issues"]
