"""Runtime dependencies for criteria-first Plan and Goal tools."""

from __future__ import annotations

from dataclasses import dataclass

from agent.planning.criteria import SuccessCriteriaService


@dataclass(frozen=True)
class PlanningToolContext:
    """Dependencies injected into formal planning tools for one graph run."""

    success_criteria: SuccessCriteriaService
