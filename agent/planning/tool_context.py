"""Runtime dependencies for criteria-first Plan and Goal tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.execution.agents import BoundAgentMap
from agent.execution.mappings import ImmutableDict
from agent.execution.tools import BoundToolMap
from agent.planning.criteria import SuccessCriteriaService


@dataclass(frozen=True, slots=True)
class PlanningToolContext:
    """Dependencies injected into formal planning tools for one graph run."""

    success_criteria: SuccessCriteriaService
    tools: BoundToolMap = field(default_factory=ImmutableDict)
    agents: BoundAgentMap = field(default_factory=ImmutableDict)
