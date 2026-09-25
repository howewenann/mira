"""Native workflow execution events for internal frontend projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.interface.events.base import EventIdentity


WorkflowPhase = Literal[
    "run_start",
    "task_start",
    "task_waiting",
    "task_resume",
    "task_finish",
    "agent_start",
    "agent_waiting",
    "agent_resume",
    "agent_finish",
    "run_finish",
    "run_cancel",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowEvent(EventIdentity):
    """One process-local native workflow lifecycle update."""

    phase: WorkflowPhase
    workflow_id: str = ""
    workflow_name: str = ""
    task_id: str = ""
    agent_id: str = ""
    name: str = ""
    step: int = 0
    status: str = ""
    error: str = ""
    inspection_id: str = ""
    task_input: str = ""
    input_state: Any = None
    result: Any = None
    result_available: bool = False
    final_state: Any = None
    final_state_available: bool = False


__all__ = ["WorkflowEvent", "WorkflowPhase"]
