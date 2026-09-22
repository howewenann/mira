"""Native workflow execution events for internal frontend projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.interface.events.base import EventIdentity


WorkflowPhase = Literal[
    "run_start",
    "task_start",
    "task_waiting",
    "task_resume",
    "task_inspection",
    "task_finish",
    "run_finish",
    "run_cancel",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowEvent(EventIdentity):
    """One process-local native workflow lifecycle update."""

    phase: WorkflowPhase
    workflow_id: str = ""
    task_id: str = ""
    name: str = ""
    step: int = 0
    status: str = ""
    error: str = ""
    inspection_id: str = ""


__all__ = ["WorkflowEvent", "WorkflowPhase"]
