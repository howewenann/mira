"""Projection of native LangGraph root tasks into Workflow UI events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.interface import FrontendEmitter


@dataclass(slots=True)
class WorkflowTask:
    """Coordinator state for one native root task."""

    task_id: str
    name: str
    step: int
    status: str = "RUNNING"
    inspection_id: str = ""


class WorkflowCoordinator:
    """Infer visible workflow steps from a native ``TasksTransformer`` stream."""

    def __init__(self, emitter: FrontendEmitter, *, workflow_id: str = "") -> None:
        self.emitter = emitter
        self.workflow_id = workflow_id
        self._next_step = 0
        self._current_step = 0
        self._active: set[str] = set()
        self._tasks: dict[str, WorkflowTask] = {}
        self._started = False
        self._finished = False

    @property
    def tasks(self) -> tuple[WorkflowTask, ...]:
        """Return task identities in their first-seen order."""
        return tuple(self._tasks.values())

    def start(self) -> None:
        """Start a fresh process-local workflow projection."""
        self._next_step = 0
        self._current_step = 0
        self._active.clear()
        self._tasks.clear()
        self._started = True
        self._finished = False
        self.emitter.workflow_started(self.workflow_id)

    async def consume(self, tasks: Any) -> None:
        """Consume the root task extension without inspecting debug metadata."""
        if not self._started:
            self.start()
        async for event in tasks:
            if not isinstance(event, dict):
                continue
            task_id = str(event.get("id") or "")
            if not task_id:
                continue
            if "input" in event:
                self._task_started(task_id, str(event.get("name") or "node"))
            else:
                self._task_finished(
                    task_id,
                    event.get("error"),
                    event.get("interrupts"),
                )

    def bind_inspection(self, task_id: str, inspection_id: str) -> bool:
        """Bind a discovered live inspection to an existing Workflow task."""
        task = self._tasks.get(str(task_id or ""))
        value = str(inspection_id or "")
        if task is None or not value:
            return False
        if task.inspection_id == value:
            return True
        task.inspection_id = value
        self.emitter.workflow_task_inspection(
            task.task_id,
            task.name,
            task.step,
            value,
            workflow_id=self.workflow_id,
        )
        return True

    def finish(self) -> None:
        """Finish the run while preserving completed rows in the frontend."""
        if self._finished:
            return
        self._finished = True
        self.emitter.workflow_finished(self.workflow_id)

    def cancel(self, error: str = "") -> None:
        """Mark any active rows terminal after cancellation or run failure."""
        if self._finished:
            return
        self._finished = True
        self._active.clear()
        self.emitter.workflow_cancelled(self.workflow_id, error=error)

    def _task_started(self, task_id: str, name: str) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            if not self._active:
                self._next_step += 1
                self._current_step = self._next_step
            task = WorkflowTask(task_id, name, self._current_step)
            self._tasks[task_id] = task
            phase = "start"
        elif task.status == "WAITING":
            phase = "resume"
        else:
            phase = "start"
        task.status = "RUNNING"
        self._active.add(task_id)
        callback = (
            self.emitter.workflow_task_resumed
            if phase == "resume"
            else self.emitter.workflow_task_started
        )
        callback(task.task_id, task.name, task.step, workflow_id=self.workflow_id)

    def _task_finished(self, task_id: str, error: Any, interrupts: Any = None) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        self._active.discard(task_id)
        if interrupts:
            task.status = "WAITING"
            self.emitter.workflow_task_waiting(
                task.task_id,
                task.name,
                task.step,
                workflow_id=self.workflow_id,
            )
            return
        error_text = str(error or "")
        task.status = "ERROR" if error_text else "DONE"
        self.emitter.workflow_task_finished(
            task.task_id,
            task.name,
            task.step,
            status=task.status,
            error=error_text,
            workflow_id=self.workflow_id,
        )


__all__ = ["WorkflowCoordinator", "WorkflowTask"]
