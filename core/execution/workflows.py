"""Projection of native LangGraph root tasks into Workflow UI events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command

from core.execution.inspection.subagents import SubagentInspectionCoordinator
from core.execution.runner import (
    annotate_filesystem_approvals,
    approval_resume_value,
    first_typed_interrupt,
    resolve_approval_decisions,
)
from core.execution.streams.output import capture_output, collect_interrupts
from core.execution.streams.subagents import consume_workflow_inspections
from core.execution.streams.tools import consume_live_tool_errors
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


async def execute_workflow(
    graph: Any,
    payload: Any,
    *,
    emitter: FrontendEmitter,
    inspection: SubagentInspectionCoordinator,
    workflow_id: str,
    config: dict[str, Any] | None = None,
    context: Any | None = None,
    inspection_title: str = "",
    action_agent: Any | None = None,
    persist_always_allow: Callable[[Any, dict[str, Any]], None] | None = None,
) -> Any:
    """Run one native graph through the established Workflow observation path."""
    coordinator = WorkflowCoordinator(emitter, workflow_id=workflow_id)
    coordinator.start()
    current_payload = payload
    run_config = dict(config or {})
    configurable = dict(run_config.get("configurable") or {})
    configurable.setdefault("thread_id", f"workflow:{uuid4()}")
    run_config["configurable"] = configurable
    always_allowed_tools: set[str] = set()
    try:
        while True:
            inspection.begin_pass()
            kwargs = {
                "config": run_config,
                "version": "v3",
                "transformers": [TasksTransformer],
            }
            if context is not None:
                kwargs["context"] = context
            run = await graph.astream_events(current_payload, **kwargs)
            tasks = run.extensions.get("tasks")
            if tasks is None:
                raise RuntimeError(
                    "Workflow did not expose the native task projection."
                )
            output: dict[str, Any] = {}
            async with run:
                await asyncio.gather(
                    coordinator.consume(tasks),
                    consume_live_tool_errors(
                        run,
                        emitter,
                        subagent_capture=inspection,
                    ),
                    consume_workflow_inspections(
                        run.subgraphs,
                        inspection,
                        coordinator.bind_inspection,
                        title=inspection_title,
                    ),
                    capture_output(run.output(), output),
                )

            final_state = output.get("value")
            interrupts = await collect_interrupts(run, final_state)
            if not interrupts:
                coordinator.finish()
                return final_state

            ask_user_interrupt = first_typed_interrupt(interrupts, "ask_user")
            mcp_interrupt = first_typed_interrupt(interrupts, "mcp_elicitation")
            if ask_user_interrupt is not None:
                answer = await emitter.ask_user(ask_user_interrupt)
                current_payload = Command(resume=answer)
                continue
            if mcp_interrupt is not None:
                answer = await emitter.answer_mcp_elicitation(mcp_interrupt)
                current_payload = Command(resume=answer)
                continue

            backend = getattr(action_agent, "mira_backend", None)
            annotate_filesystem_approvals(interrupts, backend)
            decisions = await resolve_approval_decisions(
                emitter,
                interrupts,
                action_agent,
                always_allowed_tools,
                persist_always_allow,
            )
            current_payload = Command(
                resume=approval_resume_value(interrupts, decisions)
            )
    except asyncio.CancelledError:
        inspection.cancel_standalone()
        coordinator.cancel()
        raise
    except Exception as error:
        inspection.cancel_standalone(str(error))
        coordinator.cancel(str(error))
        raise


__all__ = ["WorkflowCoordinator", "WorkflowTask"]
