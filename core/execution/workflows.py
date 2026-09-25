"""Projection of native LangGraph root tasks into Workflow UI events."""

from __future__ import annotations

import asyncio
from copy import deepcopy
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


_MISSING = object()


@dataclass(slots=True)
class WorkflowTask:
    """Coordinator state for one native root task."""

    task_id: str
    name: str
    step: int
    status: str = "RUNNING"
    input_state: Any = _MISSING
    result: Any = _MISSING
    error: str = ""


class WorkflowCoordinator:
    """Infer visible workflow steps from a native ``TasksTransformer`` stream."""

    def __init__(
        self,
        emitter: FrontendEmitter,
        *,
        workflow_id: str = "",
        workflow_name: str = "",
    ) -> None:
        self.emitter = emitter
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name or workflow_id
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
        self.emitter.workflow_started(self.workflow_id, self.workflow_name)

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
                self._task_started(
                    task_id,
                    str(event.get("name") or "node"),
                    event["input"],
                )
            else:
                self._task_finished(
                    task_id,
                    event.get("error"),
                    event.get("interrupts"),
                    result=event.get("result", _MISSING),
                )

    def agent_started(
        self,
        task_id: str,
        inspection_id: str,
        name: str,
        task_input: str,
        *,
        resumed: bool = False,
    ) -> None:
        """Project one observed child agent beneath its native root task."""
        self.emitter.workflow_agent_started(
            task_id,
            name,
            inspection_id,
            task_input=task_input,
            resumed=resumed,
            workflow_id=self.workflow_id,
        )

    def agent_waiting(
        self,
        task_id: str,
        inspection_id: str,
        name: str,
    ) -> None:
        """Keep an interrupted child agent live across the Workflow resume."""
        self.emitter.workflow_agent_waiting(
            task_id,
            name,
            inspection_id,
            workflow_id=self.workflow_id,
        )

    def agent_finished(
        self,
        task_id: str,
        inspection_id: str,
        name: str,
        *,
        status: str,
        result: str = "",
        error: str = "",
    ) -> None:
        """Project one observed child agent's terminal status."""
        self.emitter.workflow_agent_finished(
            task_id,
            name,
            inspection_id,
            status=status,
            result=result,
            error=error,
            workflow_id=self.workflow_id,
        )

    def finish(self, final_state: Any) -> None:
        """Finish the run while preserving completed rows in the frontend."""
        if self._finished:
            return
        self._finished = True
        self.emitter.workflow_finished(
            self.workflow_id,
            final_state=_retain_value(final_state),
            final_state_available=True,
        )

    def cancel(self, error: str = "") -> None:
        """Mark any active rows terminal after cancellation or run failure."""
        if self._finished:
            return
        self._finished = True
        self._active.clear()
        self.emitter.workflow_cancelled(self.workflow_id, error=error)

    def _task_started(self, task_id: str, name: str, input_state: Any) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            if not self._active:
                self._next_step += 1
                self._current_step = self._next_step
            task = WorkflowTask(
                task_id,
                name,
                self._current_step,
                input_state=_retain_value(input_state),
            )
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
        kwargs: dict[str, Any] = {"workflow_id": self.workflow_id}
        if phase == "start":
            kwargs["input_state"] = task.input_state
        callback(task.task_id, task.name, task.step, **kwargs)

    def _task_finished(
        self,
        task_id: str,
        error: Any,
        interrupts: Any = None,
        *,
        result: Any = _MISSING,
    ) -> None:
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
        task.error = error_text
        if not error_text and result is not _MISSING:
            task.result = _retain_value(_native_task_result(result))
        self.emitter.workflow_task_finished(
            task.task_id,
            task.name,
            task.step,
            status=task.status,
            error=error_text,
            result=None if task.result is _MISSING else task.result,
            result_available=task.result is not _MISSING,
            workflow_id=self.workflow_id,
        )


async def execute_workflow(
    graph: Any,
    payload: Any,
    *,
    emitter: FrontendEmitter,
    inspection: SubagentInspectionCoordinator,
    workflow_id: str,
    workflow_name: str = "",
    config: dict[str, Any] | None = None,
    context: Any | None = None,
    inspection_title: str = "",
    action_agent: Any | None = None,
    persist_always_allow: Callable[[Any, dict[str, Any]], None] | None = None,
) -> Any:
    """Run one native graph through the established Workflow observation path."""
    coordinator = WorkflowCoordinator(
        emitter,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
    )
    inspection.observe_workflow_agents(
        coordinator.agent_started,
        coordinator.agent_waiting,
        coordinator.agent_finished,
    )
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
                        coordinator.agent_started,
                        coordinator.agent_waiting,
                        coordinator.agent_finished,
                        title=inspection_title,
                    ),
                    capture_output(run.output(), output),
                )

            final_state = output.get("value")
            interrupts = await collect_interrupts(run, final_state)
            if not interrupts:
                coordinator.finish(final_state)
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


def _native_task_result(value: Any) -> Any:
    """Unwrap only LangGraph's single-root task-result carrier."""
    if isinstance(value, dict) and list(value) == ["__root__"]:
        return value["__root__"]
    return value


def _retain_value(value: Any) -> Any:
    """Keep one observed runtime boundary stable if its source later mutates."""
    try:
        return deepcopy(value)
    except Exception:
        return value


__all__ = ["WorkflowCoordinator", "WorkflowTask"]
