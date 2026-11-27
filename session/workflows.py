"""Durable, model-isolated Workflow execution history."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import replace

from core.interface.events import MessageEvent, WorkflowEvent
from typing import Any, Callable

from rich.pretty import pretty_repr


WORKFLOW_STATUSES = {
    "RUNNING",
    "WAITING",
    "DONE",
    "ERROR",
    "CANCELLED",
    "INTERRUPTED",
}
ACTIVE_WORKFLOW_STATUSES = {"RUNNING", "WAITING"}


def freeze_workflow_value(value: Any) -> dict[str, str]:
    """Freeze one arbitrary Python value into JSON-safe Rich representations."""
    try:
        display_text = pretty_repr(value, expand_all=False)
        copy_text = pretty_repr(value, expand_all=True)
        if display_text.startswith("<repr-error ") or copy_text.startswith(
            "<repr-error "
        ):
            raise ValueError("Rich reported a representation failure")
    except BaseException as exc:  # a hostile __repr__ must not affect execution
        _warn("workflow value representation failed: %s", exc)
        fallback = f"<unrepresentable {type(value).__name__}>"
        return {"display_text": fallback, "copy_text": fallback}
    return {"display_text": display_text, "copy_text": copy_text}


def normalize_workflow_event(value: Any) -> dict[str, Any] | None:
    """Return one explicit, JSON-safe Workflow transcript payload."""
    if not isinstance(value, dict):
        return None
    workflow_id = str(value.get("workflow_id") or "")
    if not workflow_id:
        return None
    created_at = str(value.get("created_at") or _now_iso())
    updated_at = str(value.get("updated_at") or created_at)
    status = _status(value.get("status"), default="RUNNING")
    tasks: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    raw_tasks = value.get("tasks")
    if isinstance(raw_tasks, list):
        for item in raw_tasks:
            task = _normalize_task(item)
            if task is None or task["task_id"] in seen_tasks:
                continue
            seen_tasks.add(task["task_id"])
            tasks.append(task)
    final_state_available = value.get("final_state_available") is True
    return {
        "workflow_id": workflow_id,
        "workflow_name": str(value.get("workflow_name") or workflow_id),
        "command": str(value.get("command") or ""),
        "status": status,
        "error": str(value.get("error") or ""),
        "updated_at": updated_at,
        "finished_at": str(value.get("finished_at") or ""),
        "duration_ms": _duration(value.get("duration_ms")),
        "tasks": tasks,
        "final_state_available": final_state_available,
        "final_state": (
            _normalize_frozen_value(value.get("final_state"))
            if final_state_available
            else None
        ),
    }


def reconcile_stale_workflows(record: dict[str, Any]) -> bool:
    """Freeze Workflow rows left active by a vanished process."""
    events = record.get("events")
    if not isinstance(events, list):
        return False
    changed = False
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "workflow":
            continue
        if str(event.get("status") or "") not in ACTIVE_WORKFLOW_STATUSES:
            continue
        changed = True
        event["status"] = "INTERRUPTED"
        event["finished_at"] = str(
            event.get("updated_at") or event.get("created_at") or _now_iso()
        )
        tasks = event.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("status") or "") in ACTIVE_WORKFLOW_STATUSES:
                task["status"] = "INTERRUPTED"
            agents = task.get("agents")
            if not isinstance(agents, list):
                continue
            for agent in agents:
                if (
                    isinstance(agent, dict)
                    and str(agent.get("status") or "") in ACTIVE_WORKFLOW_STATUSES
                ):
                    agent["status"] = "INTERRUPTED"
    return changed


class PersistentWorkflowHistory:
    """Observe one live Workflow projection and save its retrospective shape."""

    def __init__(
        self,
        frontend: Any,
        record: dict[str, Any],
        store: Any,
        *,
        command: str,
        record_user: bool = True,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.frontend = frontend
        self.record = record
        self.store = store
        self.command = command
        self.record_user = record_user
        self._event_index: int | None = None
        self._clock = clock
        self._now = now or _now_iso
        self._workflow_id = ""
        self._event: dict[str, Any] | None = None
        self._run_started = 0.0
        self._task_started: dict[str, float] = {}
        self._agent_started: dict[str, float] = {}
        self._pending_agents: dict[str, list[dict[str, Any]]] = {}

    def emit(self, event: Any) -> None:
        """Persist typed workflow observations before forwarding them once."""
        if isinstance(event, WorkflowEvent):
            self._observe(lambda: self._workflow_event(event))
            if self._event is not None:
                event = replace(event, event_id=self._event["id"], created_at=self._event["created_at"])
                if event.phase == "run_start" and self.record_user:
                    self.frontend.emit(MessageEvent(phase="user", text=self.command,
                        session_id=event.session_id, turn_id=event.turn_id,
                        event_id=event.event_id, created_at=event.created_at))
        self.frontend.emit(event)

    async def request(self, request: Any) -> Any:
        return await self.frontend.request(request)

    def _workflow_event(self, event: WorkflowEvent) -> None:
        if event.phase == "run_start":
            self._start(event.workflow_id, event.workflow_name)
            self._event["turn_id"] = event.turn_id
        elif event.phase == "task_start":
            self._start_task(event.task_id, event.name, event.step, event.input_state)
        elif event.phase in {"task_waiting", "task_resume"}:
            self._set_task_status(event.task_id, "WAITING" if event.phase == "task_waiting" else "RUNNING")
        elif event.phase == "task_finish":
            self._finish_task(event.task_id, event.status, event.error, event.result, event.result_available)
        elif event.phase in {"agent_start", "agent_resume"}:
            self._start_agent(event.task_id, event.name, event.inspection_id, event.task_input)
        elif event.phase == "agent_waiting":
            self._set_agent_status(event.inspection_id, "WAITING")
        elif event.phase == "agent_finish":
            self._finish_agent(event.inspection_id, event.status, event.result, event.error)
        elif event.phase == "run_finish":
            self._finish(event.final_state, event.final_state_available)
        elif event.phase == "run_cancel":
            self._cancel(event.error)

    def close(self) -> None:
        """Release the retained live event after the observer is detached."""
        self._workflow_id = ""
        self._event = None
        self._event_index = None

    def _observe(self, update: Callable[[], None]) -> None:
        try:
            self._rebind_event()
            if self._event is not None:
                self._refresh_durations()
            update()
            if self._event is not None:
                self._event["updated_at"] = self._now()
                self._save()
        except BaseException as exc:  # history must remain observational
            _warn("workflow history projection failed: %s", exc)

    def _start(self, workflow_id: str, workflow_name: str) -> None:
        now = self._now()
        self._workflow_id = workflow_id
        self._run_started = self._clock()
        events = self.record.setdefault("events", [])
        next_id = max(
            (
                int(item.get("id") or 0)
                for item in events
                if isinstance(item, dict)
            ),
            default=0,
        ) + 1
        self._event = {
            "id": next_id,
            "type": "workflow",
            "created_at": now,
            "updated_at": now,
            "finished_at": "",
            "workflow_id": workflow_id,
            "workflow_name": workflow_name or workflow_id,
            "command": self.command,
            "status": "RUNNING",
            "error": "",
            "duration_ms": 0,
            "tasks": [],
            "final_state_available": False,
            "final_state": None,
        }
        self._event_index = len(events)
        events.append(self._event)

    def _start_task(
        self, task_id: str, name: str, step: int, input_state: Any
    ) -> None:
        if self._event is None:
            return
        task = self._task(task_id)
        if task is None:
            task = {
                "task_id": task_id,
                "name": name,
                "step": max(0, int(step)),
                "status": "RUNNING",
                "error": "",
                "duration_ms": 0,
                "input_state": freeze_workflow_value(input_state),
                "result_available": False,
                "result": None,
                "agents": [],
            }
            self._event["tasks"].append(task)
            self._task_started[task_id] = self._clock()
            for agent in self._pending_agents.pop(task_id, []):
                task["agents"].append(agent)
        else:
            task["status"] = "RUNNING"
            task["error"] = ""
        self._event["status"] = "RUNNING"

    def _set_task_status(self, task_id: str, status: str) -> None:
        task = self._task(task_id)
        if task is not None:
            task["status"] = status
            task["error"] = ""
        if self._event is not None:
            self._event["status"] = status

    def _finish_task(
        self,
        task_id: str,
        status: str,
        error: str,
        result: Any,
        result_available: bool,
    ) -> None:
        task = self._task(task_id)
        if task is None:
            return
        task["status"] = _status(status, default="ERROR")
        task["error"] = str(error or "")
        task["result_available"] = bool(result_available)
        task["result"] = (
            freeze_workflow_value(result) if result_available else None
        )
        if self._event is not None and self._event["status"] != "WAITING":
            self._event["status"] = "RUNNING"

    def _start_agent(
        self, task_id: str, name: str, inspection_id: str, task_input: str
    ) -> None:
        agent = self._agent(inspection_id)
        if agent is None:
            agent = {
                "inspection_id": inspection_id,
                "name": name,
                "task_input": str(task_input or ""),
                "status": "RUNNING",
                "result": "",
                "error": "",
                "duration_ms": 0,
            }
            task = self._task(task_id)
            if task is None:
                self._pending_agents.setdefault(task_id, []).append(agent)
            else:
                task["agents"].append(agent)
            self._agent_started[inspection_id] = self._clock()
        else:
            agent["status"] = "RUNNING"
            agent["error"] = ""
        if self._event is not None:
            self._event["status"] = "RUNNING"

    def _set_agent_status(self, inspection_id: str, status: str) -> None:
        agent = self._agent(inspection_id)
        if agent is not None:
            agent["status"] = status
        if self._event is not None:
            self._event["status"] = status

    def _finish_agent(
        self, inspection_id: str, status: str, result: str, error: str
    ) -> None:
        agent = self._agent(inspection_id)
        if agent is None:
            return
        agent["status"] = _status(status, default="ERROR")
        agent["result"] = str(result or "")
        agent["error"] = str(error or "")
        if self._event is not None and self._event["status"] != "WAITING":
            self._event["status"] = "RUNNING"

    def _finish(self, final_state: Any, final_state_available: bool) -> None:
        if self._event is None:
            return
        self._event["status"] = "DONE"
        self._event["finished_at"] = self._now()
        self._event["final_state_available"] = bool(final_state_available)
        self._event["final_state"] = (
            freeze_workflow_value(final_state)
            if final_state_available
            else None
        )

    def _cancel(self, error: str) -> None:
        if self._event is None:
            return
        status = "ERROR" if error else "CANCELLED"
        self._event["status"] = status
        self._event["error"] = str(error or "")
        self._event["finished_at"] = self._now()
        for task in self._event["tasks"]:
            if task["status"] in ACTIVE_WORKFLOW_STATUSES:
                task["status"] = status
                task["error"] = str(error or "")
            for agent in task["agents"]:
                if agent["status"] in ACTIVE_WORKFLOW_STATUSES:
                    agent["status"] = status
                    agent["error"] = str(error or "")

    def _refresh_durations(self) -> None:
        if self._event is None:
            return
        now = self._clock()
        if self._event["status"] in ACTIVE_WORKFLOW_STATUSES:
            self._event["duration_ms"] = _elapsed(self._run_started, now)
        for task in self._event["tasks"]:
            task_id = task["task_id"]
            if (
                task["status"] in ACTIVE_WORKFLOW_STATUSES
                and task_id in self._task_started
            ):
                task["duration_ms"] = _elapsed(self._task_started[task_id], now)
            for agent in task["agents"]:
                inspection_id = agent["inspection_id"]
                if (
                    agent["status"] in ACTIVE_WORKFLOW_STATUSES
                    and inspection_id in self._agent_started
                ):
                    agent["duration_ms"] = _elapsed(
                        self._agent_started[inspection_id], now
                    )
        for agents in self._pending_agents.values():
            for agent in agents:
                inspection_id = agent["inspection_id"]
                if inspection_id in self._agent_started:
                    agent["duration_ms"] = _elapsed(
                        self._agent_started[inspection_id], now
                    )

    def _task(self, task_id: str) -> dict[str, Any] | None:
        if self._event is None:
            return None
        return next(
            (task for task in self._event["tasks"] if task["task_id"] == task_id),
            None,
        )

    def _agent(self, inspection_id: str) -> dict[str, Any] | None:
        if self._event is not None:
            for task in self._event["tasks"]:
                for agent in task["agents"]:
                    if agent["inspection_id"] == inspection_id:
                        return agent
        for agents in self._pending_agents.values():
            for agent in agents:
                if agent["inspection_id"] == inspection_id:
                    return agent
        return None

    def _save(self) -> None:
        try:
            self.store.save(self.record)
        except BaseException as exc:  # persistence must not fail execution
            _warn("workflow history persistence failed: %s", exc)
            return
        self._rebind_event()
        if self._event is not None and self._event["status"] in {
            "DONE", "ERROR", "CANCELLED", "INTERRUPTED"
        }:
            self.record["resume_context_pending"] = True

    def _rebind_event(self) -> None:
        """Resolve the canonical event after any observer may have saved."""
        if not self._workflow_id:
            return
        events = self.record.get("events", [])
        if self._event_index is not None and self._event_index < len(events):
            candidate = events[self._event_index]
            if candidate.get("workflow_id") == self._workflow_id:
                self._event = candidate
                return
        self._event = None
        for index, event in enumerate(events):
            if isinstance(event, dict) and event.get("type") == "workflow" and event.get("workflow_id") == self._workflow_id:
                self._event, self._event_index = event, index
                break


def _normalize_task(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    task_id = str(value.get("task_id") or "")
    if not task_id:
        return None
    agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    raw_agents = value.get("agents")
    if isinstance(raw_agents, list):
        for item in raw_agents:
            agent = _normalize_agent(item)
            if agent is None or agent["inspection_id"] in seen_agents:
                continue
            seen_agents.add(agent["inspection_id"])
            agents.append(agent)
    result_available = value.get("result_available") is True
    try:
        step = max(0, int(value.get("step") or 0))
    except (TypeError, ValueError):
        step = 0
    return {
        "task_id": task_id,
        "name": str(value.get("name") or "node"),
        "step": step,
        "status": _status(value.get("status"), default="ERROR"),
        "error": str(value.get("error") or ""),
        "duration_ms": _duration(value.get("duration_ms")),
        "input_state": _normalize_frozen_value(value.get("input_state")),
        "result_available": result_available,
        "result": (
            _normalize_frozen_value(value.get("result"))
            if result_available
            else None
        ),
        "agents": agents,
    }


def _normalize_agent(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    inspection_id = str(value.get("inspection_id") or "")
    if not inspection_id:
        return None
    return {
        "inspection_id": inspection_id,
        "name": str(value.get("name") or "agent"),
        "task_input": str(value.get("task_input") or ""),
        "status": _status(value.get("status"), default="ERROR"),
        "result": str(value.get("result") or ""),
        "error": str(value.get("error") or ""),
        "duration_ms": _duration(value.get("duration_ms")),
    }


def _normalize_frozen_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        display_text = value.get("display_text")
        copy_text = value.get("copy_text")
        if isinstance(display_text, str) and isinstance(copy_text, str):
            return {"display_text": display_text, "copy_text": copy_text}
    fallback = "<unavailable Workflow value>"
    return {"display_text": fallback, "copy_text": fallback}


def _status(value: Any, *, default: str) -> str:
    status = str(value or default).upper()
    return status if status in WORKFLOW_STATUSES else default


def _duration(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, round(value))
    return 0


def _elapsed(started: float, now: float) -> int:
    return max(0, round((now - started) * 1000))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _warn(message: str, error: BaseException) -> None:
    try:
        from core.diagnostics.logging import get_diagnostics_logger

        get_diagnostics_logger().warning(message, error)
    except BaseException:
        pass


__all__ = [
    "PersistentWorkflowHistory",
    "freeze_workflow_value",
    "normalize_workflow_event",
    "reconcile_stale_workflows",
]
