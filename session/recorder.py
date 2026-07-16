"""Incremental session persistence for visible runtime events."""

from __future__ import annotations

import asyncio
import json
from inspect import Parameter, signature
from typing import Any

from agent.context_overflow import pop_context_overflow_notice
from runtime.output_events import normalize_response_delta
from runtime.tool_events import CONTROL_TOOLS
from session.context import append_event, sync_deepagents_compaction, update_event_text

COMPACTION_POLL_SECONDS = 10.0


class SessionRecorder:
    """Persist visible turn events while forwarding rendering elsewhere."""

    def __init__(self, record: dict[str, Any], store: Any, mode: str) -> None:
        self.record = record
        self.store = store
        self.mode = mode
        self._assistant_id: int | None = None
        self._assistant_text = ""
        self._assistant_seen = False
        self._last_assistant_id: int | None = None
        self._reasoning_id: int | None = None
        self._reasoning_text = ""
        self._running_subagents: dict[str, str] = {}
        self._running_subagent_origins: dict[str, str] = {}
        self._running_subagent_event_ids: dict[str, int] = {}
        self._rubric_event_ids: dict[tuple[str, int], int] = {}
        self._delegation_keys: set[tuple[str, str]] = set()

    def save(self) -> None:
        self.store.save(self.record)

    def user_message(self, text: str) -> dict[str, Any]:
        return append_event(self.record, {"type": "user", "mode": self.mode, "text": text})

    def text_delta(self, delta: str) -> dict[str, Any] | None:
        delta = normalize_response_delta(self._assistant_text, delta)
        if not delta:
            return None
        self._close_reasoning_phase()
        event = None
        if self._assistant_id is None:
            event = append_event(self.record, {"type": "assistant", "mode": self.mode, "text": ""})
            self._assistant_id = int(event["id"])
            self._last_assistant_id = self._assistant_id
            self._assistant_text = ""
            self._assistant_seen = True
        self._assistant_text += str(delta)
        update_event_text(self.record, self._assistant_id, self._assistant_text)
        self.save()
        return event

    def reasoning_delta(self, delta: str) -> dict[str, Any] | None:
        if not delta:
            return None
        return self._append_reasoning(str(delta))

    def _append_reasoning(self, delta: str) -> dict[str, Any] | None:
        self._close_assistant_phase()
        event = None
        if self._reasoning_id is None:
            event = append_event(self.record, {"type": "reasoning", "mode": self.mode, "text": ""})
            self._reasoning_id = int(event["id"])
            self._reasoning_text = ""
        self._reasoning_text += str(delta)
        update_event_text(self.record, self._reasoning_id, self._reasoning_text)
        self.save()
        return event

    def tool_call(self, name: str, args: Any, call_id: str = "") -> dict[str, Any]:
        if name in CONTROL_TOOLS:
            return {}
        self.finish_main()
        event = {"type": "tool_call", "mode": self.mode, "name": name, "args": json_value(args)}
        if call_id:
            event["call_id"] = call_id
        stored = append_event(self.record, event)
        self.save()
        return stored

    def tool_result(self, name: str, output: Any, call_id: str = "") -> dict[str, Any]:
        if name in CONTROL_TOOLS:
            return {}
        self.finish_main()
        event = {"type": "tool_result", "mode": self.mode, "name": name, "output": str(output)}
        if call_id:
            event["call_id"] = call_id
        stored = append_event(self.record, event)
        self.save()
        return stored

    def completed_tool_result(self, name: str, output: Any, call_id: str = "") -> dict[str, Any]:
        """Persist a live completion beside its call without closing model output."""
        if name in CONTROL_TOOLS:
            return {}
        event = {"type": "tool_result", "mode": self.mode, "name": name, "output": str(output)}
        if call_id:
            event["call_id"] = call_id
        stored = append_event(self.record, event)
        self._move_result_after_call(stored, name, call_id)
        self.save()
        return stored

    def recovered_tool_result(self, name: str, output: Any, call_id: str = "") -> dict[str, Any]:
        """Persist a late-discovered tool result before the last assistant reply."""
        if name in CONTROL_TOOLS:
            return {}
        event = {"type": "tool_result", "mode": self.mode, "name": name, "output": str(output)}
        if call_id:
            event["call_id"] = call_id
        stored = append_event(self.record, event)
        self._move_event_before(stored, self._last_assistant_id)
        self.save()
        return stored

    def delegation_started(self, calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        calls = self._new_delegation_calls(calls)
        if not calls:
            return None
        self.finish_main()
        stored = append_event(self.record, {"type": "delegation", "mode": self.mode, "calls": json_value(calls)})
        self.save()
        return stored

    def _new_delegation_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only task calls that have not already been recorded."""
        new_calls = []
        for call in calls:
            key = delegation_key(call)
            if key in self._delegation_keys:
                continue
            self._delegation_keys.add(key)
            new_calls.append(call)
        return new_calls

    def subagent_started(self, name: str, task_input: str = "", *, origin: str = "") -> dict[str, Any]:
        self.finish_main()
        self._running_subagents[name] = task_input
        self._running_subagent_origins[name] = origin
        payload = {"type": "subagent", "mode": self.mode, "name": name, "status": "RUNNING", "task_input": task_input}
        if origin:
            payload["origin"] = origin
        event = append_event(self.record, payload)
        self._running_subagent_event_ids[name] = int(event["id"])
        self.save()
        return event

    def subagent_request_updated(self, name: str, task_input: str) -> None:
        if not task_input:
            return
        self._running_subagents[name] = task_input
        self._running_subagent_origins[name] = ""
        event_id = self._running_subagent_event_ids.get(name)
        if event_id is not None:
            update_event_field(self.record, event_id, "task_input", task_input)
            update_event_field(self.record, event_id, "origin", "")
            self.save()

    def subagent_finished(self, name: str, output: str = "") -> dict[str, Any]:
        task_input = self._running_subagents.pop(name, "")
        origin = self._running_subagent_origins.pop(name, "")
        self._running_subagent_event_ids.pop(name, None)
        payload = {
            "type": "subagent",
            "mode": self.mode,
            "name": name,
            "status": "DONE",
            "task_input": task_input,
            "output": output,
        }
        if origin:
            payload["origin"] = origin
        stored = append_event(self.record, payload)
        self.save()
        return stored

    def subagent_cancelled(self, name: str, output: str = "") -> dict[str, Any]:
        task_input = self._running_subagents.pop(name, "")
        origin = self._running_subagent_origins.pop(name, "")
        self._running_subagent_event_ids.pop(name, None)
        payload = {
            "type": "subagent",
            "mode": self.mode,
            "name": name,
            "status": "CANCELLED",
            "task_input": task_input,
            "output": output,
        }
        if origin:
            payload["origin"] = origin
        stored = append_event(self.record, payload)
        self.save()
        return stored

    def subagents_cancelled(self) -> None:
        for name in list(self._running_subagents):
            self.subagent_cancelled(name)

    def system_error(self, text: str) -> dict[str, Any]:
        self.finish_main()
        stored = append_event(self.record, {"type": "system_error", "mode": self.mode, "text": text})
        self.save()
        return stored

    def info(self, text: str) -> dict[str, Any]:
        self.finish_main()
        stored = append_event(self.record, {"type": "info", "mode": self.mode, "text": text})
        self.save()
        return stored

    def interrupted(self, text: str) -> dict[str, Any]:
        self.finish_main()
        stored = append_event(self.record, {"type": "interrupted", "mode": self.mode, "text": text})
        self.save()
        return stored

    def rubric_evaluation_finished(
        self,
        evaluation: dict[str, Any],
        max_iterations: int,
    ) -> dict[str, Any]:
        """Persist one completed rubric evaluation as a non-tool event."""
        stored = append_event(
            self.record,
            {
                "type": "rubric",
                "mode": self.mode,
                "evaluation": json_value(evaluation),
                "max_iterations": int(max_iterations),
            },
        )
        key = (str(evaluation.get("grading_run_id") or ""), int(evaluation.get("iteration") or 0) + 1)
        self._rubric_event_ids[key] = int(stored["id"])
        self.save()
        return stored

    def rubric_evaluation_status(self, run_id: str, pass_number: int, status: str) -> None:
        """Update the persisted evaluation after checkpoint reconciliation."""
        event_id = self._rubric_event_ids.get((run_id, pass_number))
        if event_id is None:
            return
        for event in self.record.get("events", []):
            if isinstance(event, dict) and int(event.get("id") or 0) == event_id:
                evaluation = event.get("evaluation")
                if isinstance(evaluation, dict):
                    evaluation["result"] = status
                self.save()
                return

    def ensure_assistant(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._last_assistant_id is not None and len(text) > len(self._assistant_text.strip()):
            update_event_text(self.record, self._last_assistant_id, text)
            self.save()
            return
        if not self._assistant_seen:
            append_event(self.record, {"type": "assistant", "mode": self.mode, "text": text})
            self.save()

    def finish_main(self) -> None:
        self._close_reasoning_phase()
        self._close_assistant_phase()

    def _close_reasoning_phase(self) -> None:
        self._reasoning_id = None
        self._reasoning_text = ""

    def _close_assistant_phase(self) -> None:
        self._assistant_id = None
        self._assistant_text = ""

    def discard_last_assistant(self) -> None:
        """Remove the currently streamed assistant answer after a cutoff retry."""
        event_id = self._assistant_id or self._last_assistant_id
        if event_id is not None:
            self.record["events"] = [
                event
                for event in self.record.get("events", [])
                if not (isinstance(event, dict) and int(event.get("id") or 0) == event_id)
            ]
            self.save()
        self._assistant_id = None
        self._last_assistant_id = None
        self._assistant_text = ""
        self._assistant_seen = False

    def _move_event_before(self, event: dict[str, Any], before_id: int | None) -> None:
        if before_id is None:
            return
        events = self.record.get("events", [])
        if not isinstance(events, list):
            return
        try:
            events.remove(event)
        except ValueError:
            return
        for index, item in enumerate(events):
            if isinstance(item, dict) and int(item.get("id") or 0) == before_id:
                events.insert(index, event)
                return
        events.append(event)

    def _move_result_after_call(self, event: dict[str, Any], name: str, call_id: str) -> None:
        """Place one result directly after its stable call, with name-order fallback."""
        events = self.record.get("events", [])
        if not isinstance(events, list):
            return
        try:
            events.remove(event)
        except ValueError:
            return

        target_index = None
        if call_id:
            for index, item in enumerate(events):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "tool_call"
                    and str(item.get("call_id") or "") == call_id
                ):
                    target_index = index
                    break
        else:
            calls = [
                index
                for index, item in enumerate(events)
                if isinstance(item, dict)
                and item.get("type") == "tool_call"
                and str(item.get("name") or "") == name
            ]
            completed = sum(
                1
                for item in events
                if isinstance(item, dict)
                and item.get("type") == "tool_result"
                and str(item.get("name") or "") == name
                and not item.get("call_id")
            )
            if completed < len(calls):
                target_index = calls[completed]

        if target_index is None:
            events.append(event)
        else:
            events.insert(target_index + 1, event)

    def discard_reasoning(self) -> None:
        """Remove the currently streamed reasoning block."""
        self._delete_reasoning_event()

    def _delete_reasoning_event(self) -> None:
        if self._reasoning_id is not None:
            event_id = self._reasoning_id
            self.record["events"] = [
                event
                for event in self.record.get("events", [])
                if not (isinstance(event, dict) and int(event.get("id") or 0) == event_id)
            ]
            self.save()
        self._reasoning_id = None
        self._reasoning_text = ""

    async def sync_compaction(self, agent: Any, thread_id: str) -> None:
        if await sync_deepagents_compaction(self.record, agent, thread_id):
            self.save()


class RecordingRenderer:
    """Forward renderer calls while recording the same visible transcript."""

    def __init__(self, renderer: Any, recorder: SessionRecorder) -> None:
        self.renderer = renderer
        self.recorder = recorder
        self._context_notice_rendered = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def reasoning_delta(self, delta: str) -> None:
        event = self.recorder.reasoning_delta(delta)
        call_renderer(self.renderer.reasoning_delta, delta, created_at=event_created_at(event))

    def discard_reasoning(self) -> None:
        callback = getattr(self.renderer, "discard_reasoning", None)
        if callable(callback):
            callback()
        self.recorder.discard_reasoning()

    def text_delta(self, delta: str) -> None:
        event = self.recorder.text_delta(delta)
        call_renderer(self.renderer.text_delta, delta, created_at=event_created_at(event))

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        if name in CONTROL_TOOLS:
            return
        event = self.recorder.tool_call(name, args, call_id=call_id)
        call_renderer(self.renderer.tool_call, name, args, call_id=call_id, created_at=event_created_at(event))

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        if name in CONTROL_TOOLS:
            return
        event = self.recorder.tool_result(name, result, call_id=call_id)
        call_renderer(self.renderer.tool_result, name, result, call_id=call_id, created_at=event_created_at(event))

    def completed_tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Record and update a completed tool without ending active model output."""
        if name in CONTROL_TOOLS:
            return
        event = self.recorder.completed_tool_result(name, result, call_id=call_id)
        callback = getattr(self.renderer, "completed_tool_result", None)
        if callable(callback):
            call_renderer(callback, name, result, call_id=call_id, created_at=event_created_at(event))
        else:
            call_renderer(self.renderer.tool_result, name, result, call_id=call_id, created_at=event_created_at(event))

    def recovered_tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Render and record a late-discovered tool result."""
        if name in CONTROL_TOOLS:
            return
        callback = getattr(self.renderer, "recovered_tool_result", None)
        event = self.recorder.recovered_tool_result(name, result, call_id=call_id)
        if callable(callback):
            call_renderer(callback, name, result, call_id=call_id, created_at=event_created_at(event))
        else:
            call_renderer(self.renderer.tool_result, name, result, call_id=call_id, created_at=event_created_at(event))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        event = self.recorder.delegation_started(calls)
        if event is not None:
            call_renderer(self.renderer.delegation_started, calls, created_at=event_created_at(event))

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        callback = getattr(self.renderer, "delegation_delta", None)
        if callable(callback):
            callback(calls)
            return
        activity = getattr(self.renderer, "model_activity", None)
        if callable(activity):
            activity()

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        callback = getattr(self.renderer, "tool_call_delta", None)
        if callable(callback):
            callback(name, args, call_id=call_id)
            return
        activity = getattr(self.renderer, "model_activity", None)
        if callable(activity):
            activity()

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
    ) -> None:
        event = self.recorder.subagent_started(subagent, task_input, origin=origin)
        call_renderer(
            self.renderer.subagent_started,
            subagent,
            task_input,
            origin=origin,
            eval_id=eval_id,
            row_id=row_id,
            model=model,
            created_at=event_created_at(event),
        )

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        callback = getattr(self.renderer, "subagent_request_updated", None)
        if callable(callback):
            callback(subagent, task_input)
        self.recorder.subagent_request_updated(subagent, task_input)

    def subagent_finished(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        event = self.recorder.subagent_finished(subagent, result)
        call_renderer(
            self.renderer.subagent_finished,
            subagent,
            result,
            eval_id=eval_id,
            row_id=row_id,
            duration_ms=duration_ms,
            created_at=event_created_at(event),
        )

    def subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        callback = getattr(self.renderer, "subagent_cancelled", None)
        event = self.recorder.subagent_cancelled(subagent, result)
        if callable(callback):
            call_renderer(
                callback,
                subagent,
                result,
                eval_id=eval_id,
                row_id=row_id,
                duration_ms=duration_ms,
                created_at=event_created_at(event),
            )

    def eval_subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        label: str = "",
    ) -> None:
        """Forward eval-internal subagent telemetry without recording it."""
        callback = getattr(self.renderer, "eval_subagent_started", None)
        if callable(callback):
            call_renderer(callback, subagent, task_input, eval_id=eval_id, row_id=row_id, model=model, label=label)

    def eval_subagent_finished(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Forward eval-internal subagent completion without recording it."""
        callback = getattr(self.renderer, "eval_subagent_finished", None)
        if callable(callback):
            callback(subagent, result, eval_id=eval_id, row_id=row_id, duration_ms=duration_ms)

    def eval_subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Forward eval-internal subagent failure without recording it."""
        callback = getattr(self.renderer, "eval_subagent_cancelled", None)
        if callable(callback):
            callback(subagent, result, eval_id=eval_id, row_id=row_id, duration_ms=duration_ms)

    def rubric_evaluation_started(self, run_id: str, pass_number: int, max_iterations: int) -> None:
        """Forward transient rubric activity without persisting a start event."""
        self.recorder.finish_main()
        callback = getattr(self.renderer, "rubric_evaluation_started", None)
        if callable(callback):
            callback(run_id, pass_number, max_iterations)

    def rubric_evaluation_finished(self, evaluation: dict[str, Any], max_iterations: int) -> None:
        """Persist and forward one completed rubric evaluation."""
        event = self.recorder.rubric_evaluation_finished(evaluation, max_iterations)
        callback = getattr(self.renderer, "rubric_evaluation_finished", None)
        if callable(callback):
            call_renderer(
                callback,
                evaluation,
                max_iterations,
                created_at=event_created_at(event),
            )

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:
        """Reconcile durable and visible rubric terminal state."""
        self.recorder.rubric_evaluation_status(run_id, pass_number, status)
        callback = getattr(self.renderer, "rubric_evaluation_status", None)
        if callable(callback):
            callback(run_id, pass_number, status, max_iterations)

    def subagents_cancelled(self) -> None:
        callback = getattr(self.renderer, "subagents_cancelled", None)
        if callable(callback):
            callback()
        self.recorder.subagents_cancelled()

    def system_message(self, text: str, *, kind: str = "system") -> None:
        callback = getattr(self.renderer, "system_message", None)
        if callable(callback):
            if kind == "info":
                event = self.recorder.info(text)
                call_renderer(callback, text, kind=kind, created_at=event_created_at(event))
            else:
                callback(text, kind=kind)
        elif hasattr(self.renderer, "console"):
            self.renderer.console.print(text)
            if kind == "info":
                self.recorder.info(text)

    def compaction_started(self) -> None:
        notice = pop_context_overflow_notice()
        self._render_context_notice(notice)
        self.renderer.compaction_started()

    def compaction_finished(self) -> None:
        self.renderer.compaction_finished()

    def finish_main(self) -> None:
        self.renderer.finish_main()
        self.recorder.finish_main()

    def discard_last_assistant(self) -> None:
        callback = getattr(self.renderer, "discard_last_assistant", None)
        if callable(callback):
            callback()
        self.recorder.discard_last_assistant()

    def context_notice_rendered(self) -> bool:
        """Return whether this turn already rendered a context-pressure notice."""
        return self._context_notice_rendered

    def mark_context_notice_rendered(self) -> None:
        """Remember that this turn already rendered a context-pressure notice."""
        self._context_notice_rendered = True

    def _render_context_notice(self, notice: str) -> bool:
        """Render one context-pressure notice at most once per turn."""
        if not notice or self._context_notice_rendered:
            return False
        self._context_notice_rendered = True
        self.system_message(notice, kind="info")
        return True


async def poll_compactions(recorder: SessionRecorder, agent: Any, thread_id: str) -> None:
    """Copy DeepAgents compactions into the session while a turn is running."""
    try:
        while True:
            await asyncio.sleep(COMPACTION_POLL_SECONDS)
            await recorder.sync_compaction(agent, thread_id)
    except asyncio.CancelledError:
        raise


def json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [json_value(item) for item in value]
        return str(value)


def update_event_field(record: dict[str, Any], event_id: int, key: str, value: Any) -> None:
    for event in record.get("events", []):
        if isinstance(event, dict) and int(event.get("id") or 0) == event_id:
            event[key] = value
            return


def update_plan_event_status(record: dict[str, Any], plan_id: str, status: str) -> None:
    """Update the persisted status for a visible plan event."""
    for event in record.get("events", []):
        if not isinstance(event, dict) or event.get("type") != "plan":
            continue
        plan = event.get("plan")
        if isinstance(plan, dict) and str(plan.get("id") or "") == plan_id:
            event["status"] = status


def update_proposal_event_status(record: dict[str, Any], proposal_id: str, status: str) -> None:
    """Update the persisted status for an explicit goal proposal."""
    for event in record.get("events", []):
        if not isinstance(event, dict) or event.get("type") != "proposal":
            continue
        value = event.get("proposal")
        if isinstance(value, dict) and str(value.get("id") or "") == proposal_id:
            event["status"] = status
            return


def event_created_at(event: dict[str, Any] | None) -> str:
    if not isinstance(event, dict):
        return ""
    return str(event.get("created_at") or "")


def call_renderer(callback: Any, *args: Any, created_at: str = "", **kwargs: Any) -> Any:
    if created_at and accepts_created_at(callback):
        kwargs["created_at"] = created_at
    kwargs = accepted_kwargs(callback, kwargs)
    return callback(*args, **kwargs)


def accepts_created_at(callback: Any) -> bool:
    try:
        parameters = signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "created_at" for parameter in parameters
    )


def accepted_kwargs(callback: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop optional renderer kwargs that a concrete renderer does not support."""
    if not kwargs:
        return {}
    try:
        parameters = signature(callback).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    names = set()
    for parameter in parameters:
        if parameter.kind == Parameter.VAR_KEYWORD:
            return kwargs
        if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}:
            names.add(parameter.name)
    return {key: value for key, value in kwargs.items() if key in names}


def delegation_key(call: Any) -> tuple[str, str]:
    """Return a stable key for deduplicating task delegation events."""
    if isinstance(call, dict):
        call_id = str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or "")
        name = str(call.get("name") or call.get("tool_name") or "task")
        args = call.get("args", call.get("input", {}))
    else:
        call_id = str(getattr(call, "id", "") or getattr(call, "call_id", "") or getattr(call, "tool_call_id", "") or "")
        name = str(getattr(call, "name", "") or getattr(call, "tool_name", "") or "task")
        args = getattr(call, "args", None)
        if args is None:
            args = getattr(call, "input", {})

    if call_id:
        return ("id", call_id)

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, json.JSONDecodeError):
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"raw": str(args)}

    description = str(args.get("description") or "")
    subagent_type = str(args.get("subagent_type") or "")
    return ("request", json.dumps([name, description, subagent_type], sort_keys=True))
