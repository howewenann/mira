"""Incremental session persistence for visible runtime events."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent.context_overflow import pop_context_overflow_notice
from runtime.compaction_filter import (
    is_compaction_reasoning,
    is_compaction_reasoning_fragment,
    is_compaction_tail_fragment,
    should_flush_reasoning_probe,
)
from runtime.output_events import normalize_response_delta
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
        self._reasoning_pending = ""
        self._running_subagents: dict[str, str] = {}
        self._running_subagent_event_ids: dict[str, int] = {}
        self._delegation_keys: set[tuple[str, str]] = set()

    def save(self) -> None:
        self.store.save(self.record)

    def user_message(self, text: str) -> None:
        append_event(self.record, {"type": "user", "mode": self.mode, "text": text})

    def text_delta(self, delta: str) -> None:
        delta = normalize_response_delta(self._assistant_text, delta)
        if not delta:
            return
        if self._assistant_id is None:
            event = append_event(self.record, {"type": "assistant", "mode": self.mode, "text": ""})
            self._assistant_id = int(event["id"])
            self._last_assistant_id = self._assistant_id
            self._assistant_text = ""
            self._assistant_seen = True
        self._assistant_text += str(delta)
        update_event_text(self.record, self._assistant_id, self._assistant_text)
        self.save()

    def reasoning_delta(self, delta: str) -> None:
        if not delta:
            return
        self._reasoning_pending += str(delta)
        if is_compaction_tail_fragment(self._reasoning_pending):
            self._reasoning_pending = ""
            self._delete_reasoning_event()
            return
        if is_compaction_reasoning(self._reasoning_pending):
            self._reasoning_pending = ""
            self._delete_reasoning_event()
            return
        if not should_flush_reasoning_probe(self._reasoning_pending):
            return
        delta = self._reasoning_pending
        self._reasoning_pending = ""
        self._append_reasoning(delta)

    def _append_reasoning(self, delta: str) -> None:
        if self._reasoning_id is None:
            event = append_event(self.record, {"type": "reasoning", "mode": self.mode, "text": ""})
            self._reasoning_id = int(event["id"])
            self._reasoning_text = ""
        self._reasoning_text += str(delta)
        update_event_text(self.record, self._reasoning_id, self._reasoning_text)
        self.save()

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self.finish_main()
        event = {"type": "tool_call", "mode": self.mode, "name": name, "args": json_value(args)}
        if call_id:
            event["call_id"] = call_id
        append_event(self.record, event)
        self.save()

    def tool_result(self, name: str, output: Any, call_id: str = "") -> None:
        self.finish_main()
        event = {"type": "tool_result", "mode": self.mode, "name": name, "output": str(output)}
        if call_id:
            event["call_id"] = call_id
        append_event(self.record, event)
        self.save()

    def recovered_tool_result(self, name: str, output: Any, call_id: str = "") -> None:
        """Persist a late-discovered tool result before the last assistant reply."""
        event = {"type": "tool_result", "mode": self.mode, "name": name, "output": str(output)}
        if call_id:
            event["call_id"] = call_id
        stored = append_event(self.record, event)
        self._move_event_before(stored, self._last_assistant_id)
        self.save()

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        calls = self._new_delegation_calls(calls)
        if not calls:
            return
        self.finish_main()
        append_event(self.record, {"type": "delegation", "mode": self.mode, "calls": json_value(calls)})
        self.save()

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

    def subagent_started(self, name: str, task_input: str = "") -> None:
        self.finish_main()
        self._running_subagents[name] = task_input
        event = append_event(
            self.record,
            {"type": "subagent", "mode": self.mode, "name": name, "status": "RUNNING", "task_input": task_input},
        )
        self._running_subagent_event_ids[name] = int(event["id"])
        self.save()

    def subagent_request_updated(self, name: str, task_input: str) -> None:
        if not task_input:
            return
        self._running_subagents[name] = task_input
        event_id = self._running_subagent_event_ids.get(name)
        if event_id is not None:
            update_event_field(self.record, event_id, "task_input", task_input)
            self.save()

    def subagent_finished(self, name: str, output: str = "") -> None:
        task_input = self._running_subagents.pop(name, "")
        self._running_subagent_event_ids.pop(name, None)
        append_event(
            self.record,
            {
                "type": "subagent",
                "mode": self.mode,
                "name": name,
                "status": "DONE",
                "task_input": task_input,
                "output": output,
            },
        )
        self.save()

    def subagent_cancelled(self, name: str, output: str = "") -> None:
        task_input = self._running_subagents.pop(name, "")
        self._running_subagent_event_ids.pop(name, None)
        append_event(
            self.record,
            {
                "type": "subagent",
                "mode": self.mode,
                "name": name,
                "status": "CANCELLED",
                "task_input": task_input,
                "output": output,
            },
        )
        self.save()

    def subagents_cancelled(self) -> None:
        for name in list(self._running_subagents):
            self.subagent_cancelled(name)

    def system_error(self, text: str) -> None:
        append_event(self.record, {"type": "system_error", "mode": self.mode, "text": text})
        self.save()

    def info(self, text: str) -> None:
        if is_compaction_notice(text):
            return
        append_event(self.record, {"type": "info", "mode": self.mode, "text": text})
        self.save()

    def interrupted(self, text: str) -> None:
        append_event(self.record, {"type": "interrupted", "mode": self.mode, "text": text})
        self.save()

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
        if self._reasoning_pending:
            if is_compaction_reasoning_fragment(self._reasoning_pending) or is_compaction_tail_fragment(
                self._reasoning_pending
            ):
                self._delete_reasoning_event()
            else:
                self._append_reasoning(self._reasoning_pending)
            self._reasoning_pending = ""
        self._assistant_id = None
        self._reasoning_id = None
        self._reasoning_text = ""

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

    def discard_reasoning(self) -> None:
        """Remove the currently streamed reasoning block after late compaction detection."""
        self._reasoning_pending = ""
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
        self.renderer.reasoning_delta(delta)
        self.recorder.reasoning_delta(delta)

    def discard_reasoning(self) -> None:
        callback = getattr(self.renderer, "discard_reasoning", None)
        if callable(callback):
            callback()
        self.recorder.discard_reasoning()

    def text_delta(self, delta: str) -> None:
        self.renderer.text_delta(delta)
        self.recorder.text_delta(delta)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self.renderer.tool_call(name, args, call_id=call_id)
        self.recorder.tool_call(name, args, call_id=call_id)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        self.renderer.tool_result(name, result, call_id=call_id)
        self.recorder.tool_result(name, result, call_id=call_id)

    def recovered_tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Render and record a late-discovered tool result."""
        callback = getattr(self.renderer, "recovered_tool_result", None)
        if callable(callback):
            callback(name, result, call_id=call_id)
        else:
            self.renderer.tool_result(name, result, call_id=call_id)
        self.recorder.recovered_tool_result(name, result, call_id=call_id)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self.renderer.delegation_started(calls)
        self.recorder.delegation_started(calls)

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

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        self.renderer.subagent_started(subagent, task_input)
        self.recorder.subagent_started(subagent, task_input)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        callback = getattr(self.renderer, "subagent_request_updated", None)
        if callable(callback):
            callback(subagent, task_input)
        self.recorder.subagent_request_updated(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        self.renderer.subagent_finished(subagent, result)
        self.recorder.subagent_finished(subagent, result)

    def subagent_cancelled(self, subagent: str, result: str = "") -> None:
        callback = getattr(self.renderer, "subagent_cancelled", None)
        if callable(callback):
            callback(subagent, result)
        self.recorder.subagent_cancelled(subagent, result)

    def subagents_cancelled(self) -> None:
        callback = getattr(self.renderer, "subagents_cancelled", None)
        if callable(callback):
            callback()
        self.recorder.subagents_cancelled()

    def system_message(self, text: str, *, kind: str = "system") -> None:
        callback = getattr(self.renderer, "system_message", None)
        if callable(callback):
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
        if not notice or self._context_notice_rendered or is_compaction_notice(notice):
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


def is_compaction_notice(text: str) -> bool:
    """Return whether an info notice is really leaked compaction reasoning."""
    return is_compaction_reasoning(text) or is_compaction_reasoning_fragment(text)


def update_event_field(record: dict[str, Any], event_id: int, key: str, value: Any) -> None:
    for event in record.get("events", []):
        if isinstance(event, dict) and int(event.get("id") or 0) == event_id:
            event[key] = value
            return


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
