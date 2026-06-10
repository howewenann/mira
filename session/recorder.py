"""Incremental session persistence for visible runtime events."""

from __future__ import annotations

import asyncio
import json
from typing import Any

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

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self.finish_main()
        append_event(self.record, {"type": "delegation", "mode": self.mode, "calls": json_value(calls)})
        self.save()

    def subagent_started(self, name: str, task_input: str = "") -> None:
        self.finish_main()
        append_event(
            self.record,
            {"type": "subagent", "mode": self.mode, "name": name, "status": "RUNNING", "task_input": task_input},
        )
        self.save()

    def subagent_finished(self, name: str, output: str = "") -> None:
        append_event(
            self.record,
            {"type": "subagent", "mode": self.mode, "name": name, "status": "DONE", "output": output},
        )
        self.save()

    def system_error(self, text: str) -> None:
        append_event(self.record, {"type": "system_error", "mode": self.mode, "text": text})
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
        self._assistant_id = None
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def reasoning_delta(self, delta: str) -> None:
        self.renderer.reasoning_delta(delta)
        self.recorder.reasoning_delta(delta)

    def text_delta(self, delta: str) -> None:
        self.renderer.text_delta(delta)
        self.recorder.text_delta(delta)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self.renderer.tool_call(name, args, call_id=call_id)
        self.recorder.tool_call(name, args, call_id=call_id)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        self.renderer.tool_result(name, result, call_id=call_id)
        self.recorder.tool_result(name, result, call_id=call_id)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self.renderer.delegation_started(calls)
        self.recorder.delegation_started(calls)

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        self.renderer.subagent_started(subagent, task_input)
        self.recorder.subagent_started(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        self.renderer.subagent_finished(subagent, result)
        self.recorder.subagent_finished(subagent, result)

    def finish_main(self) -> None:
        self.renderer.finish_main()
        self.recorder.finish_main()


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
