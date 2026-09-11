"""DeepAgents-specific projection into process-local inspection events."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from core.context.usage import field as value_field
from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.streams.messages import consume_messages
from core.execution.streams.output import (
    is_tool_message,
    message_text,
    normalized_output_tool_call,
    visible_message_text,
)
from core.execution.streams.tool_args import normalized_call
from core.execution.streams.tools import consume_tool_calls


class SubagentInspectionCapture:
    """Renderer-shaped adapter that records one native child stream."""

    def __init__(self, store: LiveInspectionStore | None, inspection_id: str) -> None:
        self.store = store
        self.inspection_id = inspection_id

    def reasoning_delta(self, value: str, **_kwargs: Any) -> None:
        if self.store is not None:
            self.store.append_delta(self.inspection_id, "reasoning", value)

    def text_delta(self, value: str, **_kwargs: Any) -> None:
        if self.store is not None:
            self.store.append_delta(self.inspection_id, "assistant", value)

    def delegation_started(self, calls: list[Any], **_kwargs: Any) -> None:
        for call in calls:
            value = normalized_call(call)
            self.tool_call(
                str(value.get("name") or "task"),
                value.get("args", {}),
                call_id=str(value.get("id") or ""),
            )

    def tool_call(self, name: str, args: Any, call_id: str = "", **_kwargs: Any) -> None:
        if self.store is not None:
            self.store.upsert_tool_call(
                self.inspection_id,
                InspectionEvent("tool_call", name=name, args=args, call_id=call_id),
            )

    def tool_call_delta(self, name: str, args: Any, call_id: str = "", **kwargs: Any) -> None:
        if self.store is None:
            return
        self.tool_call(name, args, call_id=call_id, **kwargs)

    def tool_result(self, name: str, result: str, call_id: str = "", **_kwargs: Any) -> None:
        self._append(
            InspectionEvent("tool_result", text=str(result), name=name, call_id=call_id)
        )

    completed_tool_result = tool_result

    def tool_error(self, name: str, error: str, call_id: str = "", **_kwargs: Any) -> None:
        self._append(
            InspectionEvent("tool_error", text=str(error), name=name, call_id=call_id)
        )

    completed_tool_error = tool_error

    def system_error(self, error: str) -> None:
        self._append(InspectionEvent("error", text=str(error)))

    def ensure_final_response(self, response: str) -> None:
        if self.store is not None:
            self.store.finish(
                self.inspection_id,
                status="DONE",
                final_response=response,
            )

    def fail(self, error: str, *, status: str = "ERROR") -> None:
        if self.store is not None:
            self.store.finish(self.inspection_id, status=status, error=str(error))

    def model_stream_finished(self) -> None:
        """The next event naturally closes the current ChatLog phase."""

    def _append(self, event: InspectionEvent) -> None:
        if self.store is not None:
            self.store.append(self.inspection_id, event)


async def capture_child_streams(subagent: Any, capture: SubagentInspectionCapture) -> None:
    """Observe native scoped projections without owning child lifecycle."""
    consumers = []
    messages = getattr(subagent, "messages", None)
    if messages is not None:
        consumers.append(consume_messages(messages, capture, render_normal_tools=False))
    tool_calls = getattr(subagent, "tool_calls", None)
    if tool_calls is not None:
        consumers.append(consume_tool_calls(tool_calls, capture))
    if consumers:
        await asyncio.gather(*(_isolate_capture(item, capture) for item in consumers))


async def _isolate_capture(awaitable: Any, capture: SubagentInspectionCapture) -> None:
    """Keep Inspector projection failures outside execution cleanup paths."""
    try:
        await awaitable
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # observational capture must never fail the run
        capture.system_error(f"inspection capture failed: {exc}")


class SubagentInspectionCoordinator:
    """Correlate QuickJS lifecycle IDs with restored native child handles."""

    def __init__(self, store: LiveInspectionStore | None) -> None:
        self.store = store
        self._inspection_by_row: dict[str, str] = {}
        self._native_rows_by_eval: dict[str, deque[str]] = defaultdict(deque)
        self._native_rows_by_task: dict[str, deque[str]] = defaultdict(deque)
        self._protocol_rows_by_task: dict[str, deque[str]] = defaultdict(deque)
        self._protocol_row_by_namespace: dict[tuple[str, ...], str] = {}
        self._pending_snapshots: dict[tuple[str, ...], list[Any]] = {}
        self._message_counts: dict[tuple[str, ...], int] = {}

    def eval_started(self, event: dict[str, Any]) -> str:
        row_id = str(event.get("id") or "")
        if not row_id or self.store is None:
            return ""
        inspection_id = self.store.allocate_id(row_id)
        self._inspection_by_row[row_id] = inspection_id
        eval_id = str(event.get("eval_id") or "")
        description = str(event.get("description") or "")
        if eval_id:
            self._native_rows_by_eval[eval_id].append(row_id)
        if description:
            self._native_rows_by_task[description].append(row_id)
            self._protocol_rows_by_task[description].append(row_id)
        self.store.start(
            inspection_id,
            str(event.get("subagent_type") or "subagent"),
            description,
        )
        self._bind_pending_snapshots()
        return inspection_id

    def eval_finished(self, event: dict[str, Any]) -> None:
        row_id = str(event.get("id") or "")
        inspection_id = self._inspection_by_row.get(row_id, "")
        if not inspection_id or self.store is None:
            return
        phase = str(event.get("phase") or "")
        if phase == "error":
            self.store.finish(
                inspection_id,
                status="ERROR",
                error=str(event.get("error") or "subagent failed"),
            )
        else:
            self.store.finish(inspection_id, status="DONE")

    def cancel_running(self) -> None:
        """Finalize only transient inspection state when its parent turn stops."""
        if self.store is None:
            return
        for inspection_id in self._inspection_by_row.values():
            current = self.store.get(inspection_id)
            if current is not None and current.status == "RUNNING":
                self.store.finish(inspection_id, status="CANCELLED")

    def claim_eval_child(self, subagent: Any) -> str:
        """Return an existing Eval inspection ID for one native child handle."""
        task = str(getattr(subagent, "task_input", "") or "")
        rows = self._native_rows_by_task.get(task)
        if rows:
            return self._claim_native(rows.popleft())
        path = getattr(subagent, "path", ())
        if isinstance(path, (list, tuple)):
            for part in path:
                text = str(part)
                if not text.startswith("tools:"):
                    continue
                rows = self._native_rows_by_eval.get(text.split(":", 1)[1])
                if rows:
                    return self._claim_native(rows.popleft())
        return ""

    def handle_protocol_event(self, event: Any) -> None:
        """Capture Eval child snapshots omitted by the high-level child lane."""
        if not isinstance(event, dict) or event.get("method") != "values":
            return
        params = event.get("params")
        if not isinstance(params, dict):
            return
        namespace = tuple(str(part) for part in params.get("namespace") or ())
        values = params.get("data")
        messages = values.get("messages") if isinstance(values, dict) else None
        if not namespace or not isinstance(messages, list) or not messages:
            return
        self._pending_snapshots[namespace] = messages
        if namespace not in self._protocol_row_by_namespace:
            self._bind_protocol_namespace(namespace, messages)
        self._emit_protocol_snapshot(namespace, messages)

    def _claim_native(self, row_id: str) -> str:
        for rows in (
            *self._native_rows_by_eval.values(),
            *self._native_rows_by_task.values(),
        ):
            try:
                rows.remove(row_id)
            except ValueError:
                pass
        return self._inspection_by_row.get(row_id, "")

    def _bind_pending_snapshots(self) -> None:
        for namespace, messages in self._pending_snapshots.items():
            if namespace not in self._protocol_row_by_namespace:
                self._bind_protocol_namespace(namespace, messages)
            self._emit_protocol_snapshot(namespace, messages)

    def _bind_protocol_namespace(
        self,
        namespace: tuple[str, ...],
        messages: list[Any],
    ) -> None:
        task_input = _first_user_text(messages)
        if not task_input:
            return
        for description, rows in self._protocol_rows_by_task.items():
            if rows and _same_task_input(description, task_input):
                row_id = rows.popleft()
                self._protocol_row_by_namespace[namespace] = row_id
                return

    def _emit_protocol_snapshot(
        self,
        namespace: tuple[str, ...],
        messages: list[Any],
    ) -> None:
        row_id = self._protocol_row_by_namespace.get(namespace, "")
        inspection_id = self._inspection_by_row.get(row_id, "")
        if not inspection_id or self.store is None:
            return
        start = self._message_counts.get(namespace, 0)
        for message in messages[start:]:
            for event in _inspection_events_from_message(message):
                self.store.append(inspection_id, event)
        self._message_counts[namespace] = len(messages)


def _first_user_text(messages: list[Any]) -> str:
    for message in messages:
        if str(value_field(message, "type") or "") in {"human", "user"}:
            return message_text(message)
    return ""


def _same_task_input(description: str, task_input: str) -> bool:
    return description == task_input or (
        len(description) == 200 and task_input.startswith(description)
    )


def _inspection_events_from_message(message: Any) -> list[InspectionEvent]:
    if is_tool_message(message):
        status = str(value_field(message, "status") or "")
        kind = "tool_error" if status == "error" else "tool_result"
        return [
            InspectionEvent(
                kind,
                text=message_text(message),
                name=str(value_field(message, "name") or "tool"),
                call_id=str(
                    value_field(message, "tool_call_id")
                    or value_field(message, "id")
                    or ""
                ),
            )
        ]
    if str(value_field(message, "type") or "") not in {"ai", "assistant"}:
        return []

    events: list[InspectionEvent] = []
    reasoning = _message_reasoning(message)
    if reasoning:
        events.append(InspectionEvent("reasoning", text=reasoning))
    text = visible_message_text(message)
    if text:
        events.append(InspectionEvent("assistant", text=text))
    for value in value_field(message, "tool_calls") or []:
        call = normalized_call(normalized_output_tool_call(value))
        events.append(
            InspectionEvent(
                "tool_call",
                name=str(call.get("name") or "tool"),
                args=call.get("args", {}),
                call_id=str(call.get("id") or ""),
            )
        )
    return events


def _message_reasoning(message: Any) -> str:
    values: list[str] = []
    blocks = value_field(message, "content_blocks")
    if callable(blocks):
        blocks = blocks()
    if not isinstance(blocks, (list, tuple)):
        blocks = value_field(message, "content")
    if isinstance(blocks, (list, tuple)):
        for block in blocks:
            if isinstance(block, dict) and str(block.get("type") or "") == "reasoning":
                text = str(block.get("reasoning") or block.get("text") or "")
                if text:
                    values.append(text)
    additional = value_field(message, "additional_kwargs")
    if isinstance(additional, dict):
        text = str(additional.get("reasoning_content") or additional.get("reasoning") or "")
        if text and text not in values:
            values.append(text)
    return "".join(values)


def live_inspection_store(renderer: Any) -> LiveInspectionStore | None:
    """Find the Textual-owned store through existing in-process wrappers."""
    pending = [renderer]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        store = getattr(current, "live_inspections", None)
        if isinstance(store, LiveInspectionStore):
            return store
        for attribute in ("renderer", "frontend"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)
    return None


def native_inspection_hint(subagent: Any) -> str:
    """Prefer LangGraph's triggering tool call, then its scoped path."""
    trigger = str(getattr(subagent, "trigger_call_id", "") or "")
    if trigger:
        return trigger
    path = getattr(subagent, "path", ())
    if isinstance(path, (list, tuple)) and path:
        return "/".join(str(item) for item in path)
    return ""


__all__ = [
    "SubagentInspectionCapture",
    "SubagentInspectionCoordinator",
    "capture_child_streams",
    "live_inspection_store",
    "native_inspection_hint",
]
