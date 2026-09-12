"""DeepAgents-specific projection into process-local inspection events."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from langgraph.stream import ProtocolEvent, StreamTransformer

from agent.middleware.code_interpreter import EVAL_SUBAGENT_ROW_METADATA
from core.context.usage import field as value_field
from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.streams.messages import consume_messages
from core.execution.streams.output import (
    is_tool_message,
    message_text,
    normalize_response_delta,
    normalized_output_tool_call,
    visible_message_text,
)
from core.execution.streams.provider import event_delta
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
        self._eval_task_call_ids: set[str] = set()
        self._native_row_by_trigger: dict[str, str] = {}
        self._protocol_row_by_namespace: dict[tuple[str, ...], str] = {}
        self._pending_snapshots: dict[tuple[str, ...], list[Any]] = {}
        self._pending_message_events: dict[tuple[str, ...], list[Any]] = defaultdict(list)
        self._pending_interrupts: dict[tuple[str, ...], list[Any]] = {}
        self._message_counts: dict[tuple[str, ...], int] = {}
        self._message_reasoning: dict[tuple[Any, Any], str] = {}
        self._message_text: dict[tuple[Any, Any], str] = {}
        self._raw_stream_kinds: set[tuple[str, str]] = set()
        self._row_ready: dict[str, asyncio.Event] = {}
        self._pass_tool_calls: dict[str, dict[str, Any]] = {}
        self._pass_interrupt_calls: list[dict[str, Any]] = []
        self._seen_pass_interrupt_actions: set[tuple[str, str, int]] = set()

    def begin_pass(self) -> None:
        """Forget interrupt candidates from the preceding HITL stream pass."""
        self._pass_tool_calls.clear()
        self._pass_interrupt_calls.clear()
        self._seen_pass_interrupt_actions.clear()

    def is_eval_tool_call(self, call_id: str) -> bool:
        """Return whether a projected call belongs to an Eval child row."""
        return bool(call_id) and (
            call_id in self._eval_task_call_ids
            or call_id in self._inspection_by_row
            or call_id in self._pass_tool_calls
        )

    def eval_interrupt_calls(self) -> list[dict[str, Any]]:
        """Return actions interrupted inside Eval children in this pass."""
        return list(self._pass_interrupt_calls)

    def update_eval_tool_call(
        self,
        row_id: str,
        call_id: str,
        name: str,
        args: Any,
    ) -> None:
        """Reflect an approved edit in the owning Inspector transcript."""
        inspection_id = self._inspection_by_row.get(row_id, "")
        if inspection_id and self.store is not None:
            self.store.upsert_tool_call(
                inspection_id,
                InspectionEvent(
                    "tool_call",
                    name=name,
                    args=args,
                    call_id=call_id,
                ),
            )

    def eval_started(self, event: dict[str, Any]) -> str:
        row_id = str(event.get("id") or "")
        if not row_id or self.store is None:
            return ""
        existing = self._inspection_by_row.get(row_id, "")
        if existing:
            return existing
        inspection_id = self.store.allocate_id(row_id)
        self._inspection_by_row[row_id] = inspection_id
        description = str(event.get("description") or "")
        self.store.start(
            inspection_id,
            str(event.get("subagent_type") or "subagent"),
            description,
        )
        waiter = self._row_ready.get(row_id)
        if waiter is not None:
            waiter.set()
        self._bind_pending_snapshots()
        self._drain_pending_message_events()
        self._drain_pending_interrupts()
        return inspection_id

    def update_title(self, row_id: str, title: str) -> None:
        """Apply the frontend-generated identity to an Eval inspection."""
        inspection_id = self._inspection_by_row.get(str(row_id or ""), "")
        if inspection_id and self.store is not None:
            self.store.start(inspection_id, title)

    def is_eval_child(self, subagent: Any) -> bool:
        """Identify a native child explicitly marked by QuickJS's task bridge."""
        trigger = str(getattr(subagent, "trigger_call_id", "") or "")
        return bool(trigger) and trigger in self._native_row_by_trigger

    async def wait_for_eval_inspection(self, subagent: Any) -> str:
        """Resolve a native QuickJS handle to its custom lifecycle identity."""
        trigger = str(getattr(subagent, "trigger_call_id", "") or "")
        row_id = self._native_row_by_trigger.get(trigger, "")
        inspection_id = self._inspection_by_row.get(row_id, "")
        if inspection_id or not row_id:
            return inspection_id
        waiter = self._row_ready.setdefault(row_id, asyncio.Event())
        try:
            await asyncio.wait_for(waiter.wait(), timeout=1.0)
        except TimeoutError:
            return ""
        return self._inspection_by_row.get(row_id, "")

    def register_eval_namespace(
        self,
        namespace: tuple[str, ...],
        row_id: str,
    ) -> None:
        """Bind LangGraph's native task UUID to its exact Eval lifecycle row."""
        if not namespace or not row_id:
            return
        self._eval_task_call_ids.add(row_id)
        _name, separator, trigger = namespace[-1].partition(":")
        if separator and trigger:
            self._native_row_by_trigger[trigger] = row_id
        self._protocol_row_by_namespace[namespace] = row_id
        self._bind_pending_snapshots()
        self._drain_pending_message_events()
        self._drain_pending_interrupts()

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

    def handle_protocol_event(self, event: Any) -> None:
        """Capture namespaced Eval message deltas and durable snapshots."""
        if not isinstance(event, dict):
            return
        if event.get("method") == "messages":
            self._handle_message_event(event)
            return
        if event.get("method") != "values":
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
        self._record_eval_interrupts(namespace, params.get("interrupts"))

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
        exact_row = self._row_id_from_namespace(namespace)
        if exact_row:
            self._protocol_row_by_namespace[namespace] = exact_row

    def _row_id_from_namespace(self, namespace: tuple[str, ...]) -> str:
        for part in namespace:
            _name, separator, trigger = part.partition(":")
            if separator and trigger in self._inspection_by_row:
                return trigger
        return ""

    def _row_id_for_namespace(self, namespace: tuple[str, ...]) -> str:
        exact = self._protocol_row_by_namespace.get(namespace, "")
        if exact:
            return exact
        row_id = self._row_id_from_namespace(namespace)
        if row_id:
            return row_id
        matches = [
            (len(parent), mapped_row)
            for parent, mapped_row in self._protocol_row_by_namespace.items()
            if len(parent) <= len(namespace) and namespace[: len(parent)] == parent
        ]
        return max(matches, default=(0, ""))[1]

    def _handle_message_event(self, event: dict[str, Any]) -> None:
        params = event.get("params")
        if not isinstance(params, dict):
            return
        namespace = tuple(str(part) for part in params.get("namespace") or ())
        if not namespace:
            return
        row_id = self._row_id_for_namespace(namespace)
        if not row_id:
            self._pending_message_events[namespace].append(params.get("data"))
            return
        self._emit_message_delta(row_id, params.get("data"))

    def _drain_pending_message_events(self) -> None:
        for namespace in list(self._pending_message_events):
            row_id = self._row_id_for_namespace(namespace)
            if not row_id:
                continue
            events = self._pending_message_events.pop(namespace)
            for data in events:
                self._emit_message_delta(row_id, data)

    def _record_eval_interrupts(
        self,
        namespace: tuple[str, ...],
        interrupts: Any,
    ) -> None:
        if not interrupts:
            return
        row_id = self._row_id_for_namespace(namespace)
        if not row_id:
            self._pending_interrupts[namespace] = list(interrupts)
            return
        for interrupt in interrupts:
            value = getattr(interrupt, "value", interrupt)
            actions = value.get("action_requests") if isinstance(value, dict) else None
            if not isinstance(actions, list):
                continue
            interrupt_id = str(getattr(interrupt, "id", "") or id(interrupt))
            occurrences: dict[tuple[str, str], int] = defaultdict(int)
            for index, action in enumerate(actions):
                if not isinstance(action, dict):
                    continue
                action_key = _tool_action_identity(action)
                occurrence = occurrences[action_key]
                occurrences[action_key] += 1
                seen_key = (row_id, interrupt_id, index)
                if seen_key in self._seen_pass_interrupt_actions:
                    continue
                self._seen_pass_interrupt_actions.add(seen_key)
                candidates = [
                    call
                    for call in self._pass_tool_calls.values()
                    if _tool_action_identity(call) == action_key
                ]
                call_id = str(
                    candidates[occurrence].get("call_id")
                    if occurrence < len(candidates)
                    else ""
                )
                self._pass_interrupt_calls.append(
                    {
                        "name": str(action.get("name") or "tool"),
                        "args": action.get("args", {}),
                        "call_id": call_id,
                        "row_id": row_id,
                    }
                )

    def _drain_pending_interrupts(self) -> None:
        for namespace in list(self._pending_interrupts):
            if not self._row_id_for_namespace(namespace):
                continue
            interrupts = self._pending_interrupts.pop(namespace)
            self._record_eval_interrupts(namespace, interrupts)

    def _emit_message_delta(self, row_id: str, data: Any) -> None:
        if self.store is None or not isinstance(data, (list, tuple)) or not data:
            return
        payload = data[0]
        if not isinstance(payload, dict):
            return
        if str(payload.get("event") or "") == "error":
            inspection_id = self._inspection_by_row.get(row_id, "")
            if inspection_id:
                self.store.append(
                    inspection_id,
                    InspectionEvent(
                        "error",
                        text=str(
                            payload.get("message")
                            or payload.get("error")
                            or "subagent model failed"
                        ),
                    ),
                )
            return
        if str(payload.get("event") or "") != "content-block-delta":
            return
        delta = event_delta(payload)
        delta_type = str(delta.get("type") or "")
        kind = "reasoning" if delta_type == "reasoning" else ""
        text = str(delta.get("reasoning") or delta.get("text") or "")
        if delta_type in {"reasoning_delta", "reasoning-delta"}:
            kind = "reasoning"
        elif delta_type in {"text", "text_delta", "text-delta"}:
            kind = "assistant"
        if not kind or not text:
            return
        inspection_id = self._inspection_by_row.get(row_id, "")
        if not inspection_id:
            return
        self._raw_stream_kinds.add((row_id, kind))
        self.store.append_delta(inspection_id, kind, text)

    def _emit_protocol_snapshot(
        self,
        namespace: tuple[str, ...],
        messages: list[Any],
    ) -> None:
        row_id = self._protocol_row_by_namespace.get(namespace, "")
        row_id = row_id or self._row_id_for_namespace(namespace)
        inspection_id = self._inspection_by_row.get(row_id, "")
        if not inspection_id or self.store is None:
            return
        task_input = _first_user_text(messages)
        current = self.store.get(inspection_id)
        current_task = (
            current.events[0].text
            if current is not None and current.events and current.events[0].kind == "user"
            else ""
        )
        if task_input and task_input != current_task:
            self.store.update_task(inspection_id, task_input)
        start = self._message_counts.get(namespace, 0)
        for index, message in enumerate(messages):
            message_type = str(value_field(message, "type") or "")
            if message_type in {"ai", "assistant"}:
                self._emit_protocol_assistant(
                    namespace,
                    index,
                    inspection_id,
                    message,
                    include_tool_calls=index >= start,
                )
            elif index >= start:
                for event in _inspection_events_from_message(message):
                    self.store.append(inspection_id, event)
        self._message_counts[namespace] = len(messages)

    def _emit_protocol_assistant(
        self,
        namespace: tuple[str, ...],
        index: int,
        inspection_id: str,
        message: Any,
        *,
        include_tool_calls: bool,
    ) -> None:
        row_id = self._row_id_for_namespace(namespace)
        message_id = str(value_field(message, "id") or "")
        key = (row_id, message_id) if message_id else (namespace, index)
        if (row_id, "reasoning") not in self._raw_stream_kinds:
            reasoning = _message_reasoning(message)
            reasoning_delta = _snapshot_delta(self._message_reasoning.get(key, ""), reasoning)
            if reasoning_delta:
                self.store.append_delta(inspection_id, "reasoning", reasoning_delta)
            self._message_reasoning[key] = reasoning

        if (row_id, "assistant") not in self._raw_stream_kinds:
            text = visible_message_text(message)
            text_delta = _snapshot_delta(self._message_text.get(key, ""), text)
            if text_delta:
                self.store.append_delta(inspection_id, "assistant", text_delta)
            self._message_text[key] = text

        if include_tool_calls:
            for event in _inspection_events_from_message(message):
                if event.kind == "tool_call":
                    self.store.append(inspection_id, event)
                    if event.call_id:
                        self._pass_tool_calls[event.call_id] = {
                            "name": event.name,
                            "args": event.args,
                            "call_id": event.call_id,
                            "row_id": row_id,
                        }


def _first_user_text(messages: list[Any]) -> str:
    for message in messages:
        if str(value_field(message, "type") or "") in {"human", "user"}:
            return message_text(message)
    return ""


def _tool_action_identity(value: dict[str, Any]) -> tuple[str, str]:
    return (
        str(value.get("name") or "tool"),
        json.dumps(
            value.get("args", {}),
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        ),
    )


def _snapshot_delta(previous: str, current: str) -> str:
    """Return only genuine growth from a cumulative protocol snapshot."""
    if not current or current == previous or not current.startswith(previous):
        return ""
    return normalize_response_delta(previous, current[len(previous) :])


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


class EvalInspectionTransformer(StreamTransformer):
    """Observe Eval row metadata before native task events are folded away."""

    before_builtins = True
    required_stream_modes = ("tasks",)

    def __init__(
        self,
        scope: tuple[str, ...],
        coordinator: SubagentInspectionCoordinator,
    ) -> None:
        super().__init__(scope)
        self.coordinator = coordinator

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        params = event["params"]
        data = params.get("data") or {}
        metadata = data.get("metadata") or {}
        row_id = str(metadata.get(EVAL_SUBAGENT_ROW_METADATA) or "")
        if row_id:
            namespace = tuple(str(part) for part in params.get("namespace") or ())
            self.coordinator.register_eval_namespace(namespace, row_id)
        return True


__all__ = [
    "EvalInspectionTransformer",
    "SubagentInspectionCapture",
    "SubagentInspectionCoordinator",
    "capture_child_streams",
    "live_inspection_store",
    "native_inspection_hint",
]
