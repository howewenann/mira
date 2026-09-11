"""Subagent stream consumption and status projection helpers."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import suppress
from itertools import count
from typing import Any

from agent.rubric.graphs import INTERNAL_RUBRIC_GRAPHS
from core.context.usage import field
from core.execution.streams.output import (
    is_tool_message,
    message_text,
    normalized_output_tool_call,
    visible_message_text,
)
from core.execution.streams.output import call_renderer as call_frontend
from core.execution.streams.messages import consume_messages
from core.execution.streams.provider import event_delta
from core.execution.streams.rubric import RUBRIC_TOOL_END, RUBRIC_TOOL_START, RubricEventRenderer
from core.execution.streams.tool_args import normalized_call
from core.execution.streams.tools import consume_tool_calls, tool_output_text
from session.subagent_runs import now_iso

DYNAMIC_TOOL_SUBAGENT = "dynamic_tool_subagent"
EVAL_SUBAGENT = "eval_subagent"


async def consume_subagents(
    subagents: Any,
    renderer: Any,
    rubric: RubricEventRenderer | None = None,
    protocol_capture: "SubagentProtocolCapture | None" = None,
) -> None:
    """Consume subagent streams while the status animation is active."""
    animation = None
    tasks: list[asyncio.Task[None]] = []
    cancelled = False
    visible_started = False

    try:
        async for subagent in subagents:
            if internal_rubric_subgraph(subagent):
                task = asyncio.create_task(
                    drain_internal_rubric_subgraph(subagent, rubric)
                )
            else:
                eval_task_call_id = (
                    protocol_capture.claim_high_level_subagent(subagent)
                    if protocol_capture is not None
                    else ""
                )
                if eval_task_call_id:
                    task = asyncio.create_task(
                        consume_eval_subagent(subagent, renderer, eval_task_call_id)
                    )
                    tasks.append(task)
                    await asyncio.sleep(0)
                    continue
                if not visible_started:
                    visible_started = True
                    if hasattr(renderer, "start_subagent_live"):
                        renderer.start_subagent_live()
                    if not getattr(renderer, "manages_subagent_animation", False):
                        animation = asyncio.create_task(animate_subagents(renderer))
                task = asyncio.create_task(consume_subagent(subagent, renderer))
            tasks.append(task)
            await asyncio.sleep(0)

        if tasks:
            await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        cancelled = True
        await cancel_subagent_tasks(tasks)
        if visible_started:
            call_renderer(renderer, "subagents_cancelled")
        raise
    except Exception:
        cancelled = True
        await cancel_subagent_tasks(tasks)
        if visible_started:
            call_renderer(renderer, "subagents_cancelled")
        raise
    finally:
        if animation is not None:
            animation.cancel()
            with suppress(asyncio.CancelledError):
                await animation
        if visible_started and not cancelled and hasattr(renderer, "stop_subagent_live"):
            renderer.stop_subagent_live()


def internal_rubric_subgraph(subagent: Any) -> bool:
    """Identify verifier/grader implementation graphs from LangGraph metadata."""
    graph_name = getattr(subagent, "graph_name", None)
    if isinstance(graph_name, str) and graph_name in INTERNAL_RUBRIC_GRAPHS:
        return True
    path = getattr(subagent, "path", ())
    return isinstance(path, (list, tuple)) and any(
        str(part).split(":", 1)[0] in INTERNAL_RUBRIC_GRAPHS for part in path
    )


async def drain_internal_rubric_subgraph(
    subagent: Any,
    rubric: RubricEventRenderer | None,
) -> None:
    """Consume an internal Rubric graph without projecting root transcript UI."""

    async def drain_custom() -> None:
        custom = getattr(subagent, "custom", None)
        if custom is None:
            return
        async for event in custom:
            if (
                rubric is not None
                and isinstance(event, dict)
                and event.get("type") in {RUBRIC_TOOL_START, RUBRIC_TOOL_END}
            ):
                rubric.handle(event)

    await asyncio.gather(drain_custom(), subagent_result(subagent))


async def cancel_subagent_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Cancel all child subagent consumers and wait for them to settle."""
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def call_renderer(renderer: Any, method: str) -> None:
    """Call an optional renderer lifecycle hook."""
    callback = getattr(renderer, method, None)
    if callable(callback):
        callback()


async def animate_subagents(renderer: Any) -> None:
    """Tick the subagent spinner until the parent task cancels it."""
    while True:
        if hasattr(renderer, "tick_subagents"):
            renderer.tick_subagents()
        await asyncio.sleep(0.12)


async def consume_subagent(subagent: Any, renderer: Any) -> None:
    """Render one subagent lifecycle and continuously capture its transcript."""
    name = renderer.subagent_label(subagent)
    task_input = getattr(subagent, "task_input", "")
    origin = subagent_origin(subagent)
    task_call_id = subagent_task_call_id(subagent)
    path = getattr(subagent, "path", ())
    namespace = tuple(str(item) for item in path) if isinstance(path, (list, tuple)) else ()
    metadata = {
        "graph_name": str(getattr(subagent, "graph_name", "") or ""),
    }
    call_frontend(
        renderer,
        "subagent_started",
        name,
        task_input,
        origin=origin,
        namespace=namespace,
        metadata=metadata,
        task_call_id=task_call_id,
        tool_call_id=task_call_id,
    )

    capture = SubagentTranscriptCapture(renderer, task_call_id, stream_path=namespace)
    consumers = []
    messages = getattr(subagent, "messages", None)
    if messages is not None:
        consumers.append(consume_messages(messages, capture, render_normal_tools=False))
    tool_calls = getattr(subagent, "tool_calls", None)
    if tool_calls is not None:
        consumers.append(consume_tool_calls(tool_calls, capture))

    try:
        values = await asyncio.gather(*consumers, subagent_result(subagent))
        result = values[-1]
    except asyncio.CancelledError:
        call_frontend(
            renderer,
            "subagent_cancelled",
            name,
            result="",
            namespace=namespace,
            metadata=metadata,
            task_call_id=task_call_id,
            status="CANCELLED",
        )
        raise
    except Exception as exc:
        result = f"error: {exc}"
        call_frontend(
            renderer,
            "subagent_cancelled",
            name,
            result=result,
            namespace=namespace,
            metadata=metadata,
            task_call_id=task_call_id,
            status="ERROR",
        )
        return

    capture.ensure_final_output(str(result))
    call_frontend(
        renderer,
        "subagent_finished",
        name,
        result=str(result),
        namespace=namespace,
        metadata=metadata,
        task_call_id=task_call_id,
        status="DONE",
    )


async def consume_eval_subagent(
    subagent: Any,
    renderer: Any,
    task_call_id: str,
) -> None:
    """Drain a QuickJS child through its existing eval lifecycle identity."""
    path = getattr(subagent, "path", ())
    namespace = tuple(str(item) for item in path) if isinstance(path, (list, tuple)) else ()
    capture = SubagentTranscriptCapture(renderer, task_call_id, stream_path=namespace)
    consumers = []
    messages = getattr(subagent, "messages", None)
    if messages is not None:
        consumers.append(consume_messages(messages, capture, render_normal_tools=False))
    tool_calls = getattr(subagent, "tool_calls", None)
    if tool_calls is not None:
        consumers.append(consume_tool_calls(tool_calls, capture))
    try:
        values = await asyncio.gather(*consumers, subagent_result(subagent))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        call_frontend(
            renderer,
            "subagent_run_event",
            task_call_id,
            {"type": "system_error", "text": f"error: {exc}"},
            stream_path=namespace,
        )
        return
    capture.ensure_final_output(str(values[-1]))


class SubagentProtocolCapture:
    """Capture eval-created child state snapshots omitted by the high-level lane.

    QuickJS invokes ``task`` outside ToolNode, so LangGraph's high-level
    ``subagents`` projection can collapse parallel siblings. The v3 protocol
    still exposes a distinct namespace and actual message state for every
    child. This narrow adapter consumes those native value snapshots and maps
    them to QuickJS's persisted ``ptc_task_*`` lifecycle identities.
    """

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self._waiting_runs: dict[str, deque[str]] = defaultdict(deque)
        self._waiting_eval_runs: dict[str, deque[str]] = defaultdict(deque)
        self._pending_snapshots: dict[tuple[str, ...], list[Any]] = {}
        self._pending_message_events: dict[tuple[str, ...], list[Any]] = defaultdict(list)
        self._run_by_namespace: dict[tuple[str, ...], str] = {}
        self._message_counts: dict[tuple[str, ...], int] = {}
        self._raw_text: dict[tuple[str, str, str], str] = {}
        self._raw_text_values: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._high_level_namespaces: set[tuple[str, ...]] = set()

    def run_started(self, event: dict[str, Any]) -> None:
        """Register one QuickJS task identity before or after its first snapshot."""
        task_call_id = str(event.get("id") or "")
        description = str(event.get("description") or "")
        if not task_call_id or not description:
            return
        self._waiting_runs[description].append(task_call_id)
        eval_id = str(event.get("eval_id") or "")
        if eval_id:
            self._waiting_eval_runs[eval_id].append(task_call_id)
        self._bind_pending()

    def claim_high_level_subagent(self, subagent: Any) -> str:
        """Bind a restored high-level eval child without creating a second run."""
        task_input = str(getattr(subagent, "task_input", "") or "")
        path = getattr(subagent, "path", ())
        namespace = tuple(str(item) for item in path) if isinstance(path, (list, tuple)) else ()
        if not namespace:
            return ""
        task_call_id = self._run_by_namespace.get(namespace, "")
        if task_call_id:
            self._discard_waiting_id(task_call_id)
            self._high_level_namespaces.add(namespace)
            return task_call_id
        for part in namespace:
            if not part.startswith("tools:"):
                continue
            task_ids = self._waiting_eval_runs.get(part.split(":", 1)[1])
            if task_ids:
                task_call_id = task_ids.popleft()
                self._discard_waiting_id(task_call_id)
                self._run_by_namespace[namespace] = task_call_id
                self._high_level_namespaces.add(namespace)
                return task_call_id
        if not task_input:
            return ""
        for description, task_ids in self._waiting_runs.items():
            if task_ids and _same_task_input(description, task_input):
                task_call_id = task_ids.popleft()
                self._discard_waiting_id(task_call_id)
                self._run_by_namespace[namespace] = task_call_id
                self._high_level_namespaces.add(namespace)
                call_frontend(
                    self.renderer,
                    "subagent_task_input_updated",
                    task_call_id,
                    task_input,
                )
                return task_call_id
        return ""

    def handle(self, event: Any) -> None:
        """Consume namespaced v3 message deltas and durable value snapshots."""
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
        if namespace not in self._run_by_namespace:
            self._bind_namespace(namespace, messages)
        task_call_id = self._run_by_namespace.get(namespace, "")
        task_input = _first_human_text(messages)
        if task_call_id and task_input:
            call_frontend(
                self.renderer,
                "subagent_task_input_updated",
                task_call_id,
                task_input,
            )
        self._emit_snapshot(namespace, messages)

    def _bind_pending(self) -> None:
        for namespace, messages in self._pending_snapshots.items():
            if namespace not in self._run_by_namespace:
                self._bind_namespace(namespace, messages)
            self._emit_snapshot(namespace, messages)

    def _bind_namespace(self, namespace: tuple[str, ...], messages: list[Any]) -> None:
        task_input = _first_human_text(messages)
        if not task_input:
            return
        for description, task_ids in self._waiting_runs.items():
            if task_ids and _same_task_input(description, task_input):
                task_call_id = task_ids.popleft()
                self._discard_waiting_id(task_call_id)
                self._run_by_namespace[namespace] = task_call_id
                call_frontend(
                    self.renderer,
                    "subagent_task_input_updated",
                    task_call_id,
                    task_input,
                )
                self._drain_pending_message_events()
                return

    def _discard_waiting_id(self, task_call_id: str) -> None:
        """Remove one claimed task from every fallback matching queue."""
        for queue in (*self._waiting_runs.values(), *self._waiting_eval_runs.values()):
            try:
                queue.remove(task_call_id)
            except ValueError:
                continue

    def _emit_snapshot(self, namespace: tuple[str, ...], messages: list[Any]) -> None:
        task_call_id = self._run_by_namespace.get(namespace)
        if not task_call_id:
            return
        if self._uses_high_level_lane(namespace):
            self._message_counts[namespace] = len(messages)
            return
        start = self._message_counts.get(namespace, 0)
        if start >= len(messages):
            return
        for message in messages[start:]:
            for event in _run_events_from_message(message):
                if event.get("type") in {"reasoning", "assistant"} and str(
                    event.get("text") or ""
                ) in self._raw_text_values[(task_call_id, str(event.get("type") or ""))]:
                    continue
                call_frontend(
                    self.renderer,
                    "subagent_run_event",
                    task_call_id,
                    event,
                    stream_path=namespace,
                )
        self._message_counts[namespace] = len(messages)

    def _handle_message_event(self, event: dict[str, Any]) -> None:
        params = event.get("params")
        if not isinstance(params, dict):
            return
        namespace = tuple(str(part) for part in params.get("namespace") or ())
        if not namespace:
            return
        if self._uses_high_level_lane(namespace):
            return
        data = params.get("data")
        task_call_id = self._task_call_for_namespace(namespace)
        if not task_call_id:
            self._pending_message_events[namespace].append(data)
            return
        self._emit_message_event(namespace, task_call_id, data)

    def _drain_pending_message_events(self) -> None:
        for namespace in list(self._pending_message_events):
            task_call_id = self._task_call_for_namespace(namespace)
            if not task_call_id:
                continue
            events = self._pending_message_events.pop(namespace)
            for data in events:
                self._emit_message_event(namespace, task_call_id, data)

    def _task_call_for_namespace(self, namespace: tuple[str, ...]) -> str:
        exact = self._run_by_namespace.get(namespace)
        if exact:
            return exact
        matches = [
            (len(child_namespace), task_call_id)
            for child_namespace, task_call_id in self._run_by_namespace.items()
            if len(child_namespace) <= len(namespace)
            and namespace[: len(child_namespace)] == child_namespace
        ]
        return max(matches, default=(0, ""))[1]

    def _uses_high_level_lane(self, namespace: tuple[str, ...]) -> bool:
        return any(
            len(parent) <= len(namespace) and namespace[: len(parent)] == parent
            for parent in self._high_level_namespaces
        )

    def _emit_message_event(
        self,
        namespace: tuple[str, ...],
        task_call_id: str,
        data: Any,
    ) -> None:
        if not isinstance(data, list | tuple) or not data:
            return
        payload = data[0]
        metadata = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
        if not isinstance(payload, dict):
            self._emit_whole_message(namespace, task_call_id, payload)
            return

        protocol_type = str(payload.get("event") or "")
        run_id = str(
            metadata.get("run_id")
            or payload.get("message_id")
            or payload.get("id")
            or "message"
        )
        if protocol_type == "error":
            error = str(
                payload.get("message")
                or payload.get("error")
                or "subagent model failed"
            )
            call_frontend(
                self.renderer,
                "subagent_run_event",
                task_call_id,
                {"type": "system_error", "text": error},
                stream_path=namespace,
            )
            return
        if protocol_type != "content-block-delta":
            return
        delta = event_delta(payload)
        delta_type = str(delta.get("type") or "")
        if delta_type == "reasoning-delta":
            self._emit_raw_text(
                namespace,
                task_call_id,
                run_id,
                "reasoning",
                str(delta.get("reasoning") or delta.get("text") or ""),
            )
        elif delta_type == "text-delta":
            self._emit_raw_text(
                namespace,
                task_call_id,
                run_id,
                "assistant",
                str(delta.get("text") or ""),
            )

    def _emit_whole_message(
        self,
        namespace: tuple[str, ...],
        task_call_id: str,
        message: Any,
    ) -> None:
        message_id = str(field(message, "id") or "message")
        for event in _run_events_from_message(message):
            event_type = str(event.get("type") or "")
            if event_type in {"reasoning", "assistant"}:
                text = str(event.get("text") or "")
                event["stream_id"] = f"protocol-{message_id}-{event_type}"
                self._raw_text_values[(task_call_id, event_type)].add(text)
            call_frontend(
                self.renderer,
                "subagent_run_event",
                task_call_id,
                event,
                stream_path=namespace,
            )

    def _emit_raw_text(
        self,
        namespace: tuple[str, ...],
        task_call_id: str,
        run_id: str,
        event_type: str,
        delta: str,
    ) -> None:
        if not delta:
            return
        key = (task_call_id, run_id, event_type)
        text = self._raw_text.get(key, "") + delta
        self._raw_text[key] = text
        self._raw_text_values[(task_call_id, event_type)].add(text)
        call_frontend(
            self.renderer,
            "subagent_run_event",
            task_call_id,
            {
                "type": event_type,
                "text": text,
                "stream_id": f"protocol-{run_id}-{event_type}",
            },
            stream_path=namespace,
        )


class SubagentTranscriptCapture:
    """Project native child streams into the durable run event schema."""

    def __init__(
        self,
        renderer: Any,
        task_call_id: str,
        *,
        stream_path: tuple[str, ...] = (),
    ) -> None:
        self.renderer = renderer
        self.task_call_id = task_call_id
        self.stream_path = stream_path
        self._stream_sequence = count(1)
        self._active_type = ""
        self._active_stream_id = ""
        self._active_text = ""
        self._assistant_texts: list[str] = []

    def _append(self, event: dict[str, Any]) -> None:
        event.setdefault("created_at", now_iso())
        call_frontend(
            self.renderer,
            "subagent_run_event",
            self.task_call_id,
            event,
            stream_path=self.stream_path,
        )

    def _text(self, event_type: str, delta: str) -> None:
        if not delta:
            return
        if self._active_type != event_type:
            self._active_type = event_type
            self._active_stream_id = f"message-{next(self._stream_sequence)}"
            self._active_text = ""
        self._active_text += str(delta)
        if event_type == "assistant":
            self._assistant_texts.append(str(delta))
        self._append(
            {
                "type": event_type,
                "text": self._active_text,
                "stream_id": self._active_stream_id,
            }
        )

    def reasoning_delta(self, delta: str, **_identity: Any) -> None:
        self._text("reasoning", delta)

    def text_delta(self, delta: str, **_identity: Any) -> None:
        self._text("assistant", delta)

    def model_stream_finished(self) -> None:
        self._active_type = ""
        self._active_stream_id = ""
        self._active_text = ""

    def discard_reasoning(self) -> None:
        self.model_stream_finished()

    def tool_call_delta(self, name: str, args: Any, call_id: str = "", **_identity: Any) -> None:
        self.tool_call(name, args, call_id=call_id)

    def tool_call_updated(self, name: str, args: Any, call_id: str = "", **_identity: Any) -> None:
        self.tool_call(name, args, call_id=call_id)

    def tool_call(self, name: str, args: Any, call_id: str = "", **_identity: Any) -> None:
        self.model_stream_finished()
        self._append({"type": "tool_call", "name": name, "args": args, "call_id": call_id})

    def tool_result(self, name: str, result: Any, call_id: str = "", **identity: Any) -> None:
        self._tool_result(name, result, call_id=call_id, is_error=False, **identity)

    def completed_tool_result(self, name: str, result: Any, call_id: str = "", **identity: Any) -> None:
        self._tool_result(name, result, call_id=call_id, is_error=False, **identity)

    def completed_tool_error(self, name: str, result: Any, call_id: str = "", **identity: Any) -> None:
        self._tool_result(name, result, call_id=call_id, is_error=True, **identity)

    def _tool_result(
        self,
        name: str,
        result: Any,
        *,
        call_id: str,
        is_error: bool,
        duration_ms: int | None = None,
        **_identity: Any,
    ) -> None:
        event = {
            "type": "tool_result",
            "name": name,
            "output": str(result or ""),
            "call_id": call_id,
        }
        if is_error:
            event["status"] = "error"
        if duration_ms is not None:
            event["duration_ms"] = duration_ms
        self._append(event)

    def delegation_started(self, calls: list[Any], **_identity: Any) -> None:
        self.model_stream_finished()
        self._append({"type": "delegation", "calls": calls})

    def ensure_final_output(self, text: str) -> None:
        """Keep the final child answer inspectable when no message stream exposed it."""
        if text and text not in "".join(self._assistant_texts):
            self.model_stream_finished()
            self._text("assistant", text)
            self.model_stream_finished()


def _first_human_text(messages: list[Any]) -> str:
    for message in messages:
        if str(field(message, "type") or "") in {"human", "user"}:
            return message_text(message)
    return ""


def _same_task_input(description: str, task_input: str) -> bool:
    return description == task_input or (
        len(description) == 200 and task_input.startswith(description)
    )


def _run_events_from_message(message: Any) -> list[dict[str, Any]]:
    """Project one actual child state message into the inspector event schema."""
    if is_tool_message(message):
        return [
            {
                "type": "tool_result",
                "name": str(field(message, "name") or "tool"),
                "output": message_text(message),
                "call_id": str(field(message, "tool_call_id") or field(message, "id") or ""),
                "status": "error" if str(field(message, "status") or "") == "error" else "success",
            }
        ]

    if str(field(message, "type") or "") not in {"ai", "assistant"}:
        return []

    events: list[dict[str, Any]] = []
    reasoning = _message_reasoning(message)
    if reasoning:
        events.append({"type": "reasoning", "text": reasoning})
    text = visible_message_text(message)
    if text:
        events.append({"type": "assistant", "text": text})
    for value in field(message, "tool_calls") or []:
        call = normalized_call(normalized_output_tool_call(value))
        if call.get("name") == "task":
            events.append({"type": "delegation", "calls": [call]})
        else:
            events.append(
                {
                    "type": "tool_call",
                    "name": str(call.get("name") or "tool"),
                    "args": call.get("args", {}),
                    "call_id": str(call.get("id") or ""),
                }
            )
    return events


def _message_reasoning(message: Any) -> str:
    values: list[str] = []
    blocks = field(message, "content_blocks")
    if callable(blocks):
        blocks = blocks()
    if not isinstance(blocks, list | tuple):
        blocks = field(message, "content")
    if isinstance(blocks, list | tuple):
        for block in blocks:
            if isinstance(block, dict) and str(block.get("type") or "") == "reasoning":
                text = str(block.get("reasoning") or block.get("text") or "")
                if text:
                    values.append(text)
    additional = field(message, "additional_kwargs")
    if isinstance(additional, dict):
        text = str(additional.get("reasoning_content") or additional.get("reasoning") or "")
        if text and text not in values:
            values.append(text)
    return "".join(values)


def subagent_task_call_id(subagent: Any) -> str:
    """Return the task invocation that caused this native child run."""
    trigger = getattr(subagent, "trigger_call_id", None)
    if trigger:
        return str(trigger)
    cause = getattr(subagent, "cause", None)
    if isinstance(cause, dict):
        return str(cause.get("tool_call_id") or "")
    return ""


def subagent_origin(subagent: Any) -> str:
    """Return an origin hint for subagents created from a tool namespace."""
    path = getattr(subagent, "path", None)
    if isinstance(path, list | tuple) and any(str(item).startswith("tools:") for item in path):
        return DYNAMIC_TOOL_SUBAGENT
    return ""


async def subagent_result(subagent: Any) -> str:
    """Normalize the final output from a subagent object."""
    output = subagent.output
    if callable(output) and not hasattr(output, "__aiter__") and not hasattr(output, "__await__"):
        output = output()

    if hasattr(output, "__await__"):
        output = await output
    elif hasattr(output, "__aiter__"):
        chunks: list[str] = []
        async for chunk in output:
            chunks.append(tool_output_text(chunk))
        output = "\n".join(filter(None, chunks))

    if isinstance(output, dict) and "messages" in output:
        messages = output["messages"]
        if not messages:
            return ""

        for message in reversed(messages):
            text = visible_message_text(message) or message_text(message)
            if text:
                return text
        return ""

    return tool_output_text(output)
