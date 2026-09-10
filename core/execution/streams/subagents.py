"""Subagent stream consumption and status projection helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from itertools import count
from typing import Any

from agent.rubric.graphs import INTERNAL_RUBRIC_GRAPHS
from core.execution.streams.output import message_text, visible_message_text
from core.execution.streams.output import call_renderer as call_frontend
from core.execution.streams.messages import consume_messages
from core.execution.streams.rubric import RUBRIC_TOOL_END, RUBRIC_TOOL_START, RubricEventRenderer
from core.execution.streams.tools import consume_tool_calls, tool_output_text
from session.subagent_runs import now_iso

DYNAMIC_TOOL_SUBAGENT = "dynamic_tool_subagent"
EVAL_SUBAGENT = "eval_subagent"


async def consume_subagents(
    subagents: Any,
    renderer: Any,
    rubric: RubricEventRenderer | None = None,
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

    capture = SubagentTranscriptCapture(renderer, task_call_id)
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
        capture.ensure_final_output(result)
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


class SubagentTranscriptCapture:
    """Project native child streams into the durable run event schema."""

    def __init__(self, renderer: Any, task_call_id: str) -> None:
        self.renderer = renderer
        self.task_call_id = task_call_id
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
