"""Top-level orchestration for one streamed agent turn."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from runtime.message_events import consume_messages
from runtime.output_events import (
    capture_output,
    collect_interrupts,
    final_text,
    output_has_tool_call_repr,
    output_tool_calls,
    output_tool_results,
)
from runtime.subagent_events import consume_subagents
from runtime.tool_call_args import tool_call_args
from runtime.tool_events import consume_tool_calls
from runtime.usage import (
    empty_usage,
    has_context_usage,
    has_usage,
    item_context_source,
    merge_usage,
    positive_int,
    select_context_usage,
    usage_from_output,
)


@dataclass
class TurnResult:
    """Summary of one agent turn used by REPL planning logic."""

    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0
    context_source: str = "unknown"
    usage_source: str = "unknown"
    _stream_usage: dict[str, Any] = field(default_factory=empty_usage, repr=False)
    _seen_tool_call_ids: set[str] = field(default_factory=set, repr=False)
    _seen_tool_result_ids: set[str] = field(default_factory=set, repr=False)
    _seen_tool_result_values: set[tuple[str, str]] = field(default_factory=set, repr=False)

    @property
    def usage(self) -> dict[str, Any]:
        """Return normalized token usage for this turn."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
            "context_source": self.context_source,
            "source": self.usage_source,
        }

    def add_usage(self, usage: dict[str, Any]) -> None:
        """Add one usage object to the persisted turn totals."""
        input_tokens = positive_int(usage.get("input_tokens"))
        output_tokens = positive_int(usage.get("output_tokens"))
        total_tokens = positive_int(usage.get("total_tokens")) or input_tokens + output_tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.set_context_usage(select_context_usage(usage))
        if self.usage_source == "unknown" and usage.get("source"):
            self.usage_source = str(usage["source"])

    def add_stream_usage(self, usage: dict[str, Any]) -> None:
        """Capture streamed usage as a fallback for providers that omit final usage."""
        self._stream_usage = merge_usage(self._stream_usage, usage)

    def set_context_usage(self, usage: dict[str, Any]) -> None:
        """Set current context usage without changing cumulative In/Out totals."""
        if not has_context_usage(usage):
            return
        selected = select_context_usage(usage)
        self.context_tokens = positive_int(selected.get("context_tokens"))
        self.context_source = item_context_source(selected)
        if self.usage_source == "unknown" and self.context_source != "unknown":
            self.usage_source = self.context_source

    def commit_loop_usage(self, output: Any) -> dict[str, Any]:
        """Commit LangChain token usage and return the per-loop usage delta."""
        committed = empty_usage()
        output_usage = usage_from_output(output)
        if has_usage(output_usage):
            self.add_usage(output_usage)
            committed = select_context_usage(output_usage)
        elif has_usage(self._stream_usage):
            self.add_usage(self._stream_usage)
            committed = select_context_usage(self._stream_usage)
        self._stream_usage = empty_usage()
        return committed

    def record_tool_call(self, name: str, call_id: str = "") -> bool:
        """Record one tool call while avoiding duplicate id-based reports."""
        if call_id:
            if call_id in self._seen_tool_call_ids:
                return False
            self._seen_tool_call_ids.add(call_id)
        self.tool_calls.append(name)
        return True

    def record_tool_result(self, text: str, call_id: str = "", name: str = "") -> bool:
        """Record one tool result while avoiding duplicate id-based reports."""
        value_key = (name, text)
        if value_key in self._seen_tool_result_values:
            return False
        if call_id:
            if call_id in self._seen_tool_result_ids:
                return False
            self._seen_tool_result_ids.add(call_id)
        self._seen_tool_result_values.add(value_key)
        self.tool_results.append(text)
        return True


class SubagentRequestRenderer:
    """Fill empty subagent request text from preceding task delegations."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self._pending_requests: deque[str] = deque()
        self._pending_subagents: deque[str] = deque()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        for description in task_descriptions(calls):
            if self._pending_subagents:
                callback = getattr(self.renderer, "subagent_request_updated", None)
                if callable(callback):
                    callback(self._pending_subagents.popleft(), description)
            else:
                self._pending_requests.append(description)
        self.renderer.delegation_started(calls)

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        queued_request = self._pending_requests.popleft() if self._pending_requests else ""
        request = task_input or queued_request
        self.renderer.subagent_started(subagent, request)
        if not request:
            self._pending_subagents.append(subagent)


async def run_turn(
    agent: Any,
    text: str,
    renderer: Any,
    thread_id: str,
    usage_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TurnResult:
    """Stream one top-level agent turn and handle HITL approval loops.

    DeepAgents exposes separate async event streams for messages, tool calls,
    subagents, and final output. MIRA consumes them concurrently so the terminal
    can update as soon as each event arrives. If LangGraph interrupts for a
    write approval, ask_user prompt, or structured planning prompt, this
    function asks the renderer for the needed input and resumes the same thread
    with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        event_renderer = SubagentRequestRenderer(renderer)
        output: dict[str, Any] = {}
        tool_call_start = len(result.tool_calls)
        waiting_started = getattr(renderer, "waiting_started", None)
        if callable(waiting_started):
            waiting_started()

        await asyncio.gather(
            consume_messages(stream.messages, event_renderer, result, render_normal_tools=False),
            consume_tool_calls(stream.tool_calls, event_renderer, result),
            consume_subagents(stream.subagents, event_renderer),
            capture_output(stream.output(), output),
        )

        render_output_tool_results(output.get("value"), event_renderer, result)
        result.final_text = final_text(output.get("value")) or result.final_text
        usage_delta = result.commit_loop_usage(output.get("value"))
        if usage_callback is not None and (has_usage(usage_delta) or has_context_usage(usage_delta)):
            usage_callback(usage_delta)
        waiting_finished = getattr(renderer, "waiting_finished", None)
        if callable(waiting_finished):
            waiting_finished()
        renderer.finish_main()
        interrupts = await collect_interrupts(stream, output.get("value"))

        if not interrupts:
            pending_calls = output_tool_calls(output.get("value"))
            leaked_tool_repr = output_has_tool_call_repr(output.get("value"))
            stream_tool_calls_observed = len(result.tool_calls) > tool_call_start

            if pending_calls:
                raise RuntimeError(
                    await unexecuted_tool_call_error(
                        stream,
                        output.get("value"),
                        pending_calls=pending_calls,
                        leaked_tool_repr=False,
                        stream_tool_calls_observed=stream_tool_calls_observed,
                    )
                )
            if leaked_tool_repr:
                raise RuntimeError(
                    await unexecuted_tool_call_error(
                        stream,
                        output.get("value"),
                        pending_calls=[],
                        leaked_tool_repr=True,
                        stream_tool_calls_observed=stream_tool_calls_observed,
                    )
                )
            return result

        plan_interrupt = first_typed_interrupt(interrupts, "present_plan")
        ask_user_interrupt = first_typed_interrupt(interrupts, "ask_user")
        if plan_interrupt is not None:
            payload = Command(resume=await renderer.present_plan(plan_interrupt))
        elif ask_user_interrupt is not None:
            payload = Command(resume=await renderer.ask_user(ask_user_interrupt))
        else:
            decisions = await renderer.ask_approvals(interrupts)
            payload = Command(resume={"decisions": decisions})




def first_ask_user_interrupt(interrupts: list[Any]) -> Any | None:
    """Return the first ask_user interrupt payload, if present."""
    return first_typed_interrupt(interrupts, "ask_user")


def first_typed_interrupt(interrupts: list[Any], interrupt_type: str) -> Any | None:
    """Return the first interrupt payload with the requested type."""
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        if isinstance(value, dict) and value.get("type") == interrupt_type:
            return interrupt
    return None


def interrupt_value(interrupt: Any) -> Any:
    """Extract the LangGraph interrupt value from common payload shapes."""
    return getattr(interrupt, "value", interrupt)


def render_output_tool_results(output: Any, renderer: Any, result: TurnResult) -> None:
    """Render tool results that only appear in the final graph output."""
    recovered_tool_result = getattr(renderer, "recovered_tool_result", None)
    for item in output_tool_results(output):
        text = item["output"]
        call_id = item["call_id"]
        if not result.record_tool_result(text, call_id, item["name"]):
            continue
        if callable(recovered_tool_result):
            recovered_tool_result(item["name"], text, call_id=call_id)
        else:
            renderer.tool_result(item["name"], text, call_id=call_id)


async def unexecuted_tool_call_error(
    stream: Any,
    output: Any,
    *,
    pending_calls: list[Any],
    leaked_tool_repr: bool,
    stream_tool_calls_observed: bool,
) -> str:
    """Return a compact diagnostic for a terminal output with pending tool calls."""
    names = [
        str(call.get("name") or "tool")
        for call in pending_calls
        if isinstance(call, dict)
    ]
    name_text = ", ".join(names) or "tool"
    interrupted = await stream_interrupted(stream)
    diagnostic = (
        f"interrupted={interrupted}; "
        f"stream_tool_calls_observed={stream_tool_calls_observed}; "
        f"leaked_tool_repr={leaked_tool_repr}; "
        f"final_messages={output_message_shapes(output)}"
    )
    return f"native HITL resume returned unexecuted tool call(s): {name_text}; diagnostic: {diagnostic}"


async def stream_interrupted(stream: Any) -> bool:
    """Return whether a LangGraph run stream reported an interrupt."""
    callback = getattr(stream, "interrupted", None)
    if not callable(callback):
        return False
    value = callback()
    if hasattr(value, "__await__"):
        value = await value
    return bool(value)


def output_message_shapes(output: Any) -> list[dict[str, Any]] | str:
    """Return compact final-output message shapes for failure diagnostics."""
    if not isinstance(output, dict):
        return type(output).__name__
    messages = output.get("messages")
    if not isinstance(messages, list):
        return []
    return [message_shape(message) for message in messages[-3:]]


def message_shape(message: Any) -> dict[str, Any]:
    """Return compact class, content, text, and tool-call details for one message."""
    content = message_value(message, "content")
    tool_calls = message_value(message, "tool_calls") or []
    text = message_value(message, "text")
    shape: dict[str, Any] = {
        "class": message.__class__.__name__,
        "content_type": type(content).__name__,
        "tool_calls": compact_tool_calls(tool_calls),
    }
    if isinstance(text, str) and text.strip():
        shape["text_sample"] = compact_sample(text)
    elif isinstance(content, str) and content.strip():
        shape["content_sample"] = compact_sample(content)
    return shape


def compact_tool_calls(tool_calls: Any) -> list[dict[str, str]]:
    """Return names and ids for compact tool-call diagnostics."""
    if not isinstance(tool_calls, list):
        return []
    compact = []
    for call in tool_calls[:5]:
        if isinstance(call, dict):
            compact.append({
                "name": str(call.get("name") or "tool"),
                "id": str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or ""),
            })
    return compact


def message_value(message: Any, key: str) -> Any:
    """Read a field from dict-like and object-like message shapes."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def compact_sample(value: Any, limit: int = 180) -> str:
    """Return a single-line diagnostic sample."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def task_descriptions(calls: list[Any]) -> list[str]:
    """Extract request descriptions from task tool-call payloads."""
    descriptions = []
    for call in calls:
        args = call_args(call)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, json.JSONDecodeError):
                args = {}
        if isinstance(args, dict) and args.get("description"):
            descriptions.append(str(args["description"]))
    return descriptions


def call_args(call: Any) -> Any:
    """Extract tool-call args from a dict or object."""
    return tool_call_args(call)
