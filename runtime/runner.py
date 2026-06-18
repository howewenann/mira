"""Top-level orchestration for one streamed agent turn."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from agent.context_overflow import DEFAULT_CONTEXT_PRESSURE_FRACTION, pop_compaction_retry, pop_context_floor_tokens
from runtime.message_events import consume_messages
from runtime.output_events import capture_output, collect_interrupts, final_text, output_has_tool_call_repr, output_tool_calls
from runtime.subagent_events import consume_subagents
from runtime.tool_call_args import tool_call_args
from runtime.tool_events import consume_tool_calls
from runtime.usage import (
    TokenCounter,
    context_from_output,
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
    context_floor_tokens: int = 0
    context_source: str = "unknown"
    usage_source: str = "unknown"
    possibly_truncated: bool = False
    _stream_usage: dict[str, Any] = field(default_factory=empty_usage, repr=False)
    _seen_tool_call_ids: set[str] = field(default_factory=set, repr=False)

    @property
    def usage(self) -> dict[str, Any]:
        """Return normalized token usage for this turn."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
            "context_floor_tokens": self.context_floor_tokens,
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
        self.context_floor_tokens = positive_int(selected.get("context_floor_tokens"))
        self.context_source = item_context_source(selected)
        if self.usage_source == "unknown" and self.context_source != "unknown":
            self.usage_source = self.context_source

    def commit_loop_usage(
        self,
        output: Any,
        token_counter: TokenCounter | None = None,
        context_floor_tokens: int = 0,
    ) -> dict[str, Any]:
        """Commit LangChain token usage and return the per-loop usage delta."""
        context_floor_tokens = positive_int(context_floor_tokens)
        committed = empty_usage()
        output_usage = usage_from_output(output)
        if context_floor_tokens:
            output_usage["context_floor_tokens"] = context_floor_tokens
        if has_usage(output_usage):
            self.add_usage(output_usage)
            committed = select_context_usage(output_usage)
        elif has_usage(self._stream_usage):
            if context_floor_tokens:
                self._stream_usage["context_floor_tokens"] = context_floor_tokens
            self.add_usage(self._stream_usage)
            committed = select_context_usage(self._stream_usage)

        context_usage = context_from_output(output, token_counter)
        if context_floor_tokens:
            context_usage["context_floor_tokens"] = context_floor_tokens
            context_usage["context_source"] = "request_estimate.count_tokens"
        if not has_context_usage(committed):
            self.set_context_usage(context_usage)
        if not has_context_usage(committed) and has_context_usage(context_usage):
            selected = select_context_usage(context_usage)
            committed["context_tokens"] = positive_int(selected.get("context_tokens"))
            committed["context_floor_tokens"] = positive_int(selected.get("context_floor_tokens"))
            committed["context_source"] = item_context_source(selected)
            if committed.get("source") == "unknown" and selected.get("source"):
                committed["source"] = str(selected["source"])
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
    token_counter: TokenCounter | None = None,
    usage_callback: Callable[[dict[str, Any]], None] | None = None,
    context_limit_tokens: int | None = None,
    context_pressure_fraction: float = DEFAULT_CONTEXT_PRESSURE_FRACTION,
) -> TurnResult:
    """Stream one top-level agent turn and handle HITL approval loops.

    DeepAgents exposes separate async event streams for messages, tool calls,
    subagents, and final output. MIRA consumes them concurrently so the terminal
    can update as soon as each event arrives. If LangGraph interrupts for a
    write approval or ask_user prompt, this function asks the renderer for the
    needed input and resumes the same thread with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()
    approved_fallback_actions: list[dict[str, Any]] = []
    approved_result_start = 0

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        event_renderer = SubagentRequestRenderer(renderer)
        output: dict[str, Any] = {}
        waiting_started = getattr(renderer, "waiting_started", None)
        if callable(waiting_started):
            waiting_started()

        await asyncio.gather(
            consume_messages(stream.messages, event_renderer, result, render_normal_tools=False),
            consume_tool_calls(stream.tool_calls, event_renderer, result),
            consume_subagents(stream.subagents, event_renderer),
            capture_output(stream.output(), output),
        )

        result.final_text = final_text(output.get("value")) or result.final_text
        usage_delta = result.commit_loop_usage(
            output.get("value"),
            token_counter=token_counter,
            context_floor_tokens=pop_context_floor_tokens(thread_id),
        )
        if usage_callback is not None and (has_usage(usage_delta) or has_context_usage(usage_delta)):
            usage_callback(usage_delta)
        pop_compaction_retry(thread_id)
        waiting_finished = getattr(renderer, "waiting_finished", None)
        if callable(waiting_finished):
            waiting_finished()
        renderer.finish_main()
        interrupts = await collect_interrupts(stream, output.get("value"))

        if not interrupts:
            pending_calls = output_tool_calls(output.get("value"))
            leaked_tool_repr = output_has_tool_call_repr(output.get("value"))
            if (
                approved_fallback_actions
                and len(result.tool_results) == approved_result_start
                and (pending_calls or leaked_tool_repr)
            ):
                await execute_approved_filesystem_fallbacks(agent, approved_fallback_actions, event_renderer, result)
                result.final_text = ""
                return result

            if pending_calls:
                names = ", ".join(str(call.get("name") or "tool") for call in pending_calls if isinstance(call, dict))
                raise RuntimeError(f"model returned unexecuted tool call(s): {names or 'tool'}")
            if leaked_tool_repr:
                raise RuntimeError("model returned unexecuted tool call(s): tool")
            if should_warn_after_cutoff(
                result,
                context_limit_tokens=context_limit_tokens,
                context_pressure_fraction=context_pressure_fraction,
            ):
                result.possibly_truncated = True
            return result

        ask_user_interrupt = first_ask_user_interrupt(interrupts)
        if ask_user_interrupt is not None:
            payload = Command(resume=await renderer.ask_user(ask_user_interrupt))
        else:
            decisions = await renderer.ask_approvals(interrupts)
            approved_fallback_actions = approved_actions(interrupts, decisions)
            approved_result_start = len(result.tool_results)
            payload = Command(resume={"decisions": decisions})


def should_warn_after_cutoff(
    result: TurnResult,
    *,
    context_limit_tokens: int | None,
    context_pressure_fraction: float,
) -> bool:
    """Return whether a completed response may have been cut off near the limit."""
    threshold = context_threshold(context_limit_tokens, context_pressure_fraction)
    if not threshold:
        return False
    if max(positive_int(result.context_tokens), positive_int(result.context_floor_tokens)) < threshold:
        return False
    if result.tool_calls or result.tool_results:
        return False
    return looks_truncated(result.final_text)


def context_threshold(context_limit_tokens: int | None, fraction: float) -> int:
    """Return the configured compaction threshold token count."""
    limit = positive_int(context_limit_tokens)
    if not limit:
        return 0
    try:
        parsed_fraction = float(fraction)
    except (TypeError, ValueError):
        parsed_fraction = DEFAULT_CONTEXT_PRESSURE_FRACTION
    if parsed_fraction <= 0:
        parsed_fraction = DEFAULT_CONTEXT_PRESSURE_FRACTION
    return max(1, int(limit * parsed_fraction))


def looks_truncated(text: str) -> bool:
    """Conservatively identify LM Studio-style silent mid-response cutoffs."""
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if len(stripped.split()) == 1:
        return False
    if stripped.count("```") % 2:
        return True
    if stripped.endswith((".", "!", "?", "\"", "'", ")", "]", "}", "`")):
        return False
    tail = stripped.rsplit(maxsplit=6)[-6:]
    dangling = {"and", "or", "but", "because", "since", "if", "when", "while", "the", "a", "an", "to", "of", "with"}
    if tail and tail[-1].lower().strip(",;:") in dangling:
        return True
    if len(stripped) < 120:
        return True
    return stripped.endswith((",", ";", ":", "-", "—"))


def call_renderer(renderer: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """Call an optional renderer method."""
    callback = getattr(renderer, method, None)
    if not callable(callback):
        return False
    callback(*args, **kwargs)
    return True


def first_ask_user_interrupt(interrupts: list[Any]) -> Any | None:
    """Return the first ask_user interrupt payload, if present."""
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        if isinstance(value, dict) and value.get("type") == "ask_user":
            return interrupt
    return None


def interrupt_value(interrupt: Any) -> Any:
    """Extract the LangGraph interrupt value from common payload shapes."""
    return getattr(interrupt, "value", interrupt)


def approved_actions(interrupts: list[Any], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return filesystem actions explicitly approved by the user."""
    actions = []
    decision_index = 0
    for interrupt in interrupts:
        for action in interrupt_actions(interrupt):
            if decision_index >= len(decisions):
                return actions
            decision = decisions[decision_index]
            decision_index += 1
            if decision.get("type") == "approve" and isinstance(action, dict):
                actions.append(action)
            elif decision.get("type") == "edit":
                edited = decision.get("edited_action")
                if isinstance(edited, dict):
                    actions.append(edited)
    return actions


def interrupt_actions(interrupt: Any) -> list[Any]:
    """Extract approval action requests from a LangGraph interrupt."""
    value = interrupt_value(interrupt)
    if isinstance(value, dict) and value.get("action_requests"):
        return list(value["action_requests"])
    return [value]


async def execute_approved_filesystem_fallbacks(
    agent: Any,
    actions: list[dict[str, Any]],
    renderer: Any,
    result: TurnResult,
) -> None:
    """Execute approved built-in filesystem actions if HITL resume failed to do so."""
    backend = getattr(agent, "mira_backend", None)
    if backend is None:
        raise RuntimeError("approved filesystem action was not executed and no backend fallback is available")

    for action in actions:
        name = str(action.get("name") or "")
        args = action.get("args", {})
        if name not in {"read_file", "write_file", "edit_file"} or not isinstance(args, dict):
            continue

        result.record_tool_call(name, "")
        renderer.tool_call(name, args, call_id="")
        output = execute_filesystem_action(backend, name, args)
        result.tool_results.append(output)
        renderer.tool_result(name, output, call_id="")


def execute_filesystem_action(backend: Any, name: str, args: dict[str, Any]) -> str:
    """Run one approved filesystem action against the DeepAgents backend."""
    if name == "read_file":
        response = backend.read(
            str(args.get("file_path") or args.get("path") or ""),
            offset=positive_int(args.get("offset")),
            limit=positive_int(args.get("limit")) or 2000,
        )
        error = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
        if error:
            return f"Error: {error}"
        file_data = getattr(response, "file_data", None) or (response.get("file_data") if isinstance(response, dict) else None)
        content = getattr(file_data, "content", None) or (file_data.get("content") if isinstance(file_data, dict) else "")
        return str(content)

    if name == "write_file":
        file_path = str(args.get("file_path") or args.get("path") or "")
        response = backend.write(file_path, str(args.get("content") or ""))
        error = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
        return f"Error: {error}" if error else f"Successfully wrote to {file_path}"

    file_path = str(args.get("file_path") or args.get("path") or "")
    response = backend.edit(
        file_path,
        str(args.get("old_string") or ""),
        str(args.get("new_string") or ""),
        bool(args.get("replace_all", False)),
    )
    error = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
    return f"Error: {error}" if error else f"Successfully edited {file_path}"


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
