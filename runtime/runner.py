"""Top-level orchestration for one streamed agent turn."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
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
    write approval or ask_user prompt, this function asks the renderer for the
    needed input and resumes the same thread with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()
    approved_fallback_actions: list[dict[str, Any]] = []
    approved_result_start = 0
    fallback_outputs: list[str] = []
    fallback_waiting_for_reply = False
    rerouted_after_approval = False

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
            if fallback_waiting_for_reply:
                if pending_calls or leaked_tool_repr or not result.final_text:
                    result.final_text = fallback_final_text(fallback_outputs)
                    if result.final_text:
                        event_renderer.text_delta(result.final_text)
                        event_renderer.finish_main()
                return result

            if (
                approved_fallback_actions
                and len(result.tool_results) == approved_result_start
                and (pending_calls or leaked_tool_repr)
            ):
                if not rerouted_after_approval:
                    rerouted_after_approval = True
                    payload = Command(
                        update={"messages": approved_tool_messages(approved_fallback_actions, pending_calls)},
                        goto="tools",
                    )
                    continue
                fallback_outputs, fallback_messages = await execute_approved_tool_fallbacks(
                    agent, approved_fallback_actions, pending_calls, event_renderer, result
                )
                approved_fallback_actions = []
                fallback_waiting_for_reply = True
                payload = Command(update={"messages": fallback_messages})
                continue

            if pending_calls:
                names = ", ".join(str(call.get("name") or "tool") for call in pending_calls if isinstance(call, dict))
                raise RuntimeError(f"model returned unexecuted tool call(s): {names or 'tool'}")
            if leaked_tool_repr:
                raise RuntimeError("model returned unexecuted tool call(s): tool")
            return result

        ask_user_interrupt = first_ask_user_interrupt(interrupts)
        if ask_user_interrupt is not None:
            payload = Command(resume=await renderer.ask_user(ask_user_interrupt))
        else:
            decisions = await renderer.ask_approvals(interrupts)
            approved_fallback_actions = approved_actions(interrupts, decisions)
            approved_result_start = len(result.tool_results)
            fallback_outputs = []
            fallback_waiting_for_reply = False
            rerouted_after_approval = False
            payload = Command(resume={"decisions": decisions})




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


def approved_tool_messages(actions: list[dict[str, Any]], pending_calls: list[Any]) -> list[AIMessage]:
    """Build the approved AIMessage shape expected by LangGraph's tools node."""
    tool_calls = []
    for index, action in enumerate(actions):
        name = str(action.get("name") or "")
        args = action.get("args", {})
        if not name or not isinstance(args, dict):
            continue
        call_id = fallback_call_id(action, pending_calls) or f"mira-approved-{name}-{index}"
        tool_calls.append({"name": name, "args": args, "id": call_id})
    if not tool_calls:
        return []
    return [AIMessage(content="", tool_calls=tool_calls)]


async def execute_approved_tool_fallbacks(
    agent: Any,
    actions: list[dict[str, Any]],
    pending_calls: list[Any],
    renderer: Any,
    result: TurnResult,
) -> tuple[list[str], list[ToolMessage]]:
    """Execute approved built-in actions if HITL resume failed to do so."""
    backend = getattr(agent, "mira_backend", None)
    if backend is None:
        raise RuntimeError("approved action was not executed and no backend fallback is available")

    outputs = []
    messages = []
    for action in actions:
        name = str(action.get("name") or "")
        args = action.get("args", {})
        if name not in {"read_file", "write_file", "edit_file", "execute"} or not isinstance(args, dict):
            continue

        call_id = fallback_call_id(action, pending_calls)
        result.record_tool_call(name, call_id)
        renderer.tool_call(name, args, call_id=call_id)
        output = execute_approved_action(backend, name, args)
        if result.record_tool_result(output, call_id, name):
            renderer.tool_result(name, output, call_id=call_id)
        outputs.append(output)
        messages.append(ToolMessage(content=output, name=name, tool_call_id=call_id or f"mira-fallback-{name}"))
    return outputs, messages


def fallback_call_id(action: dict[str, Any], pending_calls: list[Any]) -> str:
    """Return the pending AIMessage tool-call id for an approved fallback action."""
    explicit = str(action.get("id") or action.get("call_id") or action.get("tool_call_id") or "")
    if explicit:
        return explicit
    name = action.get("name")
    args = action.get("args")
    for call in pending_calls:
        if not isinstance(call, dict):
            continue
        if call.get("name") == name and call.get("args") == args:
            return str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or "")
    return ""


def fallback_final_text(outputs: list[str]) -> str:
    """Return a visible assistant reply when HITL fallback executed the tool."""
    if not outputs:
        return ""
    if len(outputs) == 1:
        return outputs[0]
    return "\n\n".join(outputs)


def execute_approved_action(backend: Any, name: str, args: dict[str, Any]) -> str:
    """Run one approved built-in action against the DeepAgents backend."""
    if name == "execute":
        return execute_shell_action(backend, args)
    return execute_filesystem_action(backend, name, args)


def execute_shell_action(backend: Any, args: dict[str, Any]) -> str:
    """Run one approved execute action against the DeepAgents backend."""
    timeout = positive_int(args.get("timeout")) or None
    response = backend.execute(str(args.get("command") or ""), timeout=timeout)
    output = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else None)
    if output is not None:
        return str(output)
    error = getattr(response, "error", None) or (response.get("error") if isinstance(response, dict) else None)
    return f"Error: {error}" if error else str(response)


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
