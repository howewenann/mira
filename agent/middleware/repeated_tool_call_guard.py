"""Consecutive equivalent tool-call protection, independent of MIRA.

Adapted from DeepAgents PR #3581 by kxleee (李凯祥):
https://github.com/langchain-ai/deepagents/pull/3581

MIT License
Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Command

_TERMINATED = "repeated_tool_call_guard_terminated"


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Allow repeats, offer recovery errors, then end only the current turn.

    ``max_repeats`` identical consecutive requests execute; the next
    ``max_blocked_attempts`` return errors. The following request returns a
    terminal tool error. A before-model hook then appends a visible diagnostic
    and jumps to end without another model call. Every pending parallel call
    still receives its own result before ending the turn.

    Counts come only from retained messages since the last HumanMessage.
    Parallel calls use AIMessage declaration order, never completion order.
    Failures count too: use different arguments/tools or a new user turn for
    legitimate retries. This is not semantic progress or cycle detection.
    LangChain's end route still runs after-agent hooks. A hook that requests
    a new revision can start another attempt; this guard does not override
    that hook's policy. Each new HumanMessage resets the repetition sequence.

    Example::

        create_agent(model, tools=tools, middleware=[
            RepeatedToolCallMiddleware(max_repeats=3, max_blocked_attempts=2),
        ])
    """

    def __init__(self, *, max_repeats: int = 3, max_blocked_attempts: int = 2) -> None:
        super().__init__()
        for name, value in (
            ("max_repeats", max_repeats),
            ("max_blocked_attempts", max_blocked_attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_repeats = max_repeats
        self.max_blocked_attempts = max_blocked_attempts

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        blocked = self._blocked_response(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        blocked = self._blocked_response(request)
        return blocked if blocked is not None else await handler(request)

    def _blocked_response(self, request: ToolCallRequest) -> ToolMessage | None:
        count = _consecutive_repeat_count(request)
        if count <= self.max_repeats:
            return None
        terminated = count > self.max_repeats + self.max_blocked_attempts
        name = request.tool_call["name"]
        content = (
            f"Repeated tool call blocked: `{name}` has been requested {count} "
            "consecutive times with identical schema-normalized arguments. "
            f"Only {self.max_repeats} identical calls may execute. "
            "This request was not executed. Use existing results, change "
            "strategy, or finish your response."
        )
        if terminated:
            content += (
                f" Recovery attempts exhausted ({self.max_blocked_attempts}); "
                "the current agent turn has been stopped by the repeated tool call guard."
            )
        return ToolMessage(
            content=content,
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
            additional_kwargs={_TERMINATED: True} if terminated else {},
        )

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # ToolNode has finished the entire batch, so all tool IDs are paired.
        # A checkpointed ToolMessage carries the decision across interrupts;
        # a new HumanMessage naturally resets it, with no instance counters.
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage) and message.additional_kwargs.get(_TERMINATED):
                return {
                    "messages": [AIMessage(content=str(message.content))],
                    "jump_to": "end",
                }
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def _consecutive_repeat_count(request: ToolCallRequest) -> int:
    current = request.tool_call
    fingerprint = _tool_call_fingerprint(current, request.tool)
    calls: list[Mapping[str, Any]] = []
    for message in request.state.get("messages", []):
        if isinstance(message, HumanMessage):
            calls.clear()
        elif isinstance(message, AIMessage):
            calls.extend(message.tool_calls)

    # Search backwards: an earlier turn may have reused the provider's ID.
    for index in range(len(calls) - 1, -1, -1):
        if current.get("id") is not None and calls[index].get("id") == current["id"]:
            calls = calls[:index + 1]
            break
    else:
        calls.append(current)

    count = 0
    for call in reversed(calls):
        if _tool_call_fingerprint(call, request.tool) != fingerprint:
            break
        count += 1
    return count


def _tool_call_fingerprint(call: Mapping[str, Any], tool: BaseTool | None) -> str:
    name = call.get("name", "")
    args = call.get("args", {})
    normalized = args
    if tool is not None and tool.name == name:
        # Preserve the original schema's validators/configuration. The public
        # tool-call schema is a fallback for injected runtime fields, which
        # the model cannot supply. Invalid input still reaches the handler.
        try:
            normalized = _normalize_args(args, tool.args_schema or tool.tool_call_schema)
        except (AttributeError, TypeError, ValueError):
            try:
                normalized = _normalize_args(args, tool.tool_call_schema)
            except (AttributeError, TypeError, ValueError):
                normalized = args
    return json.dumps(
        _to_jsonable({"name": name, "args": normalized}),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _normalize_args(args: Any, schema: Any) -> Any:
    if hasattr(schema, "model_validate"):
        return schema.model_validate(args).model_dump(mode="json", exclude_none=False)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(args).dict()
    return args


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


__all__ = ["RepeatedToolCallMiddleware"]
