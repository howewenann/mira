"""Message-stream consumption for reasoning, text, and tool-call events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


async def consume_messages(messages: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume streamed model messages and render reasoning, text, and tools.

    Provider integrations expose message fields in different shapes. This
    module normalizes those fields into renderer calls while preserving the
    event order reported by the stream.
    """
    async for message in messages:
        # Reasoning and response text are independent stream fields. Render
        # both before tool calls so the terminal follows the provider event
        # order as closely as possible.
        await _consume_reasoning(message, renderer)
        await _consume_text(message, renderer)

        # Tool-call chunks may need to be drained before their finalized call
        # list is available.
        call_list = await _finalized_tool_calls(message)
        if call_list:
            _render_tool_calls(call_list, renderer, result)


async def _consume_reasoning(message: Any, renderer: Any) -> None:
    """Render reasoning deltas from a streamed message."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return

    async for delta in _text_deltas(reasoning):
        renderer.reasoning_delta(delta)


async def _consume_text(message: Any, renderer: Any) -> None:
    """Render assistant text deltas from a streamed message."""
    msg_text = getattr(message, "text", None)
    if msg_text is None:
        return

    async for delta in _text_deltas(msg_text):
        renderer.text_delta(delta)


async def _text_deltas(value: Any) -> AsyncIterator[str]:
    """Yield text from plain values, awaitables, or async iterables."""
    if hasattr(value, "__aiter__"):
        async for delta in value:
            if delta:
                yield str(delta)
        return

    text = await value if hasattr(value, "__await__") else value
    if text:
        yield str(text)


async def _finalized_tool_calls(message: Any) -> list[Any]:
    """Return the finalized tool-call list for a streamed message."""
    calls = getattr(message, "tool_calls", None)
    if calls is None:
        return []

    # Some providers stream tool-call chunks through this field before exposing
    # the final parsed list through get().
    if hasattr(calls, "__aiter__"):
        async for _ in calls:
            pass

    finalized = calls.get() if hasattr(calls, "get") else (calls or [])
    if hasattr(finalized, "__await__"):
        finalized = await finalized

    return list(finalized or [])


def _render_tool_calls(call_list: list[Any], renderer: Any, result: Any | None) -> None:
    """Render task delegations and normal tool calls from a message."""
    task_calls = [call for call in call_list if _call_name(call) == "task"]
    if task_calls:
        renderer.delegation_started(task_calls)

    for call in call_list:
        name = _call_name(call)
        if result is not None:
            result.tool_calls.append(str(name))

        if name != "task":
            renderer.tool_call(str(name), _call_args(call))


def _call_name(call: Any) -> str:
    """Extract a tool-call name from a dict or object."""
    if isinstance(call, dict):
        return str(call.get("name", "tool"))

    return str(getattr(call, "name", "tool"))


def _call_args(call: Any) -> Any:
    """Extract tool-call args from a dict or object."""
    if isinstance(call, dict):
        return call.get("args", {})

    return getattr(call, "args", {})
