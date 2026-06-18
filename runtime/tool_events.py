"""Tool-call stream consumption and output normalization helpers."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from runtime.tool_call_args import normalized_call, tool_call_args, tool_call_id, tool_call_name
from runtime.usage import field


async def consume_tool_calls(tool_calls: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume DeepAgents tool-call projections and render starts promptly."""
    async for call in tool_calls:
        normalized = normalized_call(call)
        name = str(normalized["name"])
        call_id = str(normalized.get("id") or "")
        is_new_call = True
        if result is not None:
            is_new_call = result.record_tool_call(name, call_id)

        if name == "task":
            if is_new_call:
                renderer.delegation_started([normalized])
            continue

        if is_new_call:
            renderer.tool_call(name, normalized.get("args", {}), call_id=call_id)

        if field(call, "completed") is False:
            continue

        output = await tool_call_output(call)
        if isinstance(output, Command):
            continue

        if output:
            text = tool_output_text(output)
            if result is not None:
                result.tool_results.append(text)
            if name != "task":
                renderer.tool_result(name, text, call_id=call_id)


async def tool_call_output(call: Any) -> Any:
    """Return a tool call's final output, collecting streamed deltas if needed."""
    deltas: list[str] = []

    output_deltas = field(call, "output_deltas")
    if output_deltas is not None:
        async for delta in async_items(output_deltas):
            text = tool_output_text(delta)
            if text:
                deltas.append(text)
    elif hasattr(call, "__aiter__"):
        async for delta in call:
            text = tool_output_text(delta)
            if text:
                deltas.append(text)

    if isinstance(call, dict):
        if call.get("error") is not None:
            return call["error"]
        if call.get("output") is not None:
            return await maybe_await(call["output"])
        return "".join(deltas)

    error = field(call, "error")
    if error is not None:
        return await maybe_await(error)

    output = field(call, "output")
    if output is not None:
        return await maybe_await(output)

    return "".join(deltas)


async def async_items(value: Any) -> Any:
    """Yield items from sync or async iterables, ignoring plain strings."""
    if hasattr(value, "__aiter__"):
        async for item in value:
            yield item
        return

    if isinstance(value, str):
        yield value
        return

    try:
        iterator = iter(value)
    except TypeError:
        if value is not None:
            yield value
        return

    for item in iterator:
        yield item


async def maybe_await(value: Any) -> Any:
    """Resolve awaitables while leaving plain values untouched."""
    return await value if hasattr(value, "__await__") else value


def tool_output_text(output: Any) -> str:
    """Convert a LangChain tool output object into displayable text."""
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return str(content)

    return str(output)


def tool_call_input(call: Any) -> Any:
    """Return streamed tool-call input across DeepAgents event shapes."""
    return tool_call_args(call)
