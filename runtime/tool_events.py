"""Tool-call stream consumption and output normalization helpers."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command


async def consume_tool_calls(tool_calls: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume completed tool-call events and render their output."""
    async for call in tool_calls:
        name = getattr(call, "tool_name", None) or getattr(call, "name", "tool")
        if result is not None:
            result.record_tool_call(str(name), tool_call_id(call))

        output = await tool_call_output(call)
        if isinstance(output, Command):
            continue

        if output:
            text = tool_output_text(output)
            if result is not None:
                result.tool_results.append(text)
            renderer.tool_result(str(name), text, call_id=tool_call_id(call))


async def tool_call_output(call: Any) -> Any:
    """Return a tool call's final output, collecting streamed deltas if needed."""
    deltas: list[str] = []

    if hasattr(call, "__aiter__"):
        async for delta in call:
            text = tool_output_text(delta)
            if text:
                deltas.append(text)

    if isinstance(call, dict):
        if call.get("output") is not None:
            return call["output"]
        if call.get("error") is not None:
            return call["error"]
        return "".join(deltas)

    output = getattr(call, "output", None)
    if output is not None:
        return output

    error = getattr(call, "error", None)
    if error is not None:
        return error

    return "".join(deltas)


def tool_output_text(output: Any) -> str:
    """Convert a LangChain tool output object into displayable text."""
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return str(content)

    return str(output)


def tool_call_id(call: Any) -> str:
    """Return a stable id for a streamed tool call when exposed by the provider."""
    if isinstance(call, dict):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")
