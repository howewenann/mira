"""Tool-call stream consumption and output normalization helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.types import Command

from runtime.tool_call_args import normalized_call
from runtime.usage import field

CONTROL_TOOLS = {"present_plan", "prepare_goal"}
WATCHER_SHUTDOWN_SECONDS = 0.1


async def consume_tool_calls(tool_calls: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume DeepAgents tool-call projections and render starts promptly."""
    watchers: set[asyncio.Task[None]] = set()
    try:
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

            if name in CONTROL_TOOLS:
                continue

            if is_new_call:
                renderer.tool_call(name, normalized.get("args", {}), call_id=call_id)

            if not supports_completion_watch(call):
                continue
            watchers.add(
                asyncio.create_task(
                    watch_tool_result(call, name, call_id, renderer, result),
                    name=f"mira-tool-result-{call_id or name}",
                )
            )
    except BaseException:
        await cancel_watchers(watchers)
        raise
    else:
        await finish_watchers(watchers)


def supports_completion_watch(call: Any) -> bool:
    """Return whether a call is terminal or exposes a supported completion stream."""
    if field(call, "completed") is not False:
        return True
    return field(call, "output_deltas") is not None or hasattr(call, "__aiter__")


async def watch_tool_result(
    call: Any,
    name: str,
    call_id: str,
    renderer: Any,
    result: Any | None,
) -> None:
    """Follow one ordinary call to completion and deliver its final result."""
    output = await tool_call_output(call)
    if isinstance(output, Command) or not output:
        return

    text = tool_output_text(output)
    recorded = True
    if result is not None:
        recorded = result.record_tool_result(text, call_id, name)
    if not recorded:
        return

    callback = getattr(renderer, "completed_tool_result", None)
    if callable(callback):
        callback(name, text, call_id=call_id)
    else:
        renderer.tool_result(name, text, call_id=call_id)


async def finish_watchers(watchers: set[asyncio.Task[None]]) -> None:
    """Collect owned watchers, bounding cleanup if a provider never terminates one."""
    if not watchers:
        return
    done, pending = await asyncio.wait(watchers, timeout=WATCHER_SHUTDOWN_SECONDS)
    await asyncio.gather(*done, return_exceptions=True)
    await cancel_watchers(pending)


async def cancel_watchers(watchers: set[asyncio.Task[None]]) -> None:
    """Cancel and collect owned watchers without leaking task exceptions."""
    if not watchers:
        return
    for task in watchers:
        if not task.done():
            task.cancel()
    await asyncio.gather(*watchers, return_exceptions=True)


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
