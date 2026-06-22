"""Subagent stream consumption and status animation helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from runtime.output_events import message_text, visible_message_text
from runtime.tool_events import tool_output_text


async def consume_subagents(subagents: Any, renderer: Any) -> None:
    """Consume subagent streams while the status animation is active."""
    if hasattr(renderer, "start_subagent_live"):
        renderer.start_subagent_live()
    animation = asyncio.create_task(animate_subagents(renderer))
    tasks: list[asyncio.Task[None]] = []
    cancelled = False

    try:
        async for subagent in subagents:
            task = asyncio.create_task(consume_subagent(subagent, renderer))
            tasks.append(task)
            await asyncio.sleep(0)

        if tasks:
            await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        cancelled = True
        await cancel_subagent_tasks(tasks)
        call_renderer(renderer, "subagents_cancelled")
        raise
    except Exception:
        cancelled = True
        await cancel_subagent_tasks(tasks)
        call_renderer(renderer, "subagents_cancelled")
        raise
    finally:
        animation.cancel()
        with suppress(asyncio.CancelledError):
            await animation
        if not cancelled and hasattr(renderer, "stop_subagent_live"):
            renderer.stop_subagent_live()


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
    """Render one subagent lifecycle and capture its final answer text."""
    name = renderer.subagent_label(subagent)
    renderer.subagent_started(name, getattr(subagent, "task_input", ""))

    try:
        result = await subagent_result(subagent)
    except Exception as exc:
        result = f"error: {exc}"

    renderer.subagent_finished(name, result=str(result))


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
