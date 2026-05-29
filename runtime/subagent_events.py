"""Subagent stream consumption and status animation helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from runtime.tool_events import tool_output_text


async def consume_subagents(subagents: Any, renderer: Any) -> None:
    """Consume subagent streams while the status animation is active."""
    if hasattr(renderer, "start_subagent_live"):
        renderer.start_subagent_live()
    animation = asyncio.create_task(animate_subagents(renderer))
    tasks: list[asyncio.Task[None]] = []

    try:
        async for subagent in subagents:
            task = asyncio.create_task(consume_subagent(subagent, renderer))
            tasks.append(task)
            await asyncio.sleep(0)

        if tasks:
            await asyncio.gather(*tasks)
    finally:
        animation.cancel()
        with suppress(asyncio.CancelledError):
            await animation
        if hasattr(renderer, "stop_subagent_live"):
            renderer.stop_subagent_live()


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

        last = messages[-1]
        return str(last.text if hasattr(last, "text") else last)

    return tool_output_text(output)
