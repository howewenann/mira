from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command


@dataclass
class TurnResult:
    """Small summary of one agent turn used by REPL planning logic."""

    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


async def run_turn(agent: Any, text: str, renderer: Any, thread_id: str) -> TurnResult:
    """Stream one top-level agent turn and handle HITL approval loops.

    DeepAgents exposes separate async event streams for messages, tool calls,
    subagents, and final output. MIRA consumes them concurrently so the terminal
    can update as soon as each event arrives. If LangGraph interrupts for a
    write approval, this function asks the renderer for decisions and resumes
    the same thread with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        output: dict[str, Any] = {}

        await asyncio.gather(
            _consume_messages(stream.messages, renderer, result),
            _consume_tool_calls(stream.tool_calls, renderer, result),
            _consume_subagents(stream.subagents, renderer),
            _capture_output(stream.output(), output),
        )

        result.final_text = _final_text(output.get("value")) or result.final_text
        renderer.finish_main()
        interrupts = await _collect_interrupts(stream, output.get("value"))

        if not interrupts:
            return result

        decisions = await renderer.ask_approvals(interrupts)
        payload = Command(resume={"decisions": decisions})


async def _capture_output(output_stream: Any, output: dict[str, Any]) -> None:
    """Store the last final-output value from DeepAgents."""
    if hasattr(output_stream, "__aiter__"):
        async for item in output_stream:
            output["value"] = item
        return

    if hasattr(output_stream, "__await__"):
        output["value"] = await output_stream
        return

    output["value"] = output_stream


async def _consume_messages(messages: Any, renderer: Any, result: TurnResult | None = None) -> None:
    """Consume streamed model messages and render reasoning, text, and tools.

    LangChain message fields may be plain values, awaitables, or async
    iterables depending on the provider. Each branch normalizes that shape into
    text deltas before sending them to the renderer.
    """
    async for message in messages:
        reasoning = getattr(message, "reasoning", None)
        if reasoning is not None:
            if hasattr(reasoning, "__aiter__"):
                async for delta in reasoning:
                    if delta:
                        renderer.reasoning_delta(str(delta))
            else:
                text = await reasoning if hasattr(reasoning, "__await__") else reasoning
                if text:
                    renderer.reasoning_delta(str(text))

        msg_text = getattr(message, "text", None)
        if msg_text is not None:
            if hasattr(msg_text, "__aiter__"):
                async for delta in msg_text:
                    if delta:
                        renderer.text_delta(str(delta))
            else:
                text = await msg_text if hasattr(msg_text, "__await__") else msg_text
                if text:
                    renderer.text_delta(str(text))

        calls = getattr(message, "tool_calls", None)
        if calls is not None:
            if hasattr(calls, "__aiter__"):
                async for _ in calls:
                    pass
            finalized = calls.get() if hasattr(calls, "get") else (calls or [])
            if hasattr(finalized, "__await__"):
                finalized = await finalized
            call_list = finalized or []

            task_calls = [
                call
                for call in call_list
                if (call.get("name") if isinstance(call, dict) else getattr(call, "name", "")) == "task"
            ]
            if task_calls:
                renderer.delegation_started(task_calls)

            for call in call_list:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "tool")
                if result is not None:
                    result.tool_calls.append(str(name))
                if name != "task":
                    args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
                    renderer.tool_call(str(name), args)


async def _consume_tool_calls(tool_calls: Any, renderer: Any, result: TurnResult | None = None) -> None:
    """Consume completed tool-call events and render their output."""
    async for call in tool_calls:
        name = getattr(call, "tool_name", None) or getattr(call, "name", "tool")
        if result is not None:
            result.tool_calls.append(str(name))
        output = await _tool_call_output(call)

        if isinstance(output, Command):
            continue

        if output:
            text = _tool_output_text(output)
            if result is not None:
                result.tool_results.append(text)
            renderer.tool_result(str(name), text)


async def _consume_subagents(subagents: Any, renderer: Any) -> None:
    """Consume subagent streams while a small spinner animation runs."""
    if hasattr(renderer, "start_subagent_live"):
        renderer.start_subagent_live()
    animation = asyncio.create_task(_animate_subagents(renderer))
    tasks: list[asyncio.Task[None]] = []

    try:
        async for subagent in subagents:
            task = asyncio.create_task(_consume_subagent(subagent, renderer))
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


async def _animate_subagents(renderer: Any) -> None:
    """Tick the subagent spinner until the parent task cancels it."""
    while True:
        if hasattr(renderer, "tick_subagents"):
            renderer.tick_subagents()
        await asyncio.sleep(0.12)


async def _consume_subagent(subagent: Any, renderer: Any) -> None:
    """Render one subagent lifecycle and capture its final answer text."""
    name = renderer.subagent_label(subagent)
    renderer.subagent_started(name, getattr(subagent, "task_input", ""))

    try:
        output = subagent.output
        if callable(output) and not hasattr(output, "__aiter__") and not hasattr(output, "__await__"):
            output = output()

        if hasattr(output, "__await__"):
            output = await output
        elif hasattr(output, "__aiter__"):
            chunks: list[str] = []
            async for chunk in output:
                chunks.append(_tool_output_text(chunk))
            output = "\n".join(filter(None, chunks))

        if isinstance(output, dict) and "messages" in output:
            messages = output["messages"]
            if messages:
                last = messages[-1]
                result = last.text if hasattr(last, "text") else str(last)
            else:
                result = ""
        else:
            result = _tool_output_text(output)
    except Exception as exc:
        result = f"error: {exc}"

    renderer.subagent_finished(name, result=str(result))


async def _tool_call_output(call: Any) -> Any:
    """Return a tool call's final output, collecting streamed deltas if needed."""
    deltas: list[str] = []

    if hasattr(call, "__aiter__"):
        async for delta in call:
            text = _tool_output_text(delta)
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


def _tool_output_text(output: Any) -> str:
    """Convert a LangChain tool output object into displayable text."""
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return str(content)

    return str(output)


def _final_text(output: Any) -> str:
    """Extract final assistant text from a DeepAgents output payload."""
    if not isinstance(output, dict):
        return ""

    messages = output.get("messages") or []
    if not messages:
        return ""

    return _message_text(messages[-1])


def _message_text(message: Any) -> str:
    """Extract plain text from common LangChain message content shapes."""
    text = getattr(message, "text", None)
    if text is not None:
        return str(text)

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    return ""


def _find_interrupts(value: Any) -> list[Any]:
    """Find interrupts stored on an output value or output dictionary."""
    if value is None:
        return []

    if isinstance(value, dict):
        return value.get("__interrupt__", []) or value.get("interrupts", [])

    interrupts = getattr(value, "__interrupt__", None) or getattr(value, "interrupts", None)
    return interrupts or []


async def _collect_interrupts(stream: Any, output_value: Any) -> list[Any]:
    """Prefer stream interrupts, then fall back to interrupts in final output."""
    interrupts = await _stream_interrupts(stream)
    if interrupts:
        return interrupts

    return _find_interrupts(output_value)


async def _stream_interrupts(stream: Any) -> list[Any]:
    """Return interrupts from a DeepAgents stream object if it exposes them."""
    interrupts = getattr(stream, "interrupts", None)

    if callable(interrupts):
        interrupts = interrupts()

    if hasattr(interrupts, "__await__"):
        interrupts = await interrupts

    return interrupts or []
