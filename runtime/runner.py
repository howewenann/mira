import asyncio
from collections.abc import Iterable

from langgraph.types import Command


async def run_turn(agent, text: str, renderer, thread_id: str) -> None:
    payload = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        output = {}

        await asyncio.gather(
            _consume_messages(stream.messages, renderer),
            _consume_tool_calls(stream.tool_calls, renderer),
            _consume_subagents(stream.subagents, renderer),
            _capture_output(stream.output(), output),
        )

        renderer.flush_reasoning()
        interrupts = _find_interrupts(output.get("value"))

        if not interrupts:
            renderer.finish_main()
            return

        decisions = await renderer.ask_approvals(interrupts)
        payload = Command(resume={"decisions": decisions})


async def _capture_output(output_stream, output: dict) -> None:
    if hasattr(output_stream, "__aiter__"):
        async for item in output_stream:
            output["value"] = item
        return

    if hasattr(output_stream, "__await__"):
        output["value"] = await output_stream
        return

    output["value"] = output_stream


async def _consume_messages(messages, renderer) -> None:
    async for message in messages:
        reasoning = await _message_reasoning(message)
        if reasoning:
            renderer.add_reasoning(reasoning)

        text = await _message_text(message)
        if text:
            renderer.text(text)

        for call in await _message_tool_calls(message):
            renderer.tool_call(call.get("name", "tool"), call.get("args", {}))


async def _consume_tool_calls(tool_calls, renderer) -> None:
    async for call in tool_calls:
        name = getattr(call, "name", None) or getattr(call, "tool_name", "tool")
        output = await _tool_call_output(call)

        if isinstance(output, Command):
            continue

        if output:
            renderer.tool_result(name, _tool_output_text(output))


async def _consume_subagents(subagents, renderer) -> None:
    tasks = []

    async for subagent in subagents:
        tasks.append(asyncio.create_task(_consume_subagent(subagent, renderer)))

    if tasks:
        await asyncio.gather(*tasks)


async def _consume_subagent(subagent, renderer) -> None:
    name = renderer.subagent_label(subagent)

    async for call in subagent.tool_calls:
        tool_name = getattr(call, "name", None) or getattr(call, "tool_name", "tool")
        output = await _tool_call_output(call)

        if isinstance(output, Command):
            continue

        if output:
            renderer.subagent_tool_result(name, tool_name, _tool_output_text(output))


async def _tool_call_output(call):
    deltas = []

    if hasattr(call, "__aiter__"):
        async for delta in call:
            text = _tool_output_text(delta)
            if text:
                deltas.append(text)

    output = getattr(call, "output", None)
    if output is not None:
        return output

    error = getattr(call, "error", None)
    if error is not None:
        return error

    return "".join(deltas)


async def _message_text(message) -> str:
    text = getattr(message, "text", "")

    if callable(text):
        text = text()

    if hasattr(text, "__await__"):
        text = await text

    if isinstance(text, list):
        return "".join(str(part) for part in text)

    return str(text or "")


async def _message_reasoning(message) -> str:
    reasoning = getattr(message, "reasoning", "")

    if callable(reasoning):
        reasoning = reasoning()

    if hasattr(reasoning, "__await__"):
        reasoning = await reasoning

    return str(reasoning or "")


async def _message_tool_calls(message) -> Iterable[dict]:
    calls = getattr(message, "tool_calls", None)

    if callable(calls):
        calls = calls()

    if hasattr(calls, "__await__"):
        calls = await calls

    if hasattr(calls, "__aiter__"):
        return [call async for call in calls]

    return calls or []


def _tool_output_text(output) -> str:
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return str(content)

    return str(output)


def _find_interrupts(value) -> list:
    if value is None:
        return []

    if isinstance(value, dict):
        return value.get("__interrupt__", []) or value.get("interrupts", [])

    interrupts = getattr(value, "__interrupt__", None) or getattr(value, "interrupts", None)
    return interrupts or []
