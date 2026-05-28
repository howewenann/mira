import asyncio
from collections.abc import Iterable
from contextlib import suppress

from langgraph.types import Command


async def run_turn(agent, text: str, renderer, thread_id: str) -> None:
    payload = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        output = {}
        buffered_text = []

        await asyncio.gather(
            _consume_messages(stream.messages, renderer, buffered_text),
            _consume_tool_calls(stream.tool_calls, renderer),
            _consume_subagents(stream.subagents, renderer),
            _capture_output(stream.output(), output),
        )

        # Flush buffered main-agent text AFTER subagents have stopped
        for chunk in buffered_text:
            renderer.text(chunk)

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


async def _consume_messages(messages, renderer, buffered_text: list) -> None:
    async for message in messages:
        reasoning = await _message_reasoning(message)
        if reasoning:
            renderer.add_reasoning(reasoning)

        text = await _message_text(message)
        if text:
            # Buffer text so it renders after subagent panels are done
            buffered_text.append(text)

        calls = await _message_tool_calls(message)
        task_calls = [call for call in calls if call.get("name") == "task"]
        if task_calls:
            renderer.delegation_started(task_calls)

        for call in calls:
            if call.get("name") != "task":
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
    if hasattr(renderer, "start_subagent_live"):
        renderer.start_subagent_live()
    animation = asyncio.create_task(_animate_subagents(renderer))

    try:
        # Collect subagent stream entries and spawn each task immediately so
        # they run concurrently rather than waiting for all to be collected first.
        tasks = []
        async for subagent in subagents:
            task = asyncio.create_task(_consume_subagent(subagent, renderer))
            tasks.append(task)
            # Yield control so the new task can start running right away
            await asyncio.sleep(0)

        if tasks:
            await asyncio.gather(*tasks)
    finally:
        animation.cancel()
        with suppress(asyncio.CancelledError):
            await animation
        if hasattr(renderer, "stop_subagent_live"):
            renderer.stop_subagent_live()


async def _animate_subagents(renderer) -> None:
    while True:
        if hasattr(renderer, "tick_subagents"):
            renderer.tick_subagents()
        await asyncio.sleep(0.12)


async def _consume_subagent(subagent, renderer) -> None:
    name = renderer.subagent_label(subagent)
    renderer.subagent_started(name, getattr(subagent, "task_input", ""))
    final_call = None

    async for call in subagent.tool_calls:
        tool_name = _tool_call_name(call)
        # Capture args BEFORE awaiting output (output consumption may clear the object)
        args = _tool_call_args(call)
        output = await _tool_call_output(call)

        if isinstance(output, Command):
            continue

        if output:
            final_call = (tool_name, args, _tool_output_text(output))

    if final_call is None:
        renderer.subagent_finished(name)
        return

    tool_name, args, output = final_call
    renderer.subagent_finished(name, tool_name, args, output)


async def _tool_call_output(call):
    deltas = []

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


def _tool_call_name(call) -> str:
    if isinstance(call, dict):
        return call.get("name") or call.get("tool_name") or "tool"

    return getattr(call, "name", None) or getattr(call, "tool_name", "tool")


def _tool_call_args(call):
    import json

    if isinstance(call, dict):
        args = call.get("args") or call.get("input")
        if args:
            return args
        arguments = call.get("arguments") or (call.get("function") or {}).get("arguments")
        if arguments:
            try:
                return json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                return {}
        return {}

    args = getattr(call, "args", None) or getattr(call, "input", None)
    if args:
        return args
    arguments = getattr(call, "arguments", None)
    if arguments:
        try:
            return json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return {}
    return {}


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
