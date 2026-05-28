import asyncio
from contextlib import suppress
from dataclasses import dataclass, field

from langgraph.types import Command


@dataclass
class TurnResult:
    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


async def run_turn(agent, text: str, renderer, thread_id: str) -> TurnResult:
    payload = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        output = {}

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


async def _capture_output(output_stream, output: dict) -> None:
    if hasattr(output_stream, "__aiter__"):
        async for item in output_stream:
            output["value"] = item
        return

    if hasattr(output_stream, "__await__"):
        output["value"] = await output_stream
        return

    output["value"] = output_stream


async def _consume_messages(messages, renderer, result: TurnResult | None = None) -> None:
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

            task_calls = [c for c in call_list if (c.get("name") if isinstance(c, dict) else getattr(c, "name", "")) == "task"]
            if task_calls:
                renderer.delegation_started(task_calls)

            for call in call_list:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "tool")
                if result is not None:
                    result.tool_calls.append(name)
                if name != "task":
                    args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
                    renderer.tool_call(name, args)


async def _consume_tool_calls(tool_calls, renderer, result: TurnResult | None = None) -> None:
    async for call in tool_calls:
        name = getattr(call, "tool_name", None) or getattr(call, "name", "tool")
        if result is not None:
            result.tool_calls.append(name)
        output = await _tool_call_output(call)

        if isinstance(output, Command):
            continue

        if output:
            text = _tool_output_text(output)
            if result is not None:
                result.tool_results.append(text)
            renderer.tool_result(name, text)


async def _consume_subagents(subagents, renderer) -> None:
    if hasattr(renderer, "start_subagent_live"):
        renderer.start_subagent_live()
    animation = asyncio.create_task(_animate_subagents(renderer))
    tasks = []

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


async def _animate_subagents(renderer) -> None:
    while True:
        if hasattr(renderer, "tick_subagents"):
            renderer.tick_subagents()
        await asyncio.sleep(0.12)


async def _consume_subagent(subagent, renderer) -> None:
    name = renderer.subagent_label(subagent)
    renderer.subagent_started(name, getattr(subagent, "task_input", ""))

    try:
        output = subagent.output
        if callable(output) and not hasattr(output, "__aiter__") and not hasattr(output, "__await__"):
            output = output()

        if hasattr(output, "__await__"):
            output = await output
        elif hasattr(output, "__aiter__"):
            chunks = []
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
    except Exception as e:
        result = f"error: {e}"

    renderer.subagent_finished(name, result=result)


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


def _tool_output_text(output) -> str:
    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return str(content)

    return str(output)


def _final_text(output) -> str:
    if not isinstance(output, dict):
        return ""

    messages = output.get("messages") or []
    if not messages:
        return ""

    return _message_text(messages[-1])


def _message_text(message) -> str:
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


def _find_interrupts(value) -> list:
    if value is None:
        return []

    if isinstance(value, dict):
        return value.get("__interrupt__", []) or value.get("interrupts", [])

    interrupts = getattr(value, "__interrupt__", None) or getattr(value, "interrupts", None)
    return interrupts or []


async def _collect_interrupts(stream, output_value) -> list:
    interrupts = await _stream_interrupts(stream)
    if interrupts:
        return interrupts

    return _find_interrupts(output_value)


async def _stream_interrupts(stream) -> list:
    interrupts = getattr(stream, "interrupts", None)

    if callable(interrupts):
        interrupts = interrupts()

    if hasattr(interrupts, "__await__"):
        interrupts = await interrupts

    return interrupts or []
