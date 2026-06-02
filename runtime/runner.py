"""Top-level orchestration for one streamed agent turn."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

from runtime.message_events import consume_messages
from runtime.output_events import capture_output, collect_interrupts, final_text
from runtime.subagent_events import consume_subagents
from runtime.tool_events import consume_tool_calls


@dataclass
class TurnResult:
    """Summary of one agent turn used by REPL planning logic."""

    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


async def run_turn(agent: Any, text: str, renderer: Any, thread_id: str) -> TurnResult:
    """Stream one top-level agent turn and handle HITL approval loops.

    DeepAgents exposes separate async event streams for messages, tool calls,
    subagents, and final output. MIRA consumes them concurrently so the terminal
    can update as soon as each event arrives. If LangGraph interrupts for a
    write approval or ask_user prompt, this function asks the renderer for the
    needed input and resumes the same thread with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {"messages": [{"role": "user", "content": text}]}
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()

    while True:
        stream = await agent.astream_events(payload, config=config, version="v3")
        output: dict[str, Any] = {}

        await asyncio.gather(
            consume_messages(stream.messages, renderer, result),
            consume_tool_calls(stream.tool_calls, renderer, result),
            consume_subagents(stream.subagents, renderer),
            capture_output(stream.output(), output),
        )

        result.final_text = final_text(output.get("value")) or result.final_text
        renderer.finish_main()
        interrupts = await collect_interrupts(stream, output.get("value"))

        if not interrupts:
            return result

        ask_user_interrupt = first_ask_user_interrupt(interrupts)
        if ask_user_interrupt is not None:
            payload = Command(resume=await renderer.ask_user(ask_user_interrupt))
        else:
            decisions = await renderer.ask_approvals(interrupts)
            payload = Command(resume={"decisions": decisions})


def first_ask_user_interrupt(interrupts: list[Any]) -> Any | None:
    """Return the first ask_user interrupt payload, if present."""
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        if isinstance(value, dict) and value.get("type") == "ask_user":
            return interrupt
    return None


def interrupt_value(interrupt: Any) -> Any:
    """Extract the LangGraph interrupt value from common payload shapes."""
    return getattr(interrupt, "value", interrupt)
