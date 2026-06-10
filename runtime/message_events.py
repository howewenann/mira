"""Message-stream consumption for reasoning, text, and tool-call events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from runtime.output_events import (
    is_summarization_metadata_message,
    strip_compaction_summary_prefix,
    text_has_compaction_summary_shape,
)
from runtime.usage import has_usage, usage_from_message

COMPACTION_REASONING_MARKERS = (
    "context extraction assistant",
    "primary objective",
    "extract the highest quality/most relevant context",
    "conversation history to replace it",
)


async def consume_messages(messages: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume streamed model messages and render reasoning, text, and tools.

    Provider integrations expose message fields in different shapes. This
    module normalizes those fields into renderer calls while preserving the
    event order reported by the stream.
    """
    async for message in messages:
        # Reasoning and response text are independent stream fields. Render
        # both before tool calls so the terminal follows the provider event
        # order as closely as possible.
        await _consume_reasoning(message, renderer)
        await _consume_text(message, renderer)

        # Tool-call chunks may need to be drained before their finalized call
        # list is available.
        call_list = await _finalized_tool_calls(message)
        if call_list:
            _render_tool_calls(call_list, renderer, result)

        if result is not None:
            usage = usage_from_message(message)
            if has_usage(usage):
                result.add_stream_usage(usage)


async def _consume_reasoning(message: Any, renderer: Any) -> None:
    """Render reasoning deltas from a streamed message."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return

    pending = ""
    compacting = False
    async for delta in _text_deltas(reasoning):
        text = str(delta)
        if compacting:
            continue

        pending += text
        if is_compaction_reasoning(pending):
            compacting = True
            pending = ""
            _call_renderer(renderer, "compaction_started")
            continue

        if should_flush_reasoning_probe(pending):
            renderer.reasoning_delta(pending)
            pending = ""

    if compacting:
        _call_renderer(renderer, "compaction_finished")
    elif pending:
        renderer.reasoning_delta(pending)


def is_compaction_reasoning(text: str) -> bool:
    """Return whether streamed reasoning belongs to DeepAgents compaction."""
    lowered = text.lower()
    return all(marker in lowered for marker in COMPACTION_REASONING_MARKERS)


def should_flush_reasoning_probe(text: str) -> bool:
    """Return whether buffered reasoning is unlikely to be compaction metadata."""
    lowered = text.lower()
    if "thinking process" in lowered and len(text) < 600:
        return False
    if len(text) >= 600:
        return True
    if "\n\n" in text and not any(marker in lowered for marker in COMPACTION_REASONING_MARKERS):
        return True
    return False


def _call_renderer(renderer: Any, method: str) -> None:
    """Call an optional renderer method."""
    callback: Callable[[], None] | None = getattr(renderer, method, None)
    if callback is not None:
        callback()


async def _consume_text(message: Any, renderer: Any) -> None:
    """Render assistant text deltas from a streamed message."""
    if is_summarization_metadata_message(message):
        return

    msg_text = getattr(message, "text", None)
    if msg_text is None:
        return

    if hasattr(msg_text, "__aiter__"):
        await _consume_streamed_text(msg_text, renderer)
        return

    text = await msg_text if hasattr(msg_text, "__await__") else msg_text
    visible, had_summary = strip_compaction_summary_prefix(str(text or ""))
    if visible or (text and not had_summary):
        renderer.text_delta(visible)


async def _consume_streamed_text(value: Any, renderer: Any) -> None:
    """Render streamed text while stripping a leading compaction summary."""
    pending = ""
    probing = True
    async for delta in _text_deltas(value):
        if not probing:
            renderer.text_delta(delta)
            continue

        pending += delta
        visible, had_summary = strip_compaction_summary_prefix(pending)
        if had_summary:
            if visible:
                renderer.text_delta(visible)
                pending = ""
                probing = False
            continue

        if should_flush_text_probe(pending):
            renderer.text_delta(pending)
            pending = ""
            probing = False

    if probing and pending and not text_has_compaction_summary_shape(pending):
        renderer.text_delta(pending)


def should_flush_text_probe(text: str) -> bool:
    """Return whether buffered assistant text is not a compaction summary."""
    if len(text) >= 600:
        return True
    return "\n\n" in text and "## session intent" not in text.lower()


async def _text_deltas(value: Any) -> AsyncIterator[str]:
    """Yield text from plain values, awaitables, or async iterables."""
    if hasattr(value, "__aiter__"):
        async for delta in value:
            if delta:
                yield str(delta)
        return

    text = await value if hasattr(value, "__await__") else value
    if text:
        yield str(text)


async def _finalized_tool_calls(message: Any) -> list[Any]:
    """Return the finalized tool-call list for a streamed message."""
    calls = getattr(message, "tool_calls", None)
    if calls is None:
        return []

    # Some providers stream tool-call chunks through this field before exposing
    # the final parsed list through get().
    if hasattr(calls, "__aiter__"):
        async for _ in calls:
            pass

    finalized = calls.get() if hasattr(calls, "get") else (calls or [])
    if hasattr(finalized, "__await__"):
        finalized = await finalized

    return list(finalized or [])


def _render_tool_calls(call_list: list[Any], renderer: Any, result: Any | None) -> None:
    """Render task delegations and normal tool calls from a message."""
    task_calls = [call for call in call_list if _call_name(call) == "task"]
    if task_calls:
        renderer.delegation_started(task_calls)

    for call in call_list:
        name = _call_name(call)
        if result is not None:
            result.tool_calls.append(str(name))

        if name != "task":
            renderer.tool_call(str(name), _call_args(call))


def _call_name(call: Any) -> str:
    """Extract a tool-call name from a dict or object."""
    if isinstance(call, dict):
        return str(call.get("name", "tool"))

    return str(getattr(call, "name", "tool"))


def _call_args(call: Any) -> Any:
    """Extract tool-call args from a dict or object."""
    if isinstance(call, dict):
        return call.get("args", {})

    return getattr(call, "args", {})
