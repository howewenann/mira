"""Coordinator message-stream consumption for reasoning and text."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from runtime.compaction_filter import (
    ReasoningFilter,
    TextFilter,
    call_renderer,
)
from runtime.compaction_state import compaction_active
from runtime.output_events import (
    is_summarization_metadata_message,
    normalize_response_delta,
    strip_compaction_summary_prefix,
    visible_message_text,
)
from runtime.protocol_events import event_delta, is_raw_message_stream, is_tool_call_delta
from runtime.tool_call_args import ToolCallDrafts, normalized_call, tool_call_name
from runtime.usage import has_usage, usage_from_message


async def consume_messages(
    messages: Any,
    renderer: Any,
    result: Any | None = None,
    *,
    render_normal_tools: bool = True,
) -> None:
    """Consume coordinator messages and fallback provider tool-call chunks.

    DeepAgents' documented ``stream.tool_calls`` projection owns normal
    tool/task rendering in the runner. Message-level tool-call chunks are kept
    as a provider fallback for live draft UI when exposed by a chat model.
    """
    async for message in messages:
        if is_raw_message_stream(message):
            await _consume_ordered_message_stream(message, renderer)
            call_renderer(renderer, "model_stream_finished")
            call_list = await _finalized_tool_calls(message, renderer)
        else:
            if compaction_active():
                await _consume_compaction_message(message, renderer)
                call_renderer(renderer, "model_stream_finished")
                call_list = await _finalized_tool_calls(message, renderer)
                if call_list:
                    render_tool_calls(call_list, renderer, result, render_normal_tools=render_normal_tools)
                if result is not None:
                    usage = usage_from_message(message)
                    if has_usage(usage):
                        result.add_stream_usage(usage)
                continue
            call_task = asyncio.create_task(_finalized_tool_calls(message, renderer))
            compacting = await _consume_reasoning(message, renderer)
            await _consume_text(message, renderer, allow_compaction_summary=compacting)
            call_renderer(renderer, "model_stream_finished")
            call_list = await call_task

        if call_list:
            render_tool_calls(call_list, renderer, result, render_normal_tools=render_normal_tools)

        if result is not None:
            usage = usage_from_message(message)
            if has_usage(usage):
                result.add_stream_usage(usage)


async def _consume_reasoning(message: Any, renderer: Any) -> bool:
    """Render reasoning deltas from a streamed message."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return False

    if compaction_active():
        call_renderer(renderer, "compaction_started")
        async for _ in _text_deltas(reasoning):
            pass
        call_renderer(renderer, "compaction_finished")
        return True

    reasoning_filter = ReasoningFilter(renderer)
    async for delta in _text_deltas(reasoning):
        reasoning_filter.push(str(delta))

    reasoning_filter.finish()
    return reasoning_filter.was_compaction


async def _consume_ordered_message_stream(message: Any, renderer: Any) -> None:
    """Render raw ChatModelStream events in provider order."""
    if compaction_active():
        call_renderer(renderer, "compaction_started")
        async for _ in message:
            pass
        call_renderer(renderer, "compaction_finished")
        return

    reasoning_filter = ReasoningFilter(renderer)
    text_filter = TextFilter(renderer, allow_compaction_summary=lambda: reasoning_filter.compacting)
    tool_drafts = ToolCallDrafts(renderer)

    async for event in message:
        delta = event_delta(event)
        delta_type = str(delta.get("type") or "")
        if delta_type == "reasoning-delta":
            reasoning_filter.push(str(delta.get("reasoning") or delta.get("text") or ""))
        elif delta_type == "text-delta":
            text_filter.push(str(delta.get("text") or ""))
        elif is_tool_call_delta(delta_type):
            tool_drafts.push(event)

    reasoning_filter.finish()
    text_filter.finish()


async def _consume_text(message: Any, renderer: Any, *, allow_compaction_summary: bool = False) -> None:
    """Render assistant text deltas from a streamed message."""
    if is_summarization_metadata_message(message):
        return
    if compaction_active():
        call_renderer(renderer, "compaction_started")
        await _drain_message_text(message)
        call_renderer(renderer, "compaction_finished")
        return

    msg_text = getattr(message, "text", None)
    if msg_text is None or callable(msg_text):
        text = visible_message_text(message)
        if text:
            renderer.text_delta(text)
        return

    if hasattr(msg_text, "__aiter__"):
        await _consume_streamed_text(msg_text, renderer, allow_compaction_summary=allow_compaction_summary)
        return

    text = await msg_text if hasattr(msg_text, "__await__") else msg_text
    text = normalize_response_delta("", text)
    visible, had_summary = strip_compaction_summary_prefix(str(text or ""))
    if had_summary:
        if not allow_compaction_summary:
            call_renderer(renderer, "compaction_started")
            call_renderer(renderer, "compaction_finished")
        if visible:
            renderer.text_delta(visible)
        return

    if text:
        renderer.text_delta(str(text))


async def _consume_streamed_text(value: Any, renderer: Any, *, allow_compaction_summary: bool = False) -> None:
    """Render streamed text while stripping a leading compaction summary."""
    text_filter = TextFilter(renderer, allow_compaction_summary=lambda: allow_compaction_summary)
    async for delta in _text_deltas(value):
        text_filter.push(delta)
    text_filter.finish()


async def _consume_compaction_message(message: Any, renderer: Any) -> None:
    """Drain a live compaction message without recording reasoning or text."""
    call_renderer(renderer, "compaction_started")
    reasoning = getattr(message, "reasoning", None)
    if reasoning is not None:
        async for _ in _text_deltas(reasoning):
            pass
    await _drain_message_text(message)
    call_renderer(renderer, "compaction_finished")


async def _drain_message_text(message: Any) -> None:
    """Consume text projections so provider stream tasks can complete."""
    msg_text = getattr(message, "text", None)
    if msg_text is None or callable(msg_text):
        return
    async for _ in _text_deltas(msg_text):
        pass


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


async def _finalized_tool_calls(message: Any, renderer: Any) -> list[Any]:
    """Return the finalized tool-call list for a streamed message."""
    calls = getattr(message, "tool_calls", None)
    if calls is None:
        return []

    tool_drafts = ToolCallDrafts(renderer)
    if hasattr(calls, "__aiter__"):
        async for chunk in calls:
            tool_drafts.push(chunk)

    finalized = calls.get() if hasattr(calls, "get") else (calls or [])
    if hasattr(finalized, "__await__"):
        call_renderer(renderer, "model_activity")
        finalized = await finalized

    return list(finalized or [])


def render_tool_calls(
    call_list: list[Any],
    renderer: Any,
    result: Any | None,
    *,
    render_normal_tools: bool = True,
) -> None:
    """Render fallback finalized calls from message projections."""
    normalized = [normalized_call(call) for call in call_list]
    task_calls = [call for call in call_list if tool_call_name(call) == "task"]
    if render_normal_tools and task_calls:
        renderer.delegation_started(task_calls)

    for call in normalized:
        name = str(call["name"])
        call_id = str(call.get("id") or "")
        if name == "task":
            if result is not None and render_normal_tools:
                result.record_tool_call(name, call_id)
            continue

        if not render_normal_tools:
            continue

        if result is not None and not result.record_tool_call(name, call_id):
            continue

        renderer.tool_call(name, call.get("args", {}), call_id=call_id)
