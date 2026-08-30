"""Coordinator message-stream consumption for reasoning and text."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from agent.middleware.correction import CORRECTION_SOURCE
from core.execution.streams.message_metadata import MessageInvocationMetadata
from core.execution.streams.output import (
    is_correction_metadata_message,
    is_success_criteria_metadata_message,
    is_summarization_metadata_message,
    normalize_response_delta,
    visible_message_text,
)
from core.execution.streams.provider import event_delta, is_raw_message_stream, is_tool_call_delta
from core.execution.streams.output import call_renderer
from core.execution.streams.tool_args import ToolCallDrafts, normalized_call, tool_call_name
from core.execution.streams.tools import CONTROL_TOOLS
from core.context.usage import has_usage, usage_from_message


async def consume_messages(
    messages: Any,
    renderer: Any,
    result: Any | None = None,
    *,
    render_normal_tools: bool = True,
    invocation_metadata: MessageInvocationMetadata | None = None,
) -> None:
    """Consume coordinator messages and fallback provider tool-call chunks.

    DeepAgents' documented ``stream.tool_calls`` projection owns normal
    tool/task rendering in the runner. Message-level tool-call chunks are kept
    as a provider fallback for live draft UI when exposed by a chat model.
    """
    async for message in messages:
        identity = message_identity(message, invocation_metadata)
        is_success_criteria = (
            invocation_metadata is not None and invocation_metadata.is_success_criteria(message)
        ) or is_success_criteria_metadata_message(message)
        if is_success_criteria:
            await _drain_message(message)
            call_renderer(renderer, "model_stream_finished")
            if result is not None:
                usage = usage_from_message(message)
                if has_usage(usage):
                    result.add_stream_usage(usage)
            continue

        is_compaction = (
            invocation_metadata is not None and invocation_metadata.is_summarization(message)
        ) or is_summarization_metadata_message(message)
        if is_compaction:
            await _consume_compaction_message(message, renderer)
            call_renderer(renderer, "model_stream_finished")
            call_list = await _finalized_tool_calls(
                message,
                renderer,
                result,
                identity=identity,
            )
            if call_list:
                render_tool_calls(
                    call_list,
                    renderer,
                    result,
                    render_normal_tools=render_normal_tools,
                    identity=identity,
                )
            if result is not None:
                usage = usage_from_message(message)
                if has_usage(usage):
                    result.add_stream_usage(usage)
            continue

        is_protocol_message = (
            invocation_metadata is not None
            and invocation_metadata.for_message(message).get("lc_source")
            == CORRECTION_SOURCE
        ) or is_correction_metadata_message(message)
        if is_protocol_message:
            await _drain_message(message)
            call_renderer(renderer, "model_stream_finished")
            continue

        if is_raw_message_stream(message):
            await _consume_ordered_message_stream(
                message,
                renderer,
                result,
                identity=identity,
            )
            call_renderer(renderer, "model_stream_finished")
            call_list = await _finalized_tool_calls(
                message,
                renderer,
                result,
                identity=identity,
            )
        else:
            call_task = asyncio.create_task(
                _finalized_tool_calls(message, renderer, result, identity=identity)
            )
            await _consume_reasoning(message, renderer, identity=identity)
            await _consume_text(message, renderer, identity=identity)
            call_renderer(renderer, "model_stream_finished")
            call_list = await call_task

        if call_list:
            render_tool_calls(
                call_list,
                renderer,
                result,
                render_normal_tools=render_normal_tools,
                identity=identity,
            )

        if result is not None:
            usage = usage_from_message(message)
            if has_usage(usage):
                result.add_stream_usage(usage)


async def _consume_reasoning(
    message: Any,
    renderer: Any,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    """Render reasoning deltas from a streamed message."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return
    async for delta in _text_deltas(reasoning):
        text = str(delta)
        call_renderer(
            renderer,
            "reasoning_delta",
            text,
            content_blocks=({"type": "reasoning", "reasoning": text},),
            **(identity or {}),
        )


async def _consume_ordered_message_stream(
    message: Any,
    renderer: Any,
    result: Any | None = None,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    """Render raw ChatModelStream events in provider order."""
    tool_drafts = ToolCallDrafts(renderer, result, identity=identity)
    source_text = ""

    async for event in message:
        delta = event_delta(event)
        delta_type = str(delta.get("type") or "")
        event_identity = merged_event_identity(identity, event)
        native_block = event.get("content_block") if isinstance(event, dict) else None
        if delta_type == "reasoning-delta":
            text = str(delta.get("reasoning") or delta.get("text") or "")
            call_renderer(
                renderer,
                "reasoning_delta",
                text,
                content_blocks=(native_block or {"type": "reasoning", "reasoning": text},),
                **event_identity,
            )
        elif delta_type == "text-delta":
            text = normalize_response_delta(source_text, delta.get("text"))
            if text:
                source_text += text
                call_renderer(
                    renderer,
                    "text_delta",
                    text,
                    content_blocks=(native_block or {"type": "text", "text": text},),
                    **event_identity,
                )
        elif is_tool_call_delta(delta_type):
            tool_drafts.push(event)
async def _consume_text(
    message: Any,
    renderer: Any,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    """Render assistant text deltas from a streamed message."""
    if is_summarization_metadata_message(message):
        return

    msg_text = getattr(message, "text", None)
    if msg_text is None or callable(msg_text):
        text = visible_message_text(message)
        if text:
            call_renderer(
                renderer,
                "text_delta",
                text,
                content_blocks=normalized_content_blocks(message, text),
                **(identity or {}),
            )
        return

    if hasattr(msg_text, "__aiter__"):
        await _consume_streamed_text(msg_text, renderer, identity=identity)
        return

    text = await msg_text if hasattr(msg_text, "__await__") else msg_text
    text = normalize_response_delta("", text)
    if text:
        value = str(text)
        call_renderer(
            renderer,
            "text_delta",
            value,
            content_blocks=normalized_content_blocks(message, value),
            **(identity or {}),
        )


async def _consume_streamed_text(
    value: Any,
    renderer: Any,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    """Render streamed assistant text."""
    source_text = ""
    async for delta in _text_deltas(value):
        text = normalize_response_delta(source_text, delta)
        if text:
            source_text += text
            call_renderer(
                renderer,
                "text_delta",
                text,
                content_blocks=({"type": "text", "text": text},),
                **(identity or {}),
            )


def message_identity(
    message: Any,
    invocation_metadata: MessageInvocationMetadata | None,
) -> dict[str, Any]:
    """Preserve native message identity, metadata, and graph provenance."""
    message_id = getattr(message, "message_id", None) or getattr(message, "id", None)
    namespace = getattr(message, "namespace", None) or getattr(message, "path", None) or ()
    metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    if invocation_metadata is not None:
        metadata = {**invocation_metadata.for_message(message), **metadata}
    return {
        "message_id": str(message_id or ""),
        "namespace": tuple(str(item) for item in namespace) if isinstance(namespace, (list, tuple)) else (),
        "metadata": metadata,
    }


def merged_event_identity(base: dict[str, Any] | None, event: Any) -> dict[str, Any]:
    """Prefer identity/provenance attached to an ordered protocol event."""
    identity = dict(base or {})
    if not isinstance(event, dict):
        return identity
    message_id = event.get("message_id") or event.get("id")
    if message_id:
        identity["message_id"] = str(message_id)
    namespace = event.get("namespace")
    if isinstance(namespace, (list, tuple)):
        identity["namespace"] = tuple(str(item) for item in namespace)
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        identity["metadata"] = {**identity.get("metadata", {}), **metadata}
    return identity


def normalized_content_blocks(message: Any, fallback_text: str) -> tuple[Any, ...]:
    """Return LangChain normalized content blocks without a MIRA schema."""
    blocks = getattr(message, "content_blocks", None)
    if callable(blocks):
        blocks = blocks()
    if isinstance(blocks, list | tuple):
        return tuple(blocks)
    content = getattr(message, "content", None)
    if isinstance(content, list | tuple):
        return tuple(content)
    return ({"type": "text", "text": fallback_text},)


async def _consume_compaction_message(message: Any, renderer: Any) -> None:
    """Drain a live compaction message without recording reasoning or text."""
    call_renderer(renderer, "compaction_started")
    try:
        if is_raw_message_stream(message):
            async for _ in message:
                pass
        else:
            reasoning = getattr(message, "reasoning", None)
            if reasoning is not None:
                async for _ in _text_deltas(reasoning):
                    pass
            await _drain_message_text(message)
    finally:
        call_renderer(renderer, "compaction_finished")


async def _drain_message_text(message: Any) -> None:
    """Consume text projections so provider stream tasks can complete."""
    msg_text = getattr(message, "text", None)
    if msg_text is None or callable(msg_text):
        return
    async for _ in _text_deltas(msg_text):
        pass


async def _drain_message(message: Any) -> None:
    """Drain a hidden synthetic message without rendering its content."""
    if is_raw_message_stream(message):
        async for _ in message:
            pass
        return
    reasoning = getattr(message, "reasoning", None)
    if reasoning is not None:
        async for _ in _text_deltas(reasoning):
            pass
    await _drain_message_text(message)


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


async def _finalized_tool_calls(
    message: Any,
    renderer: Any,
    result: Any | None = None,
    *,
    identity: dict[str, Any] | None = None,
) -> list[Any]:
    """Return the finalized tool-call list for a streamed message."""
    calls = getattr(message, "tool_calls", None)
    if calls is None:
        return []

    tool_drafts = ToolCallDrafts(renderer, result, identity=identity)
    if hasattr(calls, "__aiter__"):
        async for chunk in calls:
            tool_drafts.push(chunk)

    finalized = calls.get() if hasattr(calls, "get") else (calls or [])
    if hasattr(finalized, "__await__"):
        finalized = await finalized

    return list(finalized or [])


def render_tool_calls(
    call_list: list[Any],
    renderer: Any,
    result: Any | None,
    *,
    render_normal_tools: bool = True,
    identity: dict[str, Any] | None = None,
) -> None:
    """Render fallback finalized calls from message projections."""
    normalized = [normalized_call(call) for call in call_list]
    task_calls = [call for call in call_list if tool_call_name(call) == "task"]
    if render_normal_tools and task_calls:
        call_renderer(renderer, "delegation_started", task_calls, **(identity or {}))

    for call in normalized:
        name = str(call["name"])
        call_id = str(call.get("id") or "")
        if name == "task":
            if result is not None and render_normal_tools:
                result.record_tool_call(name, call_id)
            continue

        if not render_normal_tools and name not in CONTROL_TOOLS:
            continue

        if result is not None and not result.record_tool_call(name, call_id):
            continue

        call_renderer(
            renderer,
            "tool_call",
            name,
            call.get("args", {}),
            call_id=call_id,
            **(identity or {}),
        )
