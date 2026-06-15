"""Message-stream consumption for reasoning, text, and tool-call events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from runtime.output_events import (
    could_be_compaction_summary_start,
    is_summarization_metadata_message,
    normalize_response_delta,
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
COMPACTION_REASONING_HINTS = (
    "context extraction assistant",
    "extract the highest quality/most relevant context",
    "conversation history to replace",
    "conversation history will be replaced",
    "conversation history",
    "due to nearing token limits",
    "due to token limits",
    "compact",
    "compaction",
    "summarization",
    "summarize",
    "session intent",
    "artifacts",
    "next steps",
    "output format",
)
COMPACTION_REASONING_START = "thinking process"


async def consume_messages(messages: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume streamed model messages and render reasoning, text, and tools.

    Provider integrations expose message fields in different shapes. This
    module normalizes those fields into renderer calls while preserving the
    event order reported by the stream.
    """
    async for message in messages:
        _call_renderer(renderer, "waiting_started")
        if is_raw_message_stream(message):
            await _consume_ordered_message_stream(message, renderer)
        else:
            compacting = await _consume_reasoning(message, renderer)
            await _consume_text(message, renderer, allow_compaction_summary=compacting)

        call_list = await _finalized_tool_calls(message)
        if call_list:
            _render_tool_calls(call_list, renderer, result)

        if result is not None:
            usage = usage_from_message(message)
            if has_usage(usage):
                result.add_stream_usage(usage)


async def _consume_reasoning(message: Any, renderer: Any) -> bool:
    """Render reasoning deltas from a streamed message."""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return False

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
    return compacting


def is_raw_message_stream(message: Any) -> bool:
    """Return whether a message can be consumed as ordered protocol events."""
    return hasattr(message, "__aiter__") and all(
        hasattr(message, name) for name in ("text", "reasoning", "tool_calls")
    )


async def _consume_ordered_message_stream(message: Any, renderer: Any) -> None:
    """Render raw ChatModelStream events in provider order."""
    reasoning_filter = ReasoningFilter(renderer)
    text_filter = TextFilter(renderer, allow_compaction_summary=lambda: reasoning_filter.compacting)

    async for event in message:
        delta = event_delta(event)
        delta_type = str(delta.get("type") or "")
        if delta_type == "reasoning-delta":
            reasoning_filter.push(str(delta.get("reasoning") or delta.get("text") or ""))
        elif delta_type == "text-delta":
            text_filter.push(str(delta.get("text") or ""))

    reasoning_filter.finish()
    text_filter.finish()


class ReasoningFilter:
    """Render reasoning while suppressing DeepAgents compaction internals."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self.pending = ""
        self.probing = True
        self.compacting = False

    def push(self, delta: str) -> None:
        if not delta or self.compacting:
            return
        if not self.probing:
            self.renderer.reasoning_delta(delta)
            return

        self.pending += delta
        if is_compaction_reasoning(self.pending):
            self.compacting = True
            self.pending = ""
            _call_renderer(self.renderer, "compaction_started")
            return

        if not could_be_compaction_reasoning_start(self.pending):
            self.renderer.reasoning_delta(self.pending)
            self.pending = ""
            self.probing = False

    def finish(self) -> None:
        if self.compacting:
            _call_renderer(self.renderer, "compaction_finished")
        elif self.pending:
            self.renderer.reasoning_delta(self.pending)


class TextFilter:
    """Render assistant text while stripping a leading compaction summary."""

    def __init__(self, renderer: Any, allow_compaction_summary: Callable[[], bool]) -> None:
        self.renderer = renderer
        self.allow_compaction_summary = allow_compaction_summary
        self.pending = ""
        self.probing = True
        self.compacting = False
        self.has_output = False

    def push(self, delta: str) -> None:
        delta = normalize_response_delta("visible" if self.has_output else self.pending, delta)
        if not delta:
            return
        if not self.probing:
            self._emit(delta)
            return

        self.pending += delta
        visible, had_summary = strip_compaction_summary_prefix(self.pending)
        if had_summary:
            if not self.compacting and not self.allow_compaction_summary():
                self.compacting = True
                _call_renderer(self.renderer, "compaction_started")
            if visible:
                if self.compacting:
                    _call_renderer(self.renderer, "compaction_finished")
                    self.compacting = False
                self._emit(visible)
                self.pending = ""
                self.probing = False
            return

        if not could_be_compaction_summary_start(self.pending):
            self._emit(self.pending)
            self.pending = ""
            self.probing = False

    def finish(self) -> None:
        if self.compacting:
            _call_renderer(self.renderer, "compaction_finished")
            self.compacting = False
        if self.probing and self.pending and not text_has_compaction_summary_shape(self.pending):
            self._emit(self.pending)

    def _emit(self, text: str) -> None:
        self.renderer.text_delta(text)
        if text:
            self.has_output = True


def event_delta(event: Any) -> dict[str, Any]:
    """Extract a content delta from LangChain protocol event shapes."""
    if not isinstance(event, dict):
        return {}
    delta = event.get("delta")
    if isinstance(delta, dict):
        return delta
    block = event.get("content_block")
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "text":
            return {"type": "text-delta", "text": block.get("text", "")}
        if block_type == "reasoning":
            return {"type": "reasoning-delta", "reasoning": block.get("reasoning", "")}
    return {}


def is_compaction_reasoning(text: str) -> bool:
    """Return whether streamed reasoning belongs to DeepAgents compaction."""
    lowered = text.lower()
    if all(marker in lowered for marker in COMPACTION_REASONING_MARKERS):
        return True
    if "context extraction assistant" in lowered and (
        "conversation history" in lowered or "replace it" in lowered or "token limit" in lowered
    ):
        return True
    if "primary objective" in lowered and "conversation history" in lowered and "replace" in lowered:
        return True
    if "output format" in lowered and "session intent" in lowered and "next steps" in lowered:
        return True
    if "session intent" in lowered and "summary" in lowered and "artifacts" in lowered and "next steps" in lowered:
        return True
    if "compact" in lowered and "conversation" in lowered and ("summary" in lowered or "token" in lowered):
        return True
    if "summarization" in lowered and "conversation" in lowered:
        return True
    return False


def should_flush_reasoning_probe(text: str) -> bool:
    """Return whether buffered reasoning is unlikely to be compaction metadata."""
    lowered = text.lower()
    if could_be_compaction_reasoning_start(text):
        return False
    if len(text) >= 1200:
        return True
    if "\n\n" in text and not any(marker in lowered for marker in COMPACTION_REASONING_HINTS):
        return True
    return False


def could_be_compaction_reasoning_start(text: str) -> bool:
    """Return whether reasoning may still be DeepAgents compaction setup."""
    stripped = text.lstrip().lower()
    if not stripped:
        return True
    if COMPACTION_REASONING_START.startswith(stripped) or stripped.startswith(COMPACTION_REASONING_START):
        return True
    if any(hint in stripped for hint in COMPACTION_REASONING_HINTS):
        return True
    return False


def _call_renderer(renderer: Any, method: str) -> None:
    """Call an optional renderer method."""
    callback: Callable[[], None] | None = getattr(renderer, method, None)
    if callback is not None:
        callback()


async def _consume_text(message: Any, renderer: Any, *, allow_compaction_summary: bool = False) -> None:
    """Render assistant text deltas from a streamed message."""
    if is_summarization_metadata_message(message):
        return

    msg_text = getattr(message, "text", None)
    if msg_text is None:
        return

    if hasattr(msg_text, "__aiter__"):
        await _consume_streamed_text(msg_text, renderer, allow_compaction_summary=allow_compaction_summary)
        return

    text = await msg_text if hasattr(msg_text, "__await__") else msg_text
    text = normalize_response_delta("", text)
    visible, had_summary = strip_compaction_summary_prefix(str(text or ""))
    if had_summary:
        if not allow_compaction_summary:
            _call_renderer(renderer, "compaction_started")
            _call_renderer(renderer, "compaction_finished")
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
            result.record_tool_call(str(name), _call_id(call))

        if name != "task":
            renderer.tool_call(str(name), _call_args(call), call_id=_call_id(call))


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


def _call_id(call: Any) -> str:
    """Extract a tool-call id from a dict or object."""
    if isinstance(call, dict):
        return str(call.get("id") or "")

    return str(getattr(call, "id", "") or "")
