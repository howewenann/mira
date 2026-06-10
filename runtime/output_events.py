"""Final-output and interrupt extraction helpers for agent streams."""

from __future__ import annotations

import re
from typing import Any

COMPACTION_SUMMARY_HEADINGS = (
    "## session intent",
    "## summary",
    "## artifacts",
    "## next steps",
)


async def capture_output(output_stream: Any, output: dict[str, Any]) -> None:
    """Store the last final-output value from DeepAgents."""
    if hasattr(output_stream, "__aiter__"):
        async for item in output_stream:
            output["value"] = item
        return

    if hasattr(output_stream, "__await__"):
        output["value"] = await output_stream
        return

    output["value"] = output_stream


def final_text(output: Any) -> str:
    """Extract final assistant text from a DeepAgents output payload."""
    if not isinstance(output, dict):
        return ""

    messages = output.get("messages") or []
    if not messages:
        return ""

    for message in reversed(messages):
        text = visible_message_text(message)
        if text:
            return text
    return ""


def is_compaction_summary_message(message: Any) -> bool:
    """Return whether a message is an internal DeepAgents compaction summary."""
    if is_summarization_metadata_message(message):
        return True

    text = message_text(message)
    visible, had_summary = strip_compaction_summary_prefix(text)
    return had_summary and not visible.strip()


def is_summarization_metadata_message(message: Any) -> bool:
    """Return whether message metadata marks a DeepAgents summary."""
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization"


def visible_message_text(message: Any) -> str:
    """Return message text with any leading compaction summary removed."""
    if is_summarization_metadata_message(message):
        return ""
    text = message_text(message)
    visible, _ = strip_compaction_summary_prefix(text)
    return visible


def strip_compaction_summary_prefix(text: str) -> tuple[str, bool]:
    """Remove a leading structured compaction summary from text."""
    if not text:
        return "", False

    lowered = text.lower()
    first_heading = lowered.find(COMPACTION_SUMMARY_HEADINGS[0])
    if first_heading < 0 or first_heading > 80:
        return text, False

    if not headings_in_order(lowered):
        return text, False

    match = re.search(r"(?im)^##\s*next steps\b.*$", text)
    if match is None:
        return "", True

    after_heading = text[match.end() :]
    paragraph_break = after_heading.find("\n\n")
    if paragraph_break < 0:
        return "", True

    return after_heading[paragraph_break:].lstrip(), True


def headings_in_order(text: str) -> bool:
    """Return whether compaction headings appear in the expected order."""
    position = -1
    for heading in COMPACTION_SUMMARY_HEADINGS:
        next_position = text.find(heading, position + 1)
        if next_position < 0:
            return False
        position = next_position
    return True


def text_has_compaction_summary_shape(text: str) -> bool:
    """Return whether text starts with the structured compaction summary shape."""
    text = text.strip().lower()
    if not text:
        return False
    return headings_in_order(text) and text.find(COMPACTION_SUMMARY_HEADINGS[0]) <= 80


def message_text(message: Any) -> str:
    """Extract plain text from common LangChain message content shapes."""
    text = field(message, "text")
    if text is not None:
        return str(text)

    content = field(message, "content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    return ""


def field(value: Any, name: str) -> Any:
    """Return a dict key or object attribute."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def find_interrupts(value: Any) -> list[Any]:
    """Find interrupts stored on an output value or output dictionary."""
    if value is None:
        return []

    if isinstance(value, dict):
        return value.get("__interrupt__", []) or value.get("interrupts", [])

    interrupts = getattr(value, "__interrupt__", None) or getattr(value, "interrupts", None)
    return interrupts or []


async def collect_interrupts(stream: Any, output_value: Any) -> list[Any]:
    """Prefer stream interrupts, then fall back to interrupts in final output."""
    interrupts = await stream_interrupts(stream)
    if interrupts:
        return interrupts

    return find_interrupts(output_value)


async def stream_interrupts(stream: Any) -> list[Any]:
    """Return interrupts from a DeepAgents stream object if it exposes them."""
    interrupts = getattr(stream, "interrupts", None)

    if callable(interrupts):
        interrupts = interrupts()

    if hasattr(interrupts, "__await__"):
        interrupts = await interrupts

    return interrupts or []
