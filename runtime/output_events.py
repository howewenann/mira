"""Final-output and interrupt extraction helpers for agent streams."""

from __future__ import annotations

import re
from typing import Any

from runtime.usage import field

COMPACTION_SUMMARY_HEADINGS = (
    "session intent",
    "summary",
    "artifacts",
    "next steps",
)
COMPACTION_HEADING_RE_TEMPLATE = r"(?im)^\s{{0,3}}(?:#{{1,6}}\s*)?(?:\*\*)?{heading}(?:\*\*)?\s*:?\b"
LEADING_REPLY_GAP_RE = re.compile(r"^\s*\n+\s*")


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
    return is_summarization_metadata_message(message) or text_has_compaction_summary_shape(message_text(message))


def is_summarization_metadata_message(message: Any) -> bool:
    """Return whether message metadata marks a DeepAgents summary."""
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization"


def visible_message_text(message: Any) -> str:
    """Return visible assistant text, hiding internal compaction summaries."""
    if is_summarization_metadata_message(message):
        return ""
    text = message_text(message)
    visible, had_summary = strip_compaction_summary_prefix(text)
    return normalize_response_delta("", visible if had_summary else text)


def normalize_response_delta(existing_text: str, delta: Any) -> str:
    """Normalize streamed assistant text, hiding blank leading gaps."""
    text = str(delta or "")
    if not text:
        return ""
    if not existing_text:
        text = LEADING_REPLY_GAP_RE.sub("", text)
        if not text.strip():
            return ""
    return text


def strip_compaction_summary_prefix(text: str) -> tuple[str, bool]:
    """Remove a leading structured compaction summary from text."""
    if not text:
        return "", False

    positions = compaction_heading_positions(text)
    first_heading = positions[0] if positions else -1
    if first_heading < 0 or first_heading > 240:
        return text, False

    if len(positions) != len(COMPACTION_SUMMARY_HEADINGS):
        return text, False

    match = compaction_heading_match(text, COMPACTION_SUMMARY_HEADINGS[-1])
    if match is None:
        return "", True

    after_heading = text[match.end() :]
    paragraph_break = after_heading.find("\n\n")
    if paragraph_break < 0:
        return "", True

    return after_heading[paragraph_break:].lstrip(), True


def headings_in_order(text: str) -> bool:
    """Return whether compaction headings appear in the expected order."""
    return len(compaction_heading_positions(text)) == len(COMPACTION_SUMMARY_HEADINGS)


def compaction_heading_positions(text: str) -> list[int]:
    """Return compaction heading positions when all headings appear in order."""
    position = -1
    positions = []
    for heading in COMPACTION_SUMMARY_HEADINGS:
        match = compaction_heading_match(text, heading, position + 1)
        if match is None:
            return []
        position = match.start()
        positions.append(position)
    return positions


def compaction_heading_match(text: str, heading: str, pos: int = 0) -> re.Match[str] | None:
    """Return the next heading match for one compaction section."""
    pattern = COMPACTION_HEADING_RE_TEMPLATE.format(heading=re.escape(heading).replace(r"\ ", r"\s+"))
    return re.compile(pattern).search(text, pos)


def text_has_compaction_summary_shape(text: str) -> bool:
    """Return whether text starts with the structured compaction summary shape."""
    text = text.strip()
    if not text:
        return False
    positions = compaction_heading_positions(text)
    return bool(positions) and positions[0] <= 240


def could_be_compaction_summary_start(text: str) -> bool:
    """Return whether streamed text may still become a compaction summary."""
    stripped = text.lstrip().lower()
    if not stripped:
        return True

    candidate = re.sub(r"^#{1,6}\s*", "", stripped).strip()
    candidate = candidate.strip("* ")
    if not candidate:
        return stripped.startswith("#")

    marker = "session intent"
    return marker.startswith(candidate) or candidate.startswith(marker)


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
