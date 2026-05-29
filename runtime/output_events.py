"""Final-output and interrupt extraction helpers for agent streams."""

from __future__ import annotations

from typing import Any


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

    return message_text(messages[-1])


def message_text(message: Any) -> str:
    """Extract plain text from common LangChain message content shapes."""
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
