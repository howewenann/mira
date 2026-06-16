"""Low-level protocol helpers for provider-specific stream chunks."""

from __future__ import annotations

from typing import Any

TOOL_CALL_DELTA_TYPES = {
    "tool_call",
    "tool_call_chunk",
    "tool-call",
    "tool-call-chunk",
    "tool-call-delta",
    "tool_call_delta",
    "function_call",
    "function_call_chunk",
    "function_call_delta",
    "function-call",
    "function-call-chunk",
    "function-call-delta",
}


def is_raw_message_stream(message: Any) -> bool:
    """Return whether a message can be consumed as ordered protocol events."""
    return hasattr(message, "__aiter__") and all(
        hasattr(message, name) for name in ("text", "reasoning", "tool_calls")
    )


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
        if is_tool_call_delta(str(block_type or "")):
            return {
                "type": "tool-call-delta",
                "id": block.get("id") or block.get("call_id") or block.get("tool_call_id"),
                "name": block.get("name"),
                "arguments": block.get("arguments") if "arguments" in block else block.get("args"),
                "index": block.get("index"),
            }
    return {}


def is_tool_call_delta(delta_type: str) -> bool:
    """Return whether a raw message delta represents tool-call JSON streaming."""
    return delta_type in TOOL_CALL_DELTA_TYPES
