"""Final-output and interrupt extraction helpers for agent streams."""

from __future__ import annotations

import re
from collections import Counter
from inspect import Parameter, signature
from typing import Any

from agent.planning.criteria import SUCCESS_CRITERIA_SOURCE

from agent.middleware.correction import CORRECTION_SOURCE
from core.context.usage import field

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


def is_summarization_metadata_message(message: Any) -> bool:
    """Return whether message metadata marks a DeepAgents summary."""
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization"


def is_success_criteria_metadata_message(message: Any) -> bool:
    """Return whether a finalized message is internal criteria generation."""
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == SUCCESS_CRITERIA_SOURCE


def visible_message_text(message: Any) -> str:
    """Return visible assistant text, hiding internal compaction summaries."""
    if is_tool_message(message):
        return ""
    if is_summarization_metadata_message(message):
        return ""
    if is_correction_metadata_message(message):
        return ""
    return normalize_response_delta("", message_text(message))


def is_correction_metadata_message(message: Any) -> bool:
    """Return whether a synthetic message belongs to correction feedback."""
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == CORRECTION_SOURCE


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


def message_text(message: Any) -> str:
    """Extract plain text from common LangChain message content shapes."""
    content = field(message, "content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    text = field(message, "text")
    if text is not None and not callable(text):
        return str(text)

    return ""


def output_tool_calls(output: Any) -> list[Any]:
    """Return pending tool calls found at the tail of final output messages."""
    if not isinstance(output, dict):
        return []

    messages = output.get("messages") or []
    for message in reversed(messages):
        message_calls = field(message, "tool_calls")
        if message_calls:
            return [normalized_output_tool_call(call) for call in message_calls]
        if visible_message_text(message) or field(message, "content"):
            return []
    return []


def output_tool_results(output: Any) -> list[dict[str, str]]:
    """Return tool results found in final output messages."""
    if not isinstance(output, dict):
        return []

    results = []
    for message in output.get("messages") or []:
        if not is_tool_message(message):
            continue
        output_text = message_text(message)
        if not output_text:
            continue
        results.append(
            {
                "name": str(field(message, "name") or "tool"),
                "output": output_text,
                "call_id": str(field(message, "tool_call_id") or field(message, "id") or ""),
                "status": str(field(message, "status") or "success"),
            }
        )
    return results


def output_tool_lifecycle(output: Any) -> list[dict[str, Any]]:
    """Return executed call/result entries in final graph-message order."""
    if not isinstance(output, dict):
        return []

    results = output_tool_results(output)
    result_ids = {item["call_id"] for item in results if item["call_id"]}
    idless_results = Counter(
        item["name"]
        for item in results
        if not item["call_id"]
    )
    lifecycle: list[dict[str, Any]] = []
    for message in output.get("messages") or []:
        for call in field(message, "tool_calls") or []:
            normalized = normalized_output_tool_call(call)
            call_id = str(
                field(normalized, "id")
                or field(normalized, "call_id")
                or field(normalized, "tool_call_id")
                or ""
            )
            name = str(field(normalized, "name") or field(normalized, "tool_name") or "tool")
            if call_id:
                if call_id not in result_ids:
                    continue
            elif idless_results[name] > 0:
                idless_results[name] -= 1
            else:
                continue
            lifecycle.append({"type": "tool_call", "call": normalized})

        if not is_tool_message(message):
            continue
        output_text = message_text(message)
        if not output_text:
            continue
        lifecycle.append(
            {
                "type": "tool_result",
                "name": str(field(message, "name") or "tool"),
                "output": output_text,
                "call_id": str(
                    field(message, "tool_call_id")
                    or field(message, "id")
                    or ""
                ),
                "status": str(field(message, "status") or "success"),
            }
        )
    return lifecycle


def is_tool_message(message: Any) -> bool:
    """Return whether a message is a LangChain tool result message."""
    return field(message, "type") == "tool" or message.__class__.__name__ == "ToolMessage"


def output_has_tool_call_repr(output: Any) -> bool:
    """Return whether final output leaked an AIMessage repr with tool calls."""
    text = final_text(output)
    return text.startswith("AIMessage(content=") and "tool_calls=[" in text


def normalized_output_tool_call(call: Any) -> Any:
    """Normalize fallback file-tool args without changing canonical streams."""
    if not isinstance(call, dict):
        return call
    if call.get("name") not in {"read_file", "write_file", "edit_file"}:
        return call

    args = call.get("args")
    if not isinstance(args, dict) or "file_path" in args or "path" not in args:
        return call

    normalized_args = dict(args)
    normalized_args["file_path"] = normalized_args.pop("path")
    return {**call, "args": normalized_args}


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


def call_renderer(renderer: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """Invoke one optional internal stream sink callback."""
    callback = getattr(renderer, method, None)
    if callback is None:
        return False
    callback(*args, **_supported_kwargs(callback, kwargs))
    return True


def _supported_kwargs(callback: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass optional native metadata supported by the stream sink."""
    if not kwargs:
        return kwargs
    try:
        parameters = signature(callback).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
        and parameters[key].kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
