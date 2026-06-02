"""Token usage extraction for LangChain and provider message shapes."""

from __future__ import annotations

from typing import Any


def empty_usage() -> dict[str, Any]:
    """Return an empty normalized usage object."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "context_tokens": 0,
        "source": "unknown",
    }


def has_usage(usage: dict[str, Any]) -> bool:
    """Return whether any meaningful token count is present."""
    return any(positive_int(usage.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens"))


def merge_usage(*items: dict[str, Any]) -> dict[str, Any]:
    """Merge per-call usage into one turn summary."""
    merged = empty_usage()
    for item in items:
        if not isinstance(item, dict):
            continue
        merged["input_tokens"] += positive_int(item.get("input_tokens"))
        merged["output_tokens"] += positive_int(item.get("output_tokens"))
        merged["total_tokens"] += positive_int(item.get("total_tokens"))
        merged["context_tokens"] = max(
            merged["context_tokens"],
            positive_int(item.get("context_tokens")),
            positive_int(item.get("input_tokens")),
        )
        if merged["source"] == "unknown" and item.get("source"):
            merged["source"] = str(item["source"])

    if not merged["total_tokens"]:
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged


def usage_from_message(message: Any) -> dict[str, Any]:
    """Extract usage metadata from one LangChain-like message."""
    direct = usage_from_mapping(field(message, "usage_metadata"), "usage_metadata")
    if has_usage(direct):
        return direct

    metadata = field(message, "response_metadata")
    if isinstance(metadata, dict):
        for key in ("token_usage", "usage"):
            usage = usage_from_mapping(metadata.get(key), f"response_metadata.{key}")
            if has_usage(usage):
                return usage
        usage = usage_from_mapping(metadata, "response_metadata")
        if has_usage(usage):
            return usage

    kwargs = field(message, "additional_kwargs")
    if isinstance(kwargs, dict):
        for key in ("usage", "token_usage"):
            usage = usage_from_mapping(kwargs.get(key), f"additional_kwargs.{key}")
            if has_usage(usage):
                return usage

    if isinstance(message, dict):
        for key, source in (
            ("usage_metadata", "usage_metadata"),
            ("response_metadata", "response_metadata"),
            ("usage", "usage"),
            ("token_usage", "token_usage"),
        ):
            usage = usage_from_mapping(message.get(key), source)
            if has_usage(usage):
                return usage

    return empty_usage()


def usage_from_output(output: Any) -> dict[str, Any]:
    """Extract usage from a final DeepAgents output payload."""
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            return merge_usage(*(usage_from_message(message) for message in messages))

    return usage_from_message(output)


def usage_from_mapping(value: Any, source: str) -> dict[str, Any]:
    """Normalize provider token keys from a mapping."""
    if not isinstance(value, dict):
        return empty_usage()

    input_tokens = first_int(
        value,
        "input_tokens",
        "prompt_tokens",
        "input",
        "tokens_in",
        "prompt_token_count",
    )
    output_tokens = first_int(
        value,
        "output_tokens",
        "completion_tokens",
        "output",
        "tokens_out",
        "completion_token_count",
    )
    total_tokens = first_int(value, "total_tokens", "total_token_count")
    if total_tokens and input_tokens and not output_tokens:
        output_tokens = max(0, total_tokens - input_tokens)
    if total_tokens and output_tokens and not input_tokens:
        input_tokens = max(0, total_tokens - output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or input_tokens + output_tokens,
        "context_tokens": input_tokens,
        "source": source if input_tokens or output_tokens or total_tokens else "unknown",
    }


def field(value: Any, name: str) -> Any:
    """Return a dict key or object attribute."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def first_int(value: dict[str, Any], *keys: str) -> int:
    """Return the first positive integer found under the given keys."""
    for key in keys:
        parsed = positive_int(value.get(key))
        if parsed:
            return parsed
    return 0


def positive_int(value: Any) -> int:
    """Return a non-negative integer from loose provider metadata."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0
