"""Token usage extraction for LangChain and provider message shapes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def empty_usage() -> dict[str, Any]:
    """Return an empty normalized usage object."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "context_tokens": 0,
        "context_floor_tokens": 0,
        "context_source": "unknown",
        "source": "unknown",
    }


def has_usage(usage: dict[str, Any]) -> bool:
    """Return whether any meaningful token count is present."""
    return any(positive_int(usage.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens"))


def has_context_usage(usage: dict[str, Any]) -> bool:
    """Return whether a context token count is present."""
    return positive_int(usage.get("context_tokens")) > 0 or positive_int(usage.get("context_floor_tokens")) > 0


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
            context_tokens_from_counts(item),
            positive_int(item.get("context_floor_tokens")),
        )
        merged["context_floor_tokens"] = max(
            merged["context_floor_tokens"],
            positive_int(item.get("context_floor_tokens")),
        )
        if item_context_source(item) != "unknown":
            merged["context_source"] = item_context_source(item)
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
        for key in ("token_usage", "usage", "stats"):
            usage = usage_from_mapping(metadata.get(key), f"response_metadata.{key}")
            if has_usage(usage):
                return usage
        usage = usage_from_mapping(metadata, "response_metadata")
        if has_usage(usage):
            return usage

    kwargs = field(message, "additional_kwargs")
    if isinstance(kwargs, dict):
        for key in ("usage", "token_usage", "stats"):
            usage = usage_from_mapping(kwargs.get(key), f"additional_kwargs.{key}")
            if has_usage(usage):
                return usage

    if isinstance(message, dict):
        for key, source in (
            ("usage_metadata", "usage_metadata"),
            ("response_metadata", "response_metadata"),
            ("usage", "usage"),
            ("token_usage", "token_usage"),
            ("stats", "stats"),
        ):
            usage = usage_from_mapping(message.get(key), source)
            if has_usage(usage):
                return usage

    return empty_usage()


TokenCounter = Callable[[str], int]


def usage_from_output(output: Any) -> dict[str, Any]:
    """Extract usage from a final DeepAgents output payload."""
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                usage = usage_from_message(message)
                if has_usage(usage):
                    return usage
            return empty_usage()

    return usage_from_message(output)


def context_from_output(output: Any, token_counter: TokenCounter | None) -> dict[str, Any]:
    """Count the current message stack with a provider tokenizer."""
    if token_counter is None:
        return empty_usage()

    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            return context_from_message_texts(messages, token_counter)

    text = message_text(output).strip()
    context_tokens = count_text_tokens(token_counter, text)
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "context_tokens": context_tokens,
        "context_floor_tokens": 0,
        "context_source": "langchain_approx.count_tokens" if context_tokens else "unknown",
        "source": "langchain_approx.count_tokens" if context_tokens else "unknown",
    }


def usage_from_mapping(value: Any, source: str) -> dict[str, Any]:
    """Normalize provider token keys from a mapping."""
    value = object_mapping(value)
    if not isinstance(value, dict):
        return empty_usage()

    input_tokens = first_int(
        value,
        "input_tokens",
        "prompt_tokens",
        "prompt_tokens_count",
        "promptTokensCount",
        "input",
        "tokens_in",
        "prompt_token_count",
    )
    output_tokens = first_int(
        value,
        "output_tokens",
        "completion_tokens",
        "completion_tokens_count",
        "completionTokensCount",
        "predicted_tokens_count",
        "predictedTokensCount",
        "output",
        "tokens_out",
        "completion_token_count",
    )
    total_tokens, total_key = first_int_with_key(
        value,
        "total_tokens",
        "total_token_count",
        "total_tokens_count",
        "totalTokensCount",
        "n_tokens",
        "nTokens",
    )
    if total_tokens and input_tokens and not output_tokens:
        output_tokens = max(0, total_tokens - input_tokens)
    if total_tokens and output_tokens and not input_tokens:
        input_tokens = max(0, total_tokens - output_tokens)

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or input_tokens + output_tokens,
        "context_tokens": context_tokens_from_counts(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        ),
        "context_floor_tokens": 0,
        "context_source": context_source_for_mapping(source, total_key, input_tokens, output_tokens, total_tokens),
        "source": source if input_tokens or output_tokens or total_tokens else "unknown",
    }
    return usage


def context_tokens_from_counts(value: dict[str, Any]) -> int:
    """Return current context occupancy from normalized provider counts."""
    total_tokens = positive_int(value.get("total_tokens"))
    if total_tokens:
        return total_tokens

    input_tokens = positive_int(value.get("input_tokens"))
    output_tokens = positive_int(value.get("output_tokens"))
    if input_tokens and output_tokens:
        return input_tokens + output_tokens
    return input_tokens


def context_from_message_texts(messages: list[Any], token_counter: TokenCounter | None) -> dict[str, Any]:
    """Estimate current context from message text without changing In/Out totals."""
    if token_counter is None or not messages:
        return empty_usage()

    context_text = "\n".join(role_prefixed_text(message) for message in messages).strip()
    context_tokens = count_text_tokens(token_counter, context_text)
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "context_tokens": context_tokens,
        "context_floor_tokens": 0,
        "context_source": "langchain_approx.count_tokens" if context_tokens else "unknown",
        "source": "langchain_approx.count_tokens" if context_tokens else "unknown",
    }


def role_prefixed_text(message: Any) -> str:
    """Return text with a lightweight role prefix for fallback token counting."""
    text = message_text(message).strip()
    if not text:
        return ""
    role = message_role(message) or "message"
    return f"{role}: {text}"


def message_text(message: Any) -> str:
    """Extract plain text from common LangChain message content shapes."""
    content = field(message, "content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    text = field(message, "text")
    if text is not None and not callable(text):
        return str(text)

    return ""


def message_role(message: Any) -> str:
    """Extract a normalized message role from dicts and LangChain messages."""
    role = field(message, "role") or field(message, "type")
    if role:
        return str(role).lower()

    class_name = message.__class__.__name__.lower()
    if "human" in class_name:
        return "human"
    if "ai" in class_name or "assistant" in class_name:
        return "ai"
    if "system" in class_name:
        return "system"
    if "tool" in class_name:
        return "tool"
    return ""


def count_text_tokens(token_counter: TokenCounter, text: str) -> int:
    """Count text tokens while keeping fallback counting best-effort."""
    if not text:
        return 0
    try:
        return positive_int(token_counter(text))
    except Exception:
        return 0


def object_mapping(value: Any) -> dict[str, Any] | None:
    """Return a loose mapping for dicts, provider structs, and metadata objects."""
    if isinstance(value, dict):
        return value

    if hasattr(value, "to_dict"):
        try:
            mapped = value.to_dict()
        except Exception:
            mapped = None
        if isinstance(mapped, dict):
            return mapped

    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_count",
        "predicted_tokens_count",
        "total_tokens_count",
        "totalTokensCount",
        "n_tokens",
        "nTokens",
    )
    mapped = {key: getattr(value, key) for key in keys if hasattr(value, key)}
    return mapped or None


def field(value: Any, name: str) -> Any:
    """Return a dict key or object attribute."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def first_int(value: dict[str, Any], *keys: str) -> int:
    """Return the first positive integer found under the given keys."""
    parsed, _key = first_int_with_key(value, *keys)
    return parsed


def first_int_with_key(value: dict[str, Any], *keys: str) -> tuple[int, str]:
    """Return the first positive integer and key found in a mapping."""
    for key in keys:
        parsed = positive_int(value.get(key))
        if parsed:
            return parsed, key
    return 0, ""


def context_source_for_mapping(
    source: str,
    total_key: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> str:
    """Return the source for current-context occupancy from provider metadata."""
    if total_key in {"total_token_count", "total_tokens_count", "totalTokensCount", "n_tokens", "nTokens"}:
        return source
    if input_tokens or output_tokens:
        return "provider.input_output_tokens"
    return source if total_tokens else "unknown"


def item_context_source(usage: dict[str, Any]) -> str:
    """Return context-specific source, falling back to usage source."""
    return str(usage.get("context_source") or usage.get("source") or "unknown")


def select_context_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Select current-context occupancy by source precedence."""
    selected = dict(usage)
    input_tokens = positive_int(usage.get("input_tokens"))
    output_tokens = positive_int(usage.get("output_tokens"))
    provider_pair = input_tokens + output_tokens
    context_tokens = positive_int(usage.get("context_tokens"))
    request_estimate = positive_int(usage.get("context_floor_tokens"))
    context_source = item_context_source(usage)

    if is_trusted_full_context_source(context_source) and context_tokens:
        selected["context_tokens"] = context_tokens
        selected["context_source"] = context_source
    elif request_estimate:
        selected["context_tokens"] = request_estimate
        selected["context_source"] = "request_estimate.count_tokens"
    elif provider_pair:
        selected["context_tokens"] = provider_pair
        selected["context_source"] = "provider.input_output_tokens"
    elif context_tokens:
        selected["context_tokens"] = context_tokens
        selected["context_source"] = context_source
    else:
        selected["context_tokens"] = 0
        selected["context_source"] = "unknown"
    return selected


def is_trusted_full_context_source(source: str) -> bool:
    """Return whether a source is known to report full context occupancy."""
    return source not in {
        "",
        "unknown",
        "langchain_approx.count_tokens",
        "provider.input_output_tokens",
        "request_estimate.count_tokens",
        "request_floor.count_tokens",
        "usage_metadata",
    }


def positive_int(value: Any) -> int:
    """Return a non-negative integer from loose provider metadata."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0
