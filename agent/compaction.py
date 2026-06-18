"""MIRA wrappers for DeepAgents compaction middleware."""

from __future__ import annotations

from functools import wraps
from typing import Any

from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from runtime.compaction_state import compaction_scope


def create_mira_summarization_tool_middleware(model: Any, backend: Any) -> Any:
    """Create DeepAgents summarization middleware with an explicit live marker."""
    middleware = create_summarization_tool_middleware(model=model, backend=backend)
    mark_summarization_engine(getattr(middleware, "_summarization", None))
    return middleware


def mark_summarization_engine(summarization: Any) -> None:
    """Wrap DeepAgents summary generation methods so MIRA can filter their stream."""
    if summarization is None or getattr(summarization, "_mira_compaction_marked", False):
        return

    create_summary = getattr(summarization, "_create_summary", None)
    if callable(create_summary):

        @wraps(create_summary)
        def wrapped_create_summary(*args: Any, **kwargs: Any) -> Any:
            with compaction_scope():
                return create_summary(*args, **kwargs)

        setattr(summarization, "_create_summary", wrapped_create_summary)

    acreate_summary = getattr(summarization, "_acreate_summary", None)
    if callable(acreate_summary):

        @wraps(acreate_summary)
        async def wrapped_acreate_summary(*args: Any, **kwargs: Any) -> Any:
            with compaction_scope():
                return await acreate_summary(*args, **kwargs)

        setattr(summarization, "_acreate_summary", wrapped_acreate_summary)

    offload = getattr(summarization, "_offload_to_backend", None)
    if callable(offload):

        @wraps(offload)
        def wrapped_offload(backend: Any, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            return offload(backend, sanitize_messages_for_archive(messages), *args, **kwargs)

        setattr(summarization, "_offload_to_backend", wrapped_offload)

    aoffload = getattr(summarization, "_aoffload_to_backend", None)
    if callable(aoffload):

        @wraps(aoffload)
        async def wrapped_aoffload(backend: Any, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            return await aoffload(backend, sanitize_messages_for_archive(messages), *args, **kwargs)

        setattr(summarization, "_aoffload_to_backend", wrapped_aoffload)

    build_messages = getattr(summarization, "_build_new_messages_with_path", None)
    if callable(build_messages):

        @wraps(build_messages)
        def wrapped_build_messages(*args: Any, **kwargs: Any) -> list[Any]:
            return normalize_summary_messages(build_messages(*args, **kwargs))

        setattr(summarization, "_build_new_messages_with_path", wrapped_build_messages)

    setattr(summarization, "_mira_compaction_marked", True)


def sanitize_messages_for_archive(messages: list[Any]) -> list[Any]:
    """Return visible-only messages for DeepAgents conversation-history archives."""
    sanitized = []
    for message in messages:
        if is_summary_message(message):
            continue
        safe = sanitize_message_for_archive(message)
        if safe is not None:
            sanitized.append(safe)
    return sanitized


def sanitize_message_for_archive(message: Any) -> Any | None:
    """Convert one LangChain message to a reasoning-free archive message."""
    text = visible_text(message)
    role = message_role(message)
    if role == "human":
        return HumanMessage(content=text)
    if role == "system":
        return SystemMessage(content=text)
    if role == "tool":
        return ToolMessage(content=text, tool_call_id=str(getattr(message, "tool_call_id", "") or "tool"))
    if role == "ai":
        tool_facts = sanitized_tool_facts(message)
        content = "\n".join(part for part in [text, tool_facts] if part).strip()
        return AIMessage(content=content)
    if text:
        return HumanMessage(content=text)
    return None


def normalize_summary_messages(messages: list[Any]) -> list[Any]:
    """Remove provider-hostile metadata from summary messages before replay."""
    normalized = []
    for message in messages:
        if is_summary_message(message):
            normalized.append(HumanMessage(content=visible_text(message)))
        else:
            normalized.append(message)
    return normalized


def visible_text(message: Any) -> str:
    """Extract visible text while dropping reasoning content blocks."""
    content = field(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    text = field(message, "text")
    return str(text or "").strip() if text is not None and not callable(text) else ""


def sanitized_tool_facts(message: Any) -> str:
    """Return compact tool-call facts without provider internals."""
    calls = field(message, "tool_calls")
    if not isinstance(calls, list):
        return ""
    lines = []
    for call in calls:
        name = field(call, "name") or (call.get("name") if isinstance(call, dict) else "")
        args = field(call, "args") or (call.get("args") if isinstance(call, dict) else {})
        if name:
            lines.append(f"Tool call: {name}({args})")
    return "\n".join(lines)


def is_summary_message(message: Any) -> bool:
    kwargs = field(message, "additional_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization"


def message_role(message: Any) -> str:
    role = field(message, "role") or field(message, "type")
    if role:
        role = str(role).lower()
        return {"user": "human", "assistant": "ai"}.get(role, role)
    name = message.__class__.__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name or "assistant" in name:
        return "ai"
    if "system" in name:
        return "system"
    if "tool" in name:
        return "tool"
    return ""


def field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


__all__ = [
    "create_mira_summarization_tool_middleware",
    "mark_summarization_engine",
    "sanitize_messages_for_archive",
]
