"""MIRA wrappers for DeepAgents compaction middleware."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, convert_to_messages

from runtime.compaction_state import compaction_scope


def create_mira_summarization_tool_middleware(model: Any, backend: Any) -> Any:
    """Create DeepAgents summarization middleware with an explicit live marker."""
    middleware = create_summarization_tool_middleware(model=model, backend=backend)
    mark_summarization_engine(getattr(middleware, "_summarization", None))
    return middleware


@dataclass(frozen=True)
class PostTurnCompactionResult:
    """Outcome of a post-turn compaction attempt."""

    compacted: bool = False
    reason: str = ""
    file_path: str = ""
    summary: str = ""


async def compact_after_turn(agent: Any, thread_id: str) -> PostTurnCompactionResult:
    """Compact older context after a completed turn without re-answering the prompt."""
    summarization = getattr(agent, "mira_summarization", None)
    if summarization is None:
        return PostTurnCompactionResult(reason="unavailable")
    if not callable(getattr(agent, "aget_state", None)) or not callable(getattr(agent, "aupdate_state", None)):
        return PostTurnCompactionResult(reason="state_unavailable")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(config)
    state = getattr(snapshot, "values", None)
    if not isinstance(state, dict):
        return PostTurnCompactionResult(reason="state_unavailable")

    try:
        messages = convert_to_messages(state.get("messages") or [])
    except Exception:
        messages = list(state.get("messages") or [])
    if not messages:
        return PostTurnCompactionResult(reason="no_messages")

    event = normalize_summarization_event(state.get("_summarization_event"))
    effective = summarization._apply_event_to_messages(messages, event)
    cutoff = int(summarization._determine_cutoff_index(effective) or 0)
    if cutoff <= 0:
        return PostTurnCompactionResult(reason="nothing_to_compact")

    to_summarize, _preserved = summarization._partition_messages(effective, cutoff)
    if not to_summarize:
        return PostTurnCompactionResult(reason="nothing_to_compact")

    original_get_thread_id = getattr(summarization, "_get_thread_id", None)
    if callable(original_get_thread_id):
        setattr(summarization, "_get_thread_id", lambda: str(thread_id))
    try:
        backend = getattr(summarization, "_backend", None)
        if callable(backend):
            return PostTurnCompactionResult(reason="backend_unavailable")
        with compaction_scope():
            file_path = await summarization._aoffload_to_backend(backend, to_summarize)
            summary = await summarization._acreate_summary(to_summarize)
    finally:
        if callable(original_get_thread_id):
            setattr(summarization, "_get_thread_id", original_get_thread_id)

    summary_message = summarization._build_new_messages_with_path(summary, file_path)[0]
    state_cutoff = summarization._compute_state_cutoff(event, cutoff)
    new_event = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_message,
        "file_path": file_path,
    }
    await agent.aupdate_state(config, {"_summarization_event": new_event})
    return PostTurnCompactionResult(
        compacted=True,
        reason="compacted",
        file_path=str(file_path or ""),
        summary=str(summary or ""),
    )


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

    apply_event = getattr(summarization, "_apply_event_to_messages", None)
    if callable(apply_event):

        @wraps(apply_event)
        def wrapped_apply_event_to_messages(messages: list[Any], event: Any) -> list[Any]:
            return apply_event(messages, normalize_summarization_event(event))

        setattr(summarization, "_apply_event_to_messages", wrapped_apply_event_to_messages)

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
            normalized.append(normalize_summary_message(message))
        else:
            normalized.append(message)
    return normalized


def normalize_summarization_event(event: Any) -> Any:
    """Return a summarization event with a replay-safe summary message."""
    if not isinstance(event, dict) or "summary_message" not in event:
        return event
    normalized = dict(event)
    normalized["summary_message"] = normalize_summary_message(event.get("summary_message"))
    return normalized


def normalize_summary_message(message: Any) -> HumanMessage:
    """Convert checkpointed summary messages into provider-safe HumanMessages."""
    if isinstance(message, str):
        return HumanMessage(content=message)
    try:
        converted = convert_to_messages([message])[0]
    except Exception:
        converted = message
    return HumanMessage(content=visible_text(converted))


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
    "PostTurnCompactionResult",
    "compact_after_turn",
    "create_mira_summarization_tool_middleware",
    "mark_summarization_engine",
    "sanitize_messages_for_archive",
]
