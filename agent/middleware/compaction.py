"""MIRA wrappers for DeepAgents compaction middleware."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any

from deepagents.backends.protocol import FILE_NOT_FOUND
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware._retry import default_retry_on
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, convert_to_messages

logger = logging.getLogger(__name__)


def create_mira_summarization_middleware(model: Any, backend: Any, summary_model: Any | None = None) -> Any:
    """Use Main for DeepAgents thresholds and an optional model for summary text."""
    middleware = create_summarization_middleware(model, backend)
    if summary_model is not None:
        # LangChain keeps summary generation separate from the model used for
        # profile limits, token counting, and reported-usage validation.
        helper = middleware._lc_helper  # noqa: SLF001
        if not hasattr(helper, "_summary_model"):
            raise RuntimeError("LangChain no longer exposes its summary-model slot")
        helper._summary_model = summary_model.with_retry(  # noqa: SLF001
            retry_if_exception_type=default_retry_on,
        )
    prepare_summarization_engine(middleware)
    return middleware


@dataclass(frozen=True)
class PostTurnCompactionResult:
    """Outcome of a post-turn compaction attempt."""

    compacted: bool = False
    reason: str = ""
    file_path: str = ""
    summary: str = ""


async def compact_after_turn(agent: Any, thread_id: str) -> PostTurnCompactionResult:
    """Compact older context without running another conversational turn."""
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
    if getattr(snapshot, "next", ()) or getattr(snapshot, "tasks", ()):
        return PostTurnCompactionResult(reason="pending_work")

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

    state_cutoff = summarization._compute_state_cutoff(event, cutoff)
    prior_cutoff = event.get("cutoff_index", 0) if isinstance(event, dict) else 0
    if not isinstance(prior_cutoff, int) or isinstance(prior_cutoff, bool):
        prior_cutoff = 0
    if state_cutoff <= prior_cutoff:
        return PostTurnCompactionResult(reason="nothing_to_compact")
    to_summarize, _preserved = summarization._partition_messages(effective, cutoff)
    if not to_summarize:
        return PostTurnCompactionResult(reason="nothing_to_compact")

    backend = getattr(summarization, "_backend", None)
    if callable(backend):
        return PostTurnCompactionResult(reason="backend_unavailable")
    summary = await summarization._acreate_summary(to_summarize)
    latest = await agent.aget_state(config)
    if getattr(latest, "config", None) != getattr(snapshot, "config", None) or getattr(latest, "values", None) != state:
        raise RuntimeError("Conversation changed while compaction was running")
    session_id = summarization._get_session_id(state)
    # Reserve the summary in the checkpoint before appending to an archive.
    # A failed state write must never leave an append that a retry duplicates.
    reserved_event = {
        "cutoff_index": state_cutoff,
        "summary_message": summarization._build_new_messages_with_path(summary, None)[0],
        "file_path": None,
    }
    reservation_config = await agent.aupdate_state(
        getattr(latest, "config", None) or config,
        {"_summarization_event": reserved_event, "_summarization_session_id": session_id},
        as_node="model",
    )
    try:
        file_path = await summarization._aoffload_to_backend(backend, to_summarize, session_id)
    except Exception:
        logger.exception("Conversation history archiving failed after reserving a summary")
        return PostTurnCompactionResult(compacted=True, reason="archive_failed", summary=summary)
    if file_path is None:
        return PostTurnCompactionResult(compacted=True, reason="archive_failed", summary=summary)

    reserved = await agent.aget_state(config)
    reserved_state = getattr(reserved, "values", None)
    if (
        (reservation_config and getattr(reserved, "config", None) != reservation_config)
        or getattr(reserved, "next", ())
        or getattr(reserved, "tasks", ())
        or not isinstance(reserved_state, dict)
        or reserved_state.get("_summarization_event") != reserved_event
    ):
        raise RuntimeError("Conversation changed before its archive could be linked")
    linked_event = {
        **reserved_event,
        "summary_message": summarization._build_new_messages_with_path(summary, file_path)[0],
        "file_path": file_path,
    }
    await agent.aupdate_state(
        getattr(reserved, "config", None) or config,
        {"_summarization_event": linked_event},
        as_node="model",
    )
    return PostTurnCompactionResult(
        compacted=True,
        reason="compacted",
        file_path=str(file_path or ""),
        summary=str(summary or ""),
    )


def prepare_summarization_engine(summarization: Any) -> None:
    """Keep DeepAgents archives and summary replay provider-safe."""
    if summarization is None or getattr(summarization, "_mira_compaction_prepared", False):
        return

    offload = getattr(summarization, "_offload_to_backend", None)
    if callable(offload):

        @wraps(offload)
        def wrapped_offload(backend: Any, messages: list[Any], session_id: str) -> Any:
            _check_archive_read(backend.download_files([summarization._get_history_path(session_id)]))
            return offload(backend, sanitize_messages_for_archive(messages), session_id)

        setattr(summarization, "_offload_to_backend", wrapped_offload)

    aoffload = getattr(summarization, "_aoffload_to_backend", None)
    if callable(aoffload):

        @wraps(aoffload)
        async def wrapped_aoffload(backend: Any, messages: list[Any], session_id: str) -> Any:
            _check_archive_read(await backend.adownload_files([summarization._get_history_path(session_id)]))
            return await aoffload(backend, sanitize_messages_for_archive(messages), session_id)

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

    setattr(summarization, "_mira_compaction_prepared", True)


def _check_archive_read(responses: Any) -> None:
    """Refuse an append when DeepAgents could mistake a read error for a new file."""
    if not responses or len(responses) != 1:
        raise RuntimeError("Could not read conversation history before compaction")
    response = responses[0]
    if response.error not in (None, FILE_NOT_FOUND):
        raise RuntimeError(f"Could not read conversation history: {response.error}")
    if response.error is None and response.content is None:
        raise RuntimeError("Could not read conversation history before compaction")


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
        normalized = HumanMessage(content=message)
        object.__setattr__(normalized, "_mira_summary", True)
        return normalized
    try:
        converted = convert_to_messages([message])[0]
    except Exception:
        converted = message
    normalized = HumanMessage(content=visible_text(converted))
    object.__setattr__(normalized, "_mira_summary", True)
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
    if getattr(message, "_mira_summary", False):
        return True
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
    "create_mira_summarization_middleware",
    "prepare_summarization_engine",
    "sanitize_messages_for_archive",
]
