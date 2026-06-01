"""Durable session context helpers for resume and compaction."""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timezone
from typing import Any

UNTITLED_SESSION = "Untitled session"
DEFAULT_MAX_CHARS = 40000
DEFAULT_RECENT_MESSAGES = 10
DEFAULT_SUMMARY_MAX_CHARS = 6000
SUMMARY_KEYS = (
    "objective",
    "current_status",
    "important_decisions",
    "user_preferences",
    "relevant_files",
    "next_steps",
)
TITLE_UPDATE_INTERVAL = 5
EARLY_TITLE_UPDATE_TURNS = {1, 2}


def context_policy(config: dict[str, Any] | None = None) -> dict[str, int]:
    """Return the session context policy from runtime config."""
    config = config or {}
    return {
        "max_chars": positive_int(config.get("session_max_chars", config.get("max_chars")), DEFAULT_MAX_CHARS),
        "recent_messages": positive_int(
            config.get("session_recent_messages", config.get("recent_messages")),
            DEFAULT_RECENT_MESSAGES,
        ),
        "summary_max_chars": positive_int(
            config.get("session_summary_max_chars", config.get("summary_max_chars")),
            DEFAULT_SUMMARY_MAX_CHARS,
        ),
    }


def positive_int(value: Any, default: int) -> int:
    """Return value as a positive integer, falling back to default."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def normalize_session(record: dict[str, Any], policy: dict[str, int] | None = None) -> dict[str, Any]:
    """Return a session record with the V1 fields in readable order."""
    policy = policy or context_policy(record.get("context_policy"))
    return {
        "id": str(record.get("id", "")),
        "title": safe_title(record.get("title") or UNTITLED_SESSION),
        "workspace": str(record.get("workspace", "")),
        "created_at": str(record.get("created_at", now_iso())),
        "updated_at": str(record.get("updated_at", record.get("created_at", now_iso()))),
        "turns": int(record.get("turns") or 0),
        "context_policy": policy,
        "summary": normalize_summary(record.get("summary")),
        "messages": normalize_messages(record.get("messages")),
    }


def normalize_summary(value: Any) -> dict[str, Any] | None:
    """Return a valid summary object or None."""
    if not isinstance(value, dict):
        return None
    state = value.get("state")
    if not isinstance(state, dict):
        return None
    return {
        "version": int(value.get("version") or 1),
        "kind": str(value.get("kind") or "llm_compaction"),
        "through_message": int(value.get("through_message") or 0),
        "updated_at": str(value.get("updated_at") or now_iso()),
        "state": normalize_state(state),
    }


def normalize_state(value: dict[str, Any]) -> dict[str, Any]:
    """Return a structured continuation state with expected keys."""
    state: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        item = value.get(key)
        if key in {"objective", "current_status"}:
            state[key] = compact_line(item)
        else:
            state[key] = [compact_line(part) for part in as_list(item) if compact_line(part)]
    return state


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    """Return valid session messages with stable ids."""
    if not isinstance(value, list):
        return []

    messages = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append(
            {
                "id": int(item.get("id") or index),
                "role": role,
                "mode": str(item.get("mode") or "action"),
                "created_at": str(item.get("created_at") or now_iso()),
                "content": content,
            }
        )
    return messages


def append_turn(record: dict[str, Any], user_text: str, assistant_text: str, mode: str) -> None:
    """Append one completed user/assistant turn to a session record."""
    append_message(record, "user", user_text, mode)
    if assistant_text.strip():
        append_message(record, "assistant", assistant_text, mode)


def append_message(record: dict[str, Any], role: str, content: str, mode: str) -> None:
    """Append a single session message."""
    content = str(content or "").strip()
    if role not in {"user", "assistant"} or not content:
        return

    messages = record.setdefault("messages", [])
    next_id = max((int(message.get("id", 0)) for message in messages), default=0) + 1
    messages.append(
        {
            "id": next_id,
            "role": role,
            "mode": mode,
            "created_at": now_iso(),
            "content": content,
        }
    )


def needs_title_update(record: dict[str, Any]) -> bool:
    """Return whether the session title should be generated or refreshed."""
    if not first_user_message(record) or not first_assistant_message(record):
        return False

    if safe_title(record.get("title")) == UNTITLED_SESSION:
        return True

    turns = int(record.get("turns") or 0)
    return turns in EARLY_TITLE_UPDATE_TURNS or (
        turns > 0 and turns % TITLE_UPDATE_INTERVAL == 0
    )


def needs_title(record: dict[str, Any]) -> bool:
    """Return whether the session needs its first generated title."""
    return needs_title_update(record)


async def update_title(record: dict[str, Any], model: Any) -> None:
    """Generate or refresh a concise title for the current session."""
    if not needs_title_update(record):
        return

    messages = normalize_messages(record.get("messages"))[-12:]
    prompt = TITLE_PROMPT.format(messages=json.dumps(messages, indent=2))
    raw_title = compact_line(await model_text(model, prompt))
    if not raw_title and safe_title(record.get("title")) != UNTITLED_SESSION:
        return

    title = safe_title(raw_title) if raw_title else fallback_title(first_user_message(record))
    record["title"] = title


async def update_title_once(record: dict[str, Any], model: Any) -> None:
    """Backward-compatible alias for title generation."""
    await update_title(record, model)


async def compact_if_needed(record: dict[str, Any], model: Any) -> None:
    """Compact older messages into structured continuation state when needed."""
    policy = context_policy(record.get("context_policy"))
    record["context_policy"] = policy
    messages = normalize_messages(record.get("messages"))
    record["messages"] = messages

    if not will_compact(record):
        return

    keep = max(1, policy["recent_messages"])
    older = messages[:-keep]
    recent = messages[-keep:]
    if not older:
        return

    summary = await summarize_messages(record.get("summary"), older, recent, model, policy["summary_max_chars"])
    if summary is None:
        return
    summary["through_message"] = older[-1]["id"]
    record["summary"] = summary
    record["messages"] = recent


def will_compact(record: dict[str, Any]) -> bool:
    """Return whether the current session record is large enough to compact."""
    policy = context_policy(record.get("context_policy"))
    messages = normalize_messages(record.get("messages"))
    if message_chars(messages) <= policy["max_chars"]:
        return False

    keep = max(1, policy["recent_messages"])
    return bool(messages[:-keep])


async def summarize_messages(
    previous_summary: Any,
    older: list[dict[str, Any]],
    recent: list[dict[str, Any]],
    model: Any,
    max_chars: int,
) -> dict[str, Any] | None:
    """Ask the configured LLM for structured continuation state."""
    previous = json.dumps(previous_summary, indent=2) if previous_summary else "None"
    prompt = SUMMARY_PROMPT.format(
        max_chars=max_chars,
        previous_summary=previous,
        older_messages=json.dumps(older, indent=2),
        recent_messages=json.dumps(recent, indent=2),
    )
    response = await model_text(model, prompt)
    if not response.strip():
        return None
    state = parse_state(response)
    summary = build_summary(state)
    if summary_chars(summary) <= max_chars:
        return summary

    retry_prompt = prompt + "\n\nYour previous response was too long. Return a much shorter JSON object now."
    response = await model_text(model, retry_prompt)
    if not response.strip():
        return trim_summary(summary, max_chars)
    state = parse_state(response)
    summary = build_summary(state)
    return trim_summary(summary, max_chars)


def build_resume_context(record: dict[str, Any]) -> str:
    """Build context text to inject once at the start of a resumed session."""
    summary = normalize_summary(record.get("summary"))
    messages = normalize_messages(record.get("messages"))
    if not summary and not messages:
        return ""

    parts = ["Previous MIRA session context:"]
    if summary:
        parts.append("Continuation state:")
        parts.append(json.dumps(summary["state"], indent=2))
    if messages:
        parts.append("Recent messages:")
        for message in messages:
            parts.append(f"{message['role']} ({message.get('mode', 'action')}): {message['content']}")
    parts.append("Continue from this context without assuming unstated details.")
    return "\n".join(parts)


def with_resume_context(session: dict[str, Any], text: str) -> str:
    """Inject restored context once into the next user request."""
    if not session.pop("resume_context_pending", False):
        return text

    context = build_resume_context(session)
    if not context:
        return text

    return f"{context}\n\nCurrent user request:\n{text}"


def mark_resume_context_pending(record: dict[str, Any], *, resumed: bool) -> None:
    """Mark whether restored context should be injected into the next request."""
    record["resume_context_pending"] = resumed and (bool(record.get("summary")) or bool(record.get("messages")))


def message_chars(messages: list[dict[str, Any]]) -> int:
    """Return total message content characters."""
    return sum(len(str(message.get("content") or "")) for message in messages)


def first_user_message(record: dict[str, Any]) -> str:
    """Return the first stored user message content."""
    for message in normalize_messages(record.get("messages")):
        if message["role"] == "user":
            return message["content"]
    return ""


def first_assistant_message(record: dict[str, Any]) -> str:
    """Return the first stored assistant message content."""
    for message in normalize_messages(record.get("messages")):
        if message["role"] == "assistant":
            return message["content"]
    return ""


async def model_text(model: Any, prompt: str) -> str:
    """Return text from a LangChain-like model."""
    try:
        invoke = getattr(model, "ainvoke", None)
        if callable(invoke):
            value = invoke(prompt)
        else:
            invoke = getattr(model, "invoke", None)
            value = invoke(prompt) if callable(invoke) else ""

        if inspect.isawaitable(value):
            value = await value
    except Exception:
        return ""

    return extract_text(value)


def extract_text(value: Any) -> str:
    """Extract plain text from common model response shapes."""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if text is not None:
        return str(text)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(value or "")


def parse_state(text: str) -> dict[str, Any]:
    """Parse an LLM JSON object into continuation state."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return fallback_state(text)
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback_state(text)

    if isinstance(value, dict) and isinstance(value.get("state"), dict):
        value = value["state"]
    return normalize_state(value if isinstance(value, dict) else {})


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Build a summary object around continuation state."""
    return {
        "version": 1,
        "kind": "llm_compaction",
        "through_message": 0,
        "updated_at": now_iso(),
        "state": normalize_state(state),
    }


def trim_summary(summary: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Conservatively shorten summary fields until it fits."""
    state = normalize_state(summary.get("state", {}))
    while summary_chars({"state": state}) > max_chars:
        changed = False
        for key in ("next_steps", "relevant_files", "user_preferences", "important_decisions"):
            items = state.get(key, [])
            if isinstance(items, list) and len(items) > 1:
                state[key] = items[:-1]
                changed = True
                break
        if not changed:
            for key in ("current_status", "objective"):
                value = str(state.get(key) or "")
                if len(value) > 200:
                    state[key] = value[:197].rstrip() + "..."
                    changed = True
                    break
        if not changed:
            for key in ("next_steps", "relevant_files", "user_preferences", "important_decisions"):
                items = state.get(key, [])
                if isinstance(items, list) and any(len(str(item)) > 120 for item in items):
                    state[key] = [str(item)[:117].rstrip() + "..." if len(str(item)) > 120 else item for item in items]
                    changed = True
                    break
        if not changed:
            break
    if summary_chars({"state": state}) > max_chars:
        state = {
            "objective": str(state.get("objective") or "")[:197].rstrip() + "...",
            "current_status": str(state.get("current_status") or "")[:197].rstrip() + "...",
            "important_decisions": [],
            "user_preferences": [],
            "relevant_files": [],
            "next_steps": [],
        }
    summary["state"] = state
    return summary


def summary_chars(summary: dict[str, Any]) -> int:
    """Return serialized summary length."""
    return len(json.dumps(summary.get("state", summary), ensure_ascii=False))


def fallback_state(text: str) -> dict[str, Any]:
    """Return minimal state if an LLM response cannot be parsed."""
    return normalize_state(
        {
            "objective": compact_line(text)[:500],
            "current_status": "",
            "important_decisions": [],
            "user_preferences": [],
            "relevant_files": [],
            "next_steps": [],
        }
    )


def as_list(value: Any) -> list[Any]:
    """Return value as a list."""
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def safe_title(value: Any) -> str:
    """Return a clean one-line session title."""
    title = compact_line(value)
    if not title:
        return UNTITLED_SESSION
    title = title.strip("\"'` ")
    return title[:80].rstrip() or UNTITLED_SESSION


def fallback_title(user_text: str) -> str:
    """Return a deterministic title from user text."""
    return safe_title(user_text) if user_text else UNTITLED_SESSION


def compact_line(value: Any) -> str:
    """Collapse text to a single line."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def now_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


TITLE_PROMPT = """Create a concise title for this MIRA coding session.
Return only the title, with no quotes or punctuation wrapper.
Maximum 8 words.
Use the current goal of the session, not greetings or setup chatter.
Avoid generic titles like "MIRA Session Kickoff" or "Greeting".

Recent session messages:
{messages}
"""

SUMMARY_PROMPT = """Compact this MIRA session for durable resume.
Return only a JSON object with exactly these keys:
objective: string
current_status: string
important_decisions: array of strings
user_preferences: array of strings
relevant_files: array of strings
next_steps: array of strings

This is continuation state for the next model invocation, not a user-facing recap.
Preserve concrete goals, decisions, constraints, current work status, relevant files, and next steps.
Do not invent completed work. Keep the JSON under {max_chars} characters.

Previous summary:
{previous_summary}

Messages to compact:
{older_messages}

Recent messages kept verbatim for context:
{recent_messages}
"""
