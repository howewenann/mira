"""Durable session transcript helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from session.dashboard import normalize_dashboard

UNTITLED_SESSION = "Untitled session"
TITLE_MAX_CHARS = 48
TITLE_MESSAGE_LIMIT = 3
RESUME_MESSAGE_LIMIT = 20

TECH_RE = re.compile(r"[A-Za-z0-9_./\\:-]+")
SUMMARY_RE = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL | re.IGNORECASE)
FILLER_PHRASES = (
    "can you",
    "could you",
    "would you",
    "please",
    "help me",
    "lets",
    "let's",
    "i want to",
    "i was thinking",
)
CHATTER = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank",
    "ok",
    "okay",
    "cool",
    "nice",
    "yes",
    "no",
    "yep",
    "nope",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "but",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "with",
    "you",
    "your",
}
ACTION_WORDS = {
    "add",
    "build",
    "check",
    "debug",
    "fix",
    "implement",
    "inspect",
    "remove",
    "review",
    "update",
}


def normalize_session(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id", "")),
        "title": safe_title(record.get("title")),
        "workspace": str(record.get("workspace", "")),
        "created_at": str(record.get("created_at", now_iso())),
        "updated_at": str(record.get("updated_at", record.get("created_at", now_iso()))),
        "turns": int(record.get("turns") or 0),
        "dashboard": normalize_dashboard(record.get("dashboard")),
        "events": normalize_events(record.get("events")),
    }


def normalize_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    events = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        event_type = str(item.get("type") or "")
        event = {
            "id": int(item.get("id") or index),
            "type": event_type,
            "created_at": str(item.get("created_at") or now_iso()),
        }
        mode = str(item.get("mode") or "")
        if mode:
            event["mode"] = mode

        if event_type in {"user", "assistant", "reasoning", "system_error", "interrupted"}:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            event["text"] = text
        elif event_type == "tool_call":
            event["name"] = compact_line(item.get("name") or "tool")
            event["args"] = item.get("args", {})
        elif event_type == "tool_result":
            output = str(item.get("output") or "").strip()
            if not output:
                continue
            event["name"] = compact_line(item.get("name") or "tool")
            event["output"] = output
        elif event_type == "delegation":
            calls = item.get("calls")
            if not isinstance(calls, list) or not calls:
                continue
            event["calls"] = calls
        elif event_type == "subagent":
            event["name"] = compact_line(item.get("name") or "subagent")
            event["status"] = compact_line(item.get("status") or "RUNNING")
            event["task_input"] = str(item.get("task_input") or "")
            event["output"] = str(item.get("output") or "")
        elif event_type == "compaction":
            summary = compact_text(item.get("summary"))
            file_path = compact_line(item.get("file_path"))
            cutoff_index = int(item.get("cutoff_index") or 0)
            if not summary and not file_path and cutoff_index <= 0:
                continue
            event["cutoff_index"] = cutoff_index
            event["file_path"] = file_path
            event["summary"] = summary
        else:
            continue

        events.append(event)
    return events


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    messages = []
    for item in normalize_events(value):
        if item["type"] not in {"user", "assistant"}:
            continue
        messages.append({
            "id": item["id"],
            "role": item["type"],
            "mode": str(item.get("mode") or "action"),
            "created_at": item["created_at"],
            "content": item["text"],
        })
    return messages


def normalize_compactions(value: Any) -> list[dict[str, Any]]:
    compactions = []
    for item in normalize_events(value):
        if item["type"] != "compaction":
            continue
        compactions.append(
            {
                "cutoff_index": item["cutoff_index"],
                "file_path": item["file_path"],
                "summary": item["summary"],
                "created_at": item["created_at"],
            }
        )
    return compactions


def append_message(record: dict[str, Any], role: str, content: str, mode: str) -> None:
    content = str(content or "").strip()
    if role not in {"user", "assistant"} or not content:
        return

    append_event(record, {"type": role, "mode": mode, "text": content})


def append_event(record: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    events = record.setdefault("events", [])
    next_id = max((int(item.get("id", 0)) for item in events if isinstance(item, dict)), default=0) + 1
    stored = {"id": next_id, "created_at": now_iso(), **event}
    events.append(stored)
    return stored


def update_event_text(record: dict[str, Any], event_id: int, text: str) -> None:
    for event in record.get("events", []):
        if isinstance(event, dict) and int(event.get("id") or 0) == event_id:
            event["text"] = text
            return


def update_title(record: dict[str, Any]) -> None:
    title = title_from_messages(normalize_messages(record.get("events")))
    record["title"] = title or UNTITLED_SESSION


def title_from_messages(messages: list[dict[str, Any]]) -> str:
    recent = [message["content"] for message in messages if message["role"] == "user"][-TITLE_MESSAGE_LIMIT:]
    words = title_words(" ".join(reversed(recent)))
    if not words:
        return UNTITLED_SESSION

    title = " ".join(display_word(word) for word in words)
    return safe_title(title)


def title_words(text: str) -> list[str]:
    cleaned = clean_title_source(text)
    if compact_line(cleaned).lower() in CHATTER:
        return []

    words: list[str] = []
    for match in TECH_RE.finditer(cleaned):
        raw = match.group(0).strip(".,!?;()[]{}'\"`")
        lowered = raw.lower()
        if not raw or lowered in CHATTER or lowered in STOPWORDS:
            continue
        if raw.startswith("/") and len(raw) < 3:
            continue
        if lowered in ACTION_WORDS or is_technical(raw) or len(raw) > 2:
            words.append(raw)
        if len(words) >= 7:
            break
    return trim_title_words(words)


def clean_title_source(text: str) -> str:
    cleaned = text.lower()
    for phrase in FILLER_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def trim_title_words(words: list[str]) -> list[str]:
    while words and words[0].lower() not in ACTION_WORDS and len(words) > 5:
        words.pop(0)
    while words and len(" ".join(display_word(word) for word in words)) > TITLE_MAX_CHARS:
        words.pop()
    return words


def display_word(word: str) -> str:
    if is_technical(word):
        return word
    return word.capitalize()


def is_technical(word: str) -> bool:
    return (
        any(character in word for character in "_./\\:-")
        or any(character.isdigit() for character in word)
        or any(character.isupper() for character in word[1:])
    )


async def sync_deepagents_compaction(record: dict[str, Any], agent: Any, thread_id: str) -> bool:
    state = await agent_state(agent, thread_id)
    event = state.get("_summarization_event")
    if not isinstance(event, dict):
        return False

    compaction = compaction_from_event(event)
    if compaction is None or is_known_compaction(record, compaction):
        return False

    append_event(record, {"type": "compaction", **compaction})
    return True


async def agent_state(agent: Any, thread_id: str) -> dict[str, Any]:
    getter = getattr(agent, "aget_state", None)
    if not callable(getter):
        return {}

    snapshot = await getter({"configurable": {"thread_id": thread_id}})
    values = getattr(snapshot, "values", None)
    return values if isinstance(values, dict) else {}


def compaction_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    cutoff_index = int(event.get("cutoff_index") or 0)
    file_path = compact_line(event.get("file_path"))
    summary = first_summary_text(
        event.get("summary_message"),
        event.get("summary"),
        event.get("summary_text"),
    )
    if not summary and not file_path and cutoff_index <= 0:
        return None
    return {
        "cutoff_index": cutoff_index,
        "file_path": file_path,
        "summary": summary,
        "created_at": now_iso(),
    }


def first_summary_text(*values: Any) -> str:
    for value in values:
        summary = summary_text(value)
        if summary:
            return summary
    return ""


def summary_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    text = str(content or "")
    match = SUMMARY_RE.search(text)
    return compact_text(match.group(1) if match else text)


def is_known_compaction(record: dict[str, Any], compaction: dict[str, Any]) -> bool:
    for existing in normalize_compactions(record.get("events")):
        if (
            existing["cutoff_index"] == compaction["cutoff_index"]
            and existing["file_path"] == compaction["file_path"]
        ):
            return True
    return False


def build_resume_context(record: dict[str, Any]) -> str:
    compactions = normalize_compactions(record.get("events"))
    messages = normalize_messages(record.get("events"))[-RESUME_MESSAGE_LIMIT:]
    if not compactions and not messages:
        return ""

    parts = ["Previous MIRA session context:"]
    if compactions:
        latest = compactions[-1]
        if latest["summary"]:
            parts.append("Latest DeepAgents compaction summary:")
            parts.append(latest["summary"])
        if latest["file_path"]:
            parts.append(f"Evicted conversation archive: {latest['file_path']}")
    if messages:
        parts.append("Recent visible transcript:")
        for message in messages:
            parts.append(f"{message['role']} ({message['mode']}): {message['content']}")
    parts.append("Continue from this context without assuming unstated details.")
    return "\n".join(parts)


def with_resume_context(session: dict[str, Any], text: str) -> str:
    if not session.pop("resume_context_pending", False):
        return text

    context = build_resume_context(session)
    if not context:
        return text
    return f"{context}\n\nCurrent user request:\n{text}"


def mark_resume_context_pending(record: dict[str, Any], *, resumed: bool) -> None:
    record["resume_context_pending"] = resumed and (
        bool(normalize_compactions(record.get("events"))) or bool(normalize_messages(record.get("events")))
    )


def safe_title(value: Any) -> str:
    title = compact_line(value).strip("\"'` ")
    if not title:
        return UNTITLED_SESSION
    return title[:TITLE_MAX_CHARS].rstrip() or UNTITLED_SESSION


def compact_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def compact_text(value: Any) -> str:
    lines = [compact_line(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
