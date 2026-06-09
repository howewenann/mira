"""Helpers for LangGraph interrupt payloads shown in the UI."""

from __future__ import annotations

import json
from typing import Any

ASK_USER_OPEN_OPTION = "Tell MIRA what to do differently"
ACTION_TEXT_LIMIT = 220


def ask_user_request(interrupt: Any) -> dict[str, Any]:
    """Extract an ask_user request from a LangGraph interrupt payload."""
    value = getattr(interrupt, "value", interrupt)
    return value if isinstance(value, dict) else {}


def ask_user_question(request: dict[str, Any]) -> str:
    """Return the ask_user question text with a compact fallback."""
    question = " ".join(str(request.get("question") or "").split())
    return question or "MIRA needs a decision."


def ask_user_options(request: dict[str, Any]) -> list[str]:
    """Return unique concrete choices with the open-ended option last."""
    raw_options = request.get("options", [])
    if not isinstance(raw_options, list | tuple):
        raw_options = []

    options = []
    seen = set()
    for option in raw_options:
        text = " ".join(str(option).split())
        if not text or text == ASK_USER_OPEN_OPTION or text in seen:
            continue
        options.append(text)
        seen.add(text)

    options.append(ASK_USER_OPEN_OPTION)
    return options


def action_requests(interrupt: Any) -> list[Any]:
    """Extract approval action requests from a LangGraph interrupt payload."""
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict) and value.get("action_requests"):
        return list(value["action_requests"])
    return [value]


def action_text(action: Any) -> str:
    """Format an approval action as readable text."""
    if not isinstance(action, dict):
        return str(action)

    name = str(action.get("name") or "tool")
    args = action.get("args", {})
    if not isinstance(args, dict):
        return f"{name}\n\n{_preview_text(args)}"

    lines = [
        _action_header(name, args),
        "",
        json.dumps(_preview_value(args), indent=2),
        "",
        "Full args available with e edit.",
    ]
    return "\n".join(lines)


def _action_header(name: str, args: dict[str, Any]) -> str:
    target = _target_arg(args)
    return f"{name}\ntarget: {target}" if target else name


def _target_arg(args: dict[str, Any]) -> str:
    for key in ("file_path", "path", "filename", "command"):
        value = args.get(key)
        if value:
            return _preview_text(value, limit=120)
    return ""


def _preview_value(value: Any) -> Any:
    if isinstance(value, str):
        return _preview_text(value)
    if isinstance(value, dict):
        return {str(key): _preview_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_preview_value(item) for item in value[:20]]
    return value


def _preview_text(value: Any, *, limit: int = ACTION_TEXT_LIMIT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ... truncated ..."
