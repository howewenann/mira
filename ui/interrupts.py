"""Helpers for LangGraph interrupt payloads shown in the UI."""

from __future__ import annotations

import json
from typing import Any

ASK_USER_OPEN_OPTION = "Tell MIRA what to do differently"


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
    if isinstance(action, dict):
        name = action.get("name", "tool")
        args = action.get("args", {})
        return f"{name}\n\n{json.dumps(args, indent=2)}"
    return str(action)
