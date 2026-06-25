"""Helpers for LangGraph interrupt payloads shown in the UI."""

from __future__ import annotations

import json
import re
from typing import Any

ASK_USER_OPEN_OPTION = "Tell MIRA what to do differently"
PRESENT_PLAN_TYPE = "present_plan"
ACTION_TEXT_LIMIT = 220
ACTION_PREVIEW_VALUE_LIMIT = 68
ACTION_PREVIEW_KEY_WIDTH = 10
DEFAULT_APPROVAL_DECISIONS = ["approve", "edit", "reject"]
DECISION_LABELS = {
    "approve": ("a", "Approve (a)"),
    "edit": ("e", "Edit (e)"),
    "reject": ("r", "Reject (r)"),
}


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


def plan_request(interrupt: Any) -> dict[str, Any]:
    """Extract a structured plan request from a LangGraph interrupt payload."""
    value = getattr(interrupt, "value", interrupt)
    return normalize_plan(value if isinstance(value, dict) else {})


def normalize_plan(value: dict[str, Any]) -> dict[str, Any]:
    """Return a compact structured plan from an interrupt payload."""
    title = compact_text(value.get("title")) or "Implementation Plan"
    summary = compact_items(value.get("summary"))
    key_changes = compact_items(value.get("key_changes"))
    assumptions = compact_items(value.get("assumptions"))
    return {
        "title": title,
        "summary": summary,
        "key_changes": key_changes,
        "assumptions": assumptions,
    }


def compact_items(value: Any) -> list[str]:
    """Return compact non-empty plan list items."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return []
    items = []
    for item in value:
        text = compact_text(item)
        if text:
            items.append(text)
    return items


def compact_text(value: Any) -> str:
    """Collapse whitespace in a display string."""
    return " ".join(str(value or "").split())


def action_requests(interrupt: Any) -> list[Any]:
    """Extract approval action requests from a LangGraph interrupt payload."""
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict) and value.get("action_requests"):
        return list(value["action_requests"])
    return [value]


def action_choices(interrupt: Any, action: Any, index: int) -> list[tuple[str, str]]:
    """Return prompt choices allowed for one interrupted action."""
    choices = []
    for decision in allowed_decisions(interrupt, action, index):
        label = DECISION_LABELS.get(decision)
        if label is not None:
            choices.append(label)
    return choices or [DECISION_LABELS["approve"], DECISION_LABELS["reject"]]


def allowed_decisions(interrupt: Any, action: Any, index: int) -> list[str]:
    """Return DeepAgents review decisions allowed for one action."""
    config = review_config(interrupt, action, index)
    raw_decisions = config.get("allowed_decisions") if isinstance(config, dict) else None
    if not isinstance(raw_decisions, list):
        raw_decisions = DEFAULT_APPROVAL_DECISIONS

    allowed = []
    for decision in raw_decisions:
        text = str(decision)
        if text in DECISION_LABELS and text not in allowed:
            allowed.append(text)
    return allowed


def review_config(interrupt: Any, action: Any, index: int) -> dict[str, Any]:
    """Return the matching review config for an interrupted action."""
    value = getattr(interrupt, "value", interrupt)
    configs = value.get("review_configs") if isinstance(value, dict) else None
    if not isinstance(configs, list):
        return {}

    action_name = action.get("name") if isinstance(action, dict) else None
    for config in configs:
        if isinstance(config, dict) and config.get("action_name") == action_name:
            return config
    if index < len(configs) and isinstance(configs[index], dict):
        return configs[index]
    return {}


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


def action_title(action: Any) -> str:
    """Return the approval drawer title for one action."""
    if not isinstance(action, dict):
        return "Approval"
    return f"Approval: {str(action.get('name') or 'tool')}"


def action_preview(action: Any) -> str:
    """Return a scan-friendly preview for the approval drawer."""
    if not isinstance(action, dict):
        return _preview_text(action)

    args = action.get("args", {})
    rows: list[tuple[str, str]] = []
    if isinstance(args, dict):
        target = _target_arg(args)
        if target:
            rows.append(("target", target))
        for key, value in args.items():
            rows.append((str(key), _preview_inline(value)))
    else:
        rows.append(("args", _preview_inline(args)))

    if not rows:
        rows.append(("args", "{}"))

    key_width = max(len(key) for key, _ in rows)
    key_width = max(ACTION_PREVIEW_KEY_WIDTH, min(key_width, 18))
    lines = [f"{key.ljust(key_width)} {_preview_text(value, limit=ACTION_PREVIEW_VALUE_LIMIT)}" for key, value in rows]
    lines.extend(["", "Press e to inspect or edit full args"])
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


def _preview_inline(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(_preview_value(value), ensure_ascii=True)
        except TypeError:
            return _preview_text(value)
    return re.sub(r"\s+", " ", str(value)).strip()


def _preview_text(value: Any, *, limit: int = ACTION_TEXT_LIMIT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ... truncated ..."
