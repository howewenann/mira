"""Tool-call argument normalization and live draft rendering."""

from __future__ import annotations

import json
import re
from typing import Any

from runtime.compaction_filter import call_renderer
from runtime.protocol_events import event_delta, is_tool_call_delta


class ToolCallDrafts:
    """Accumulate streamed tool-call argument chunks for live draft rendering."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self._calls: dict[str, dict[str, Any]] = {}

    def push(self, chunk: Any) -> None:
        data = tool_call_chunk_data(chunk)
        if data is None:
            call_renderer(self.renderer, "model_activity")
            return

        key = data["key"]
        call = self._calls.setdefault(key, {"id": data["id"] or key, "name": "", "args_raw": ""})
        if data["id"]:
            call["id"] = data["id"]
        if data["name"]:
            call["name"] = data["name"]
        if data["replace_args"]:
            call["args_raw"] = data["args"]
        elif data["args"]:
            call["args_raw"] = f"{call.get('args_raw') or ''}{data['args']}"

        name = str(call.get("name") or "")
        if not name:
            call_renderer(self.renderer, "model_activity")
            return

        draft_call = self.draft_call(call)
        if name == "task":
            task_calls = [self.draft_call(value) for value in self._calls.values() if value.get("name") == "task"]
            if not call_renderer(self.renderer, "delegation_delta", task_calls):
                call_renderer(self.renderer, "model_activity")
            return

        if not call_renderer(
            self.renderer,
            "tool_call_delta",
            name,
            draft_call["args"],
            call_id=str(call.get("id") or ""),
        ):
            call_renderer(self.renderer, "model_activity")

    def draft_call(self, call: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "id": str(call.get("id") or ""),
            "name": str(call.get("name") or "tool"),
            "args": draft_args(call.get("args_raw", "")),
        }


def normalized_call(call: Any) -> dict[str, Any]:
    """Return a renderer-friendly tool call shape."""
    name = tool_call_name(call)
    return {
        "type": "tool_call",
        "id": tool_call_id(call),
        "name": name,
        "args": tool_call_args(call),
    }


def tool_call_id(call: Any) -> str:
    """Return a stable id for a streamed or finalized tool call."""
    if isinstance(call, dict):
        return str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or "")
    return str(getattr(call, "id", "") or getattr(call, "call_id", "") or getattr(call, "tool_call_id", "") or "")


def tool_call_name(call: Any) -> str:
    """Return a tool-call name across DeepAgents and LangChain shapes."""
    if isinstance(call, dict):
        return str(call.get("tool_name") or call.get("name") or "tool")
    return str(getattr(call, "tool_name", "") or getattr(call, "name", "") or "tool")


def tool_call_args(call: Any) -> Any:
    """Return tool-call input/args across DeepAgents and LangChain shapes."""
    if isinstance(call, dict):
        args = call.get("input") if "input" in call else call.get("args")
        return {} if args is None else args
    args = getattr(call, "input", None)
    if args is not None:
        return args
    args = getattr(call, "args", None)
    return {} if args is None else args


def tool_call_chunk_data(chunk: Any) -> dict[str, Any] | None:
    """Return normalized data from a provider tool-call chunk."""
    source = chunk
    data = event_delta(chunk) if isinstance(chunk, dict) and ("delta" in chunk or "content_block" in chunk) else chunk
    if not isinstance(data, dict):
        return None

    delta_type = str(data.get("type") or "")
    if delta_type and not is_tool_call_delta(delta_type):
        return None

    function = data.get("function") if isinstance(data.get("function"), dict) else {}
    name = data.get("name") or function.get("name") or ""
    raw_args = first_present(data, "args", "arguments", "input")
    if raw_args is None:
        raw_args = first_present(function, "arguments", "args")
    call_id = str(data.get("id") or data.get("call_id") or data.get("tool_call_id") or "")
    index = data.get("index")

    if not name and raw_args is None:
        return None

    if isinstance(raw_args, bytes):
        args = raw_args.decode("utf-8", errors="replace")
    elif isinstance(raw_args, str):
        args = raw_args
    elif raw_args is None:
        args = ""
    else:
        args = json.dumps(raw_args, ensure_ascii=False)

    key = f"index:{index}" if index is not None else (call_id or "index:0")
    return {
        "key": key,
        "id": call_id,
        "name": str(name),
        "args": args,
        "replace_args": isinstance(source, dict) and "content_block" in source,
    }


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    """Return the first present key from a mapping, preserving falsey values."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def draft_args(raw: Any) -> Any:
    """Best-effort readable args for incomplete JSON tool-call chunks."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return raw
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        pass

    fields: dict[str, str] = {}
    for key in ("description", "subagent_type"):
        value = partial_json_string_field(raw, key)
        if value:
            fields[key] = value
    if fields:
        return fields
    return raw


def partial_json_string_field(raw: str, key: str) -> str:
    """Extract a readable partial JSON string field if the value is incomplete."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)', raw)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1).replace('\\"', '"')
