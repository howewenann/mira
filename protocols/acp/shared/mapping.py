"""Shared presentation translations from MIRA events to stock ACP updates."""

from __future__ import annotations

import json
from typing import Any, Mapping
from uuid import uuid4

from acp import (
    plan_entry,
    start_edit_tool_call,
    start_read_tool_call,
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_plan,
    update_tool_call,
)
from acp.schema import AgentMessageChunk, AgentThoughtChunk, UserMessageChunk

from mira.api import MessageEvent, ToolEvent


def message_updates(event: MessageEvent) -> list[Any]:
    """Convert one streamed MIRA message event to stable ACP chunks."""
    if event.phase not in {"content", "reasoning"}:
        return []
    blocks = [_content_block(block) for block in event.content_blocks]
    blocks = [block for block in blocks if block is not None]
    if not blocks and event.text:
        blocks = [text_block(event.text)]
    update_type = AgentThoughtChunk if event.phase == "reasoning" else AgentMessageChunk
    session_update = "agent_thought_chunk" if event.phase == "reasoning" else "agent_message_chunk"
    return [
        update_type(
            session_update=session_update,
            content=block,
            message_id=event.message_id or None,
        )
        for block in blocks
    ]


def conversational_message_updates(text: str, *, message_id: str | None = None) -> list[Any]:
    """Project adapter control-surface text through the ordinary message path."""
    return message_updates(
        MessageEvent(
            phase="content",
            text=text,
            message_id=message_id or uuid4().hex,
        )
    )


def tool_start(call_id: str, name: str, arguments: Mapping[str, Any]) -> Any:
    """Create a concise stock-ACP presentation for one MIRA tool start."""
    args = dict(arguments)
    path = str(args.get("file_path") or args.get("path") or "")
    if name == "read_file":
        return start_read_tool_call(call_id, f"Read `{path}`" if path else "Read file", path)
    if name == "edit_file":
        old_text = _optional_text(args.get("old_string"))
        new_text = _optional_text(args.get("new_string"))
        if path and new_text is not None:
            diff = tool_diff_content(path, new_text, old_text)
            return start_edit_tool_call(
                call_id,
                f"Edit `{path}`",
                path,
                new_text,
                extra_options=[diff],
            )
        return start_tool_call(
            call_id,
            f"Edit `{path}`" if path else "Edit file",
            kind="edit",
            status="pending",
            raw_input=args,
        )
    if name == "write_file":
        content = _optional_text(args.get("content"))
        if path and content is not None:
            diff = tool_diff_content(path, content)
            return start_edit_tool_call(
                call_id,
                f"Write `{path}`",
                path,
                content,
                extra_options=[diff],
            )
        return start_tool_call(
            call_id,
            f"Write `{path}`" if path else "Write file",
            kind="edit",
            status="pending",
            raw_input=args,
        )
    if name == "execute":
        command = str(args.get("command") or "")
        return start_tool_call(
            call_id,
            command or "Execute command",
            kind="execute",
            status="pending",
            raw_input=args,
        )
    if name in {"ls", "glob", "grep"}:
        detail = str(args.get("pattern") or path or "")
        titles = {"ls": "List files", "glob": "Find files", "grep": "Search files"}
        title = f"{titles[name]}: `{detail}`" if detail else titles[name]
        return start_tool_call(
            call_id,
            title,
            kind="search",
            status="pending",
            raw_input=args,
        )
    return start_tool_call(
        call_id,
        name.replace("_", " ").strip().title() or "Tool",
        kind="other",
        status="pending",
        raw_input=args,
    )


def tool_result_update(event: ToolEvent, tool: Mapping[str, Any] | None) -> Any:
    """Create the stock ACP completion update for one MIRA tool event."""
    failed = event.phase in {"error", "completed_error", "recovered_error"}
    name = str((tool or {}).get("name") or event.name)
    args = (tool or {}).get("args")
    rendered = _render_value(event.result)
    if name == "execute" and isinstance(args, Mapping):
        rendered = format_execute_result(str(args.get("command") or ""), rendered)
    content = None if name in {"edit_file", "write_file"} else [tool_content(text_block(rendered))]
    return update_tool_call(
        event.tool_call_id,
        status="failed" if failed else "completed",
        content=content,
        raw_output=event.result,
    )


def todo_update(todos: Any) -> Any:
    """Project only MIRA write_todos state to an ACP plan update."""
    entries = []
    if isinstance(todos, list):
        for item in todos:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "pending")
            if status not in {"pending", "in_progress", "completed"}:
                status = "pending"
            entries.append(
                plan_entry(
                    str(item.get("content") or ""),
                    status=status,
                    priority="medium",
                )
            )
    return update_plan(entries)


def interaction_end(call_id: str, *, cancelled: bool = False) -> Any:
    """Close a control-tool presentation without manufacturing model input."""
    return update_tool_call(
        call_id,
        status="failed" if cancelled else "completed",
        raw_output={"outcome": "cancelled" if cancelled else "reply_in_chat"},
    )


def artifact_ready(call_id: str, artifact: Mapping[str, Any]) -> Any:
    """Complete a finalizer before its artifact is emitted as a new message."""
    return update_tool_call(
        call_id,
        status="completed",
        raw_output=dict(artifact),
    )


def replay_message(event: Mapping[str, Any]) -> Any | None:
    """Project one persisted textual transcript entry to its ACP replay shape."""
    event_type = event.get("type")
    text = str(event.get("text") or "")
    message_id = str(event.get("id") or "") or None
    if event_type == "user":
        return UserMessageChunk(
            session_update="user_message_chunk",
            content=text_block(text),
            message_id=message_id,
        )
    if event_type == "assistant":
        return AgentMessageChunk(
            session_update="agent_message_chunk",
            content=text_block(text),
            message_id=message_id,
        )
    if event_type == "reasoning":
        return AgentThoughtChunk(
            session_update="agent_thought_chunk",
            content=text_block(text),
            message_id=message_id,
        )
    if event_type in {"goal", "plan"} and isinstance(event.get(event_type), Mapping):
        return conversational_message_updates(
            artifact_text(str(event_type), event[event_type]),
            message_id=message_id,
        )[0]
    return None


def artifact_text(kind: str, artifact: Mapping[str, Any]) -> str:
    """Render a formal MIRA artifact as ordinary ACP-visible Markdown."""
    lines = [f"## MIRA {kind.title()}: {artifact.get('title') or kind.title()}"]
    for heading, key in (
        ("Objective", "objective"),
        ("Context and Constraints", "context_and_constraints"),
        ("Key Changes", "key_changes"),
        ("Test Plan", "test_plan"),
        ("Assumptions", "assumptions"),
        ("Success Criteria", "success_criteria"),
    ):
        value = artifact.get(key)
        if not value:
            continue
        lines.extend(["", f"### {heading}"])
        if isinstance(value, list):
            lines.extend(f"- {item}" for item in value)
        else:
            lines.append(str(value))
    return "\n".join(lines)


def format_execute_result(command: str, output: str) -> str:
    """Format command output locally without coupling ACP to DeepAgents helpers."""
    return f"**Command:**\n```text\n{command}\n```\n\n**Output:**\n```text\n{output}\n```"


def _content_block(block: Any) -> Any | None:
    if not isinstance(block, Mapping):
        return None
    kind = str(block.get("type") or "")
    if kind in {"text", "output_text"} and block.get("text") is not None:
        return text_block(str(block["text"]))
    return None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


__all__ = [
    "artifact_text",
    "artifact_ready",
    "conversational_message_updates",
    "format_execute_result",
    "interaction_end",
    "message_updates",
    "replay_message",
    "todo_update",
    "tool_result_update",
    "tool_start",
]
