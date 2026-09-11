"""Read-only transcript inspector shared by live and restored subagent runs."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual import on
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ui.textual.widgets.chat_log import ChatLog


class SubagentInspector(Vertical):
    """A viewport-sized projection of one durable child transcript."""

    class Closed(Message):
        pass

    can_focus = False

    def __init__(self, *, tool_output_chars: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_output_chars = tool_output_chars
        self.run_id = ""

    def compose(self) -> Any:
        with Horizontal(id="subagent-inspector-header"):
            yield Static("subagent", id="subagent-inspector-title")
            yield Button("x", id="subagent-inspector-close", compact=True)
        yield Static("", id="subagent-inspector-task")
        chat = ChatLog(tool_output_chars=self.tool_output_chars, id="subagent-inspector-log")
        chat.can_focus = False
        yield chat

    def show_run(self, run: dict[str, Any]) -> None:
        """Reconstruct the inspector solely from one persisted run record."""
        self.run_id = str(run.get("id") or "")
        name = str(run.get("display_name") or run.get("name") or "subagent")
        self.query_one("#subagent-inspector-title", Static).update(f"Subagent Inspector  -  {escape(name)}")
        task = str(run.get("task_input") or "")
        self.query_one("#subagent-inspector-task", Static).update(
            f"Task: {escape(task)}" if task else "Task details unavailable"
        )
        chat = self.query_one("#subagent-inspector-log", ChatLog)
        chat.clear_log()
        events = list(run.get("events") or [])
        if run.get("status") == "ERROR" and run.get("output") and not any(
            event.get("type") == "system_error"
            or (event.get("type") == "tool_result" and event.get("status") == "error")
            for event in events
            if isinstance(event, dict)
        ):
            events.append({"type": "system_error", "text": str(run["output"])})
        chat.restore_session({"events": events})
        self.display = True

    @on(Button.Pressed, "#subagent-inspector-close")
    def close_inspector(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Closed())
