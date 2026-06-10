"""Scrollable chat output for the Textual TUI."""

from __future__ import annotations

import json
import re
from itertools import count
from typing import Any

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from session.context import normalize_events
from ui.splash import splash_text

DEFAULT_TOOL_OUTPUT_CHARS = 240
SPINNER_FRAMES = ["-", "\\", "|", "/"]


class ChatLog(VerticalScroll):
    """A small scrollable chat transcript with streaming message updates."""

    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_output_chars = int(tool_output_chars)
        self._assistant_text = ""
        self._assistant_block: Static | None = None
        self._reasoning_text = ""
        self._reasoning_block: Static | None = None
        self._subagent_labels: dict[int, str] = {}
        self._subagent_blocks: dict[str, dict[str, str]] = {}
        self._subagent_widgets: dict[str, Static] = {}
        self._compaction_block: Static | None = None
        self._compaction_running = False
        self._fallback_suffixes = count(1)
        self._spinner_index = 0
        self._faker: Any | None = None
        try:
            from faker import Faker

            self._faker = Faker()
        except Exception:
            self._faker = None

    def startup(self, *, model_name: str, session_id: str, workspace: str) -> None:
        """Show session metadata when the app opens."""
        self._add_block(
            "mira",
            splash_text(model_name=model_name, session_id=session_id, workspace=workspace),
            "message startup",
        )

    def user_message(self, text: str, *, planning: bool = False) -> None:
        """Append a submitted user message."""
        title = "you (plan)" if planning else "you"
        self._add_block(title, Text(text), "message user")

    def assistant_message(self, text: str) -> None:
        """Append a completed assistant message."""
        self._add_block("mira", Text(text), "message assistant")

    def restore_session(self, session: dict[str, Any]) -> None:
        """Replay persisted visible session events."""
        self.finish_main()
        for event in normalize_events(session.get("events")):
            event_type = event["type"]
            if event_type == "user":
                self.user_message(event["text"], planning=event.get("mode") == "planning")
            elif event_type == "assistant":
                self.assistant_message(event["text"])
            elif event_type == "reasoning":
                self._add_block("thinking", Text(event["text"]), "message reasoning")
            elif event_type == "tool_call":
                self.tool_call(event["name"], event.get("args", {}))
            elif event_type == "tool_result":
                self.tool_result(event["name"], event["output"])
            elif event_type == "delegation":
                self.delegation_started(event["calls"])
            elif event_type == "subagent":
                if event.get("status") == "DONE":
                    self.subagent_finished(event["name"], event.get("output", ""))
                else:
                    self.subagent_started(event["name"], event.get("task_input", ""))
            elif event_type == "compaction":
                self._add_block("session compacted", self._compaction_text(event), "message summary")
            elif event_type in {"system_error", "interrupted"}:
                self.system_message(event["text"], kind="error" if event_type == "system_error" else "warning")

    def reasoning_delta(self, delta: str) -> None:
        """Append streamed reasoning text to the current reasoning block."""
        cleaned = re.sub(r"</?[^>]+>", "", delta)
        if not cleaned:
            return

        if self._reasoning_block is None:
            self._reasoning_text = ""
            self._reasoning_block = self._add_block("thinking", Text(""), "message reasoning")

        self._reasoning_text += cleaned
        self._reasoning_block.update(Text(self._reasoning_text))
        self._scroll_to_end()

    def text_delta(self, delta: str) -> None:
        """Append streamed assistant text to the current response block."""
        if not delta:
            return

        if self._assistant_block is None:
            self._assistant_text = ""
            self._assistant_block = self._add_block("mira", Text(""), "message assistant")

        self._assistant_text += delta
        self._assistant_block.update(Text(self._assistant_text))
        self._scroll_to_end()

    def finish_main(self) -> None:
        """Close the current streamed blocks so the next turn starts fresh."""
        self._assistant_block = None
        self._assistant_text = ""
        self._reasoning_block = None
        self._reasoning_text = ""

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Append a system, status, warning, or error message."""
        title = "mira" if kind == "startup" else kind
        self._add_block(title, Text(text), f"message {kind}")

    def command_output(self, renderable: Any) -> None:
        """Append command output, including Rich renderables such as tables."""
        self._add_block("output", renderable, "message command")

    def compaction_started(self) -> None:
        """Show that DeepAgents is compacting conversation context."""
        self.finish_main()
        self._compaction_running = True
        text = self._render_compaction()
        if self._compaction_block is None:
            self._compaction_block = self._add_block("mira", text, "message status")
        else:
            self._compaction_block.update(text)
            self._scroll_to_end()

    def compaction_finished(self) -> None:
        """Mark the compaction status as complete."""
        if self._compaction_block is None:
            return
        self._compaction_running = False
        self._compaction_block.update(Text("context compacted", style="bold green"))
        self._compaction_block = None
        self._scroll_to_end()

    def tick_compaction(self) -> None:
        """Advance the spinner while context compaction is running."""
        if not self._compaction_running or self._compaction_block is None:
            return
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        self._compaction_block.update(self._render_compaction())
        self._scroll_to_end()

    def tool_call(self, name: str, args: Any) -> None:
        """Append a coordinator-level tool call in transcript order."""
        self.finish_main()
        text = Text()
        text.append("args: ", style="bold cyan")
        text.append(self.truncate(args))
        self._add_block(f"tool - {name}", text, "message tool-call")

    def tool_result(self, name: str, result: str) -> None:
        """Append a coordinator-level tool result in transcript order."""
        if not result:
            return
        self.finish_main()
        text = Text()
        text.append("output: ", style="dim")
        text.append(self.truncate(result), style="dim")
        self._add_block(f"{name} result", text, "message tool-result")

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Append a compact task delegation summary."""
        descriptions, errors = self._delegation_details(calls)
        if not descriptions and not errors:
            return

        self.finish_main()
        text = Text()
        label = "subagent" if len(descriptions) == 1 else "subagents"
        if descriptions:
            text.append(f"delegating to {len(descriptions)} {label}\n", style="bold yellow")
        for description in descriptions:
            text.append("request: ", style="bold cyan")
            text.append(self.truncate(description) + "\n")
        for error in errors:
            text.append(f"failed: {error}\n", style="red")
        self._add_block("task", text, "message delegation")

    def start_subagent_live(self) -> None:
        """Reset subagent state for a new delegation group."""
        self._subagent_blocks = {}
        self._subagent_widgets = {}
        self._spinner_index = 0

    def stop_subagent_live(self) -> None:
        """Finalize subagent display."""
        for label in list(self._subagent_blocks):
            self._update_subagent(label)

    def tick_subagents(self) -> None:
        """Advance the spinner on running subagents."""
        if not self.has_running_subagents():
            return

        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        for label, block in self._subagent_blocks.items():
            if block.get("status") == "RUNNING":
                self._update_subagent(label)

    def has_running_subagents(self) -> bool:
        """Return whether a subagent is still running."""
        return any(block.get("status") == "RUNNING" for block in self._subagent_blocks.values())

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable readable label for a subagent object."""
        key = id(subagent)
        if key not in self._subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self._subagent_labels[key] = f"{name} [{self._next_suffix()}]"
        return self._subagent_labels[key]

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Create or replace the block for a running subagent."""
        self.finish_main()
        self._subagent_blocks[subagent] = {
            "request": task_input,
            "status": "RUNNING",
            "output": "",
        }
        widget = self._add_block(f"subagent - {subagent}", self._render_subagent(subagent), "message subagent")
        self._subagent_widgets[subagent] = widget

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Mark a subagent block as done and attach its final output."""
        block = self._subagent_blocks.setdefault(
            subagent,
            {
                "request": "",
                "status": "RUNNING",
                "output": "",
            },
        )
        block["status"] = "DONE"
        block["output"] = result
        self._update_subagent(subagent)

    def plan(self, plan_id: int, text: str) -> None:
        """Append a saved planning-mode result."""
        self._add_block(f"plan #{plan_id}", Text(text), "message plan")

    def no_plans(self) -> None:
        """Append the empty state for saved plans."""
        self.system_message("no saved plans", kind="muted")

    def clear_log(self) -> None:
        """Remove all chat messages."""
        self.finish_main()
        self._subagent_blocks = {}
        self._subagent_widgets = {}
        self._compaction_block = None
        self._compaction_running = False
        for child in list(self.children):
            child.remove()

    def truncate(self, value: Any) -> str:
        """Return a compact one-line display string."""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text
        return text[: self.tool_output_chars].rstrip() + " ... truncated ..."

    def _add_block(self, title: str, renderable: Any, classes: str) -> Static:
        """Mount one bordered transcript block."""
        block = Static(renderable, classes=classes)
        block.border_title = title
        self.mount(block)
        self._scroll_to_end()
        return block

    def _scroll_to_end(self) -> None:
        """Keep new output visible."""
        self.call_after_refresh(self.scroll_end, animate=False, force=True)

    def _delegation_details(self, calls: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """Return valid task descriptions and compact parse errors."""
        descriptions: list[str] = []
        errors: list[str] = []
        for call in calls:
            raw_args = call.get("args", {}) if isinstance(call, dict) else {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (TypeError, json.JSONDecodeError):
                    errors.append(f"could not parse args: {str(raw_args)[:60]}")
                    continue
            args = raw_args if isinstance(raw_args, dict) else {}
            description = args.get("description")
            if description:
                descriptions.append(str(description))
            else:
                errors.append(f"missing description in args: {str(args)[:60]}")
        return descriptions, errors

    def _compaction_text(self, compaction: dict[str, Any]) -> Text:
        """Render a DeepAgents compaction marker."""
        text = Text()
        if compaction.get("summary"):
            text.append("summary", style="bold cyan")
            text.append(": ")
            text.append(str(compaction["summary"]))
            text.append("\n")
        if compaction.get("file_path"):
            text.append("archive", style="bold cyan")
            text.append(": ")
            text.append(str(compaction["file_path"]))
            text.append("\n")
        text.append("cutoff", style="bold cyan")
        text.append(": ")
        text.append(str(compaction.get("cutoff_index", 0)))
        return text

    def _render_subagent(self, label: str) -> Text:
        """Render one subagent status block."""
        block = self._subagent_blocks[label]
        status = block["status"]
        text = Text()

        if block.get("request"):
            text.append("request: ", style="bold cyan")
            text.append(self.truncate(block["request"]))
            text.append("\n")

        text.append("status: ", style="bold cyan")
        if status == "RUNNING":
            text.append(f"{SPINNER_FRAMES[self._spinner_index]} RUNNING", style="bold yellow")
        else:
            text.append("DONE", style="bold green")

        if block.get("output"):
            text.append("\noutput: ", style="bold cyan")
            text.append(self.truncate(block["output"]))

        return text

    def _render_compaction(self) -> Text:
        """Render the live context compaction status."""
        text = Text()
        text.append(f"{SPINNER_FRAMES[self._spinner_index]} ", style="bold yellow")
        text.append("compacting context...", style="bold yellow")
        return text

    def _update_subagent(self, label: str) -> None:
        """Update an existing subagent widget."""
        widget = self._subagent_widgets.get(label)
        if widget is None:
            widget = self._add_block(f"subagent - {label}", self._render_subagent(label), "message subagent")
            self._subagent_widgets[label] = widget
            return
        widget.update(self._render_subagent(label))
        self._scroll_to_end()

    def _next_suffix(self) -> str:
        """Return a short cute suffix for delegated workers."""
        if self._faker is None:
            return str(next(self._fallback_suffixes))

        for _ in range(8):
            word = re.sub(r"[^a-z0-9-]", "", str(self._faker.unique.first_name()).lower())
            if word:
                return word
        return str(next(self._fallback_suffixes))
