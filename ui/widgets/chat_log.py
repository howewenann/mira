"""Scrollable chat output for the Textual TUI."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from itertools import count
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from runtime.output_events import normalize_response_delta
from session.context import normalize_events
from ui.names import generate_slug
from ui.splash import loading_splash_text, splash_text

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
        self._waiting_block: Static | None = None
        self._activity_block: Static | None = None
        self._delegation_block: Static | None = None
        self._delegation_draft_block: Static | None = None
        self._delegation_calls: list[dict[str, Any]] = []
        self._delegation_keys: set[tuple[str, str]] = set()
        self._startup_block: Static | None = None
        self._startup_state = "starting"
        self._startup_workspace = ""
        self._startup_spinner_index = 0
        self._subagent_labels: dict[int, str] = {}
        self._subagent_blocks: dict[str, dict[str, str]] = {}
        self._subagent_widgets: dict[str, Static] = {}
        self._compaction_block: Static | None = None
        self._compaction_spinner_index = 0
        self._compaction_running = False
        self._fallback_suffixes = count(1)
        self._waiting_spinner_index = 0
        self._subagent_spinner_index = 0
        self._tool_sequence = count(1)
        self._tool_blocks: dict[str, dict[str, Any]] = {}
        self._tool_name_queues: dict[str, deque[str]] = defaultdict(deque)
        self._pending_tool_results_by_id: dict[str, str] = {}
        self._pending_tool_results_by_name: dict[str, deque[str]] = defaultdict(deque)
        self._subagent_aliases: dict[str, str] = {}

    def startup(self, *, model_name: str, session_id: str, workspace: str) -> None:
        """Show session metadata when the app opens."""
        self._startup_block = None
        self._add_block(
            "mira",
            splash_text(model_name=model_name, session_id=session_id, workspace=workspace),
            "message startup",
        )

    def startup_loading(self, *, workspace: str, state: str = "starting") -> None:
        """Show a startup splash before the session is ready."""
        self._startup_workspace = workspace
        self._startup_state = state
        text = loading_splash_text(
            workspace=workspace,
            state=state,
            frame=SPINNER_FRAMES[self._startup_spinner_index],
        )
        if self._startup_block is None:
            self._startup_block = self._add_block("mira", text, "message startup")
            return
        self._startup_block.update(text)
        self._scroll_to_end()

    def startup_progress(self, state: str) -> None:
        """Update the startup splash status line."""
        if self._startup_block is None:
            self.startup_loading(workspace=self._startup_workspace or ".", state=state)
            return
        self._startup_state = state
        self.startup_loading(workspace=self._startup_workspace, state=state)

    def user_message(self, text: str, *, planning: bool = False) -> None:
        """Append a submitted user message."""
        self._reset_delegation_group()
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
                text = str(event.get("text") or "")
                if text.strip():
                    self._add_block("thinking", Text(text), "message reasoning")
            elif event_type == "tool_call":
                self.tool_call(event["name"], event.get("args", {}), call_id=str(event.get("call_id") or ""))
            elif event_type == "tool_result":
                self.tool_result(event["name"], event["output"], call_id=str(event.get("call_id") or ""))
            elif event_type == "delegation":
                self.delegation_started(event["calls"])
            elif event_type == "subagent":
                if event.get("status") == "DONE":
                    self.subagent_finished(event["name"], event.get("output", ""), event.get("task_input", ""))
                elif event.get("status") == "CANCELLED":
                    self.subagent_cancelled(event["name"], event.get("output", ""), event.get("task_input", ""))
                else:
                    self.subagent_started(event["name"], event.get("task_input", ""))
            elif event_type == "compaction":
                self._add_block("session compacted", self._compaction_text(event), "message summary")
            elif event_type in {"info", "system_error", "interrupted"}:
                kind = {"info": "info", "system_error": "error", "interrupted": "warning"}[event_type]
                self.system_message(event["text"], kind=kind)

    def reasoning_delta(self, delta: str) -> None:
        """Append streamed reasoning text to the current reasoning block."""
        cleaned = re.sub(r"</?[^>]+>", "", delta)
        if not cleaned.strip() and not self._reasoning_text:
            return
        self.hide_waiting()
        self.hide_model_activity()

        if self._reasoning_block is None:
            self._reasoning_text = ""
            self._reasoning_block = self._add_block("thinking", Text(""), "message reasoning")

        self._reasoning_text += cleaned
        self._reasoning_block.update(Text(self._reasoning_text))
        self._scroll_to_end()

    def text_delta(self, delta: str) -> None:
        """Append streamed assistant text to the current response block."""
        delta = normalize_response_delta(self._assistant_text, delta)
        if not delta:
            return
        self.hide_waiting()
        self.hide_model_activity()

        if self._assistant_block is None:
            self._assistant_text = ""
            self._assistant_block = self._add_block("mira", Text(""), "message assistant")

        self._assistant_text += delta
        self._assistant_block.update(Text(self._assistant_text))
        self._scroll_to_end()

    def finish_main(self) -> None:
        """Close the current streamed blocks so the next turn starts fresh."""
        self.hide_model_activity()
        self._assistant_block = None
        self._assistant_text = ""
        self._reasoning_block = None
        self._reasoning_text = ""

    def discard_reasoning(self) -> None:
        """Remove the current streamed reasoning block."""
        if self._reasoning_block is not None:
            self._reasoning_block.remove()
        self._reasoning_block = None
        self._reasoning_text = ""
        self._scroll_to_end()

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Append a system, info, status, warning, or error message."""
        self.hide_model_activity()
        title = "mira" if kind == "startup" else kind
        self._add_block(title, Text(text), f"message {kind}")

    def command_output(self, renderable: Any) -> None:
        """Append command output, including Rich renderables such as tables."""
        self.hide_model_activity()
        self._add_block("output", renderable, "message command")

    def compaction_started(self) -> None:
        """Show that DeepAgents is compacting conversation context."""
        self.hide_waiting()
        self.hide_model_activity()
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
        """Advance the spinner on an active compaction status."""
        if self._compaction_block is None or not self._compaction_running:
            return
        self._compaction_spinner_index = (self._compaction_spinner_index + 1) % len(SPINNER_FRAMES)
        self._compaction_block.update(self._render_compaction())
        self._scroll_to_end()

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """Append a coordinator-level tool call in transcript order."""
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        key = self._tool_update_key(name, call_id)
        block = self._tool_blocks.get(key)
        if block is None:
            widget = self._add_block(f"tool - {name}", Text(""), "message tool-call")
            block = {"name": name, "args": args, "result": "", "widget": widget, "draft": False}
            self._tool_blocks[key] = block
            self._tool_name_queues[name].append(key)
        else:
            block["name"] = name
            block["args"] = args
            block["draft"] = False

        pending = self._take_pending_tool_result(name, call_id)
        if pending:
            block["result"] = pending
        self._update_tool_block(key)

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        """Create or update a live draft of a streamed tool call."""
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        key = self._tool_update_key(name, call_id)
        block = self._tool_blocks.get(key)
        if block is None:
            widget = self._add_block(f"tool - {name}", Text(""), "message tool-call")
            block = {"name": name, "args": args, "result": "", "widget": widget, "draft": True}
            self._tool_blocks[key] = block
            self._tool_name_queues[name].append(key)
        else:
            block["name"] = name
            block["args"] = args
            block["draft"] = True
        self._update_tool_block(key)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Append a coordinator-level tool result in transcript order."""
        if not result:
            return
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        key = self._resolve_tool_key(name, call_id)
        if key is None:
            self._queue_pending_tool_result(name, result, call_id)
            return
        block = self._tool_blocks[key]
        block["result"] = result
        self._update_tool_block(key)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Append a compact task delegation summary."""
        new_calls = self._new_delegation_calls(calls)
        if not new_calls:
            return
        self._delegation_calls.extend(new_calls)
        text = self._render_delegation(self._delegation_calls, draft=False)
        if text is None:
            return

        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        if self._delegation_draft_block is not None:
            self._delegation_draft_block.update(text)
            self._scroll_to_end()
            self._delegation_block = self._delegation_draft_block
            self._delegation_draft_block = None
            return
        if self._delegation_block is not None:
            self._delegation_block.update(text)
            self._scroll_to_end()
            return
        self._delegation_block = self._add_block("task", text, "message delegation")

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        """Create or update a live draft of streamed task delegation input."""
        text = self._render_delegation(calls, draft=True)
        if text is None:
            self.hide_waiting()
            self.hide_model_activity()
            self.finish_main()
            placeholder = Text("preparing subagent tasks...")
            if self._delegation_draft_block is None:
                self._delegation_draft_block = self._add_block("info", placeholder, "message info")
                return
            self._style_block(self._delegation_draft_block, title="info", classes="message info")
            self._delegation_draft_block.update(placeholder)
            self._scroll_to_end()
            return

        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        if self._delegation_draft_block is None:
            self._delegation_draft_block = self._add_block("task", text, "message delegation")
            return
        self._style_block(self._delegation_draft_block, title="task", classes="message delegation")
        self._delegation_draft_block.update(text)
        self._scroll_to_end()

    def start_subagent_live(self) -> None:
        """Reset subagent state for a new delegation group."""
        self._subagent_blocks = {}
        self._subagent_widgets = {}

    def stop_subagent_live(self) -> None:
        """Finalize subagent display."""
        for label in list(self._subagent_blocks):
            self._update_subagent(label)

    def subagents_cancelled(self) -> None:
        """Mark all running subagents as cancelled."""
        for label, block in list(self._subagent_blocks.items()):
            if block.get("status") == "RUNNING":
                block["status"] = "CANCELLED"
                self._update_subagent(label)

    def tick_subagents(self) -> None:
        """Advance the spinner on running subagents."""
        if not self.has_running_subagents():
            return

        self._subagent_spinner_index = (self._subagent_spinner_index + 1) % len(SPINNER_FRAMES)
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
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        subagent = self._subagent_display_label(subagent)
        self._subagent_blocks[subagent] = {
            "request": task_input,
            "status": "RUNNING",
            "output": "",
        }
        widget = self._add_block(f"subagent - {subagent}", self._render_subagent(subagent), "message subagent")
        self._subagent_widgets[subagent] = widget

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Fill request text into an existing running subagent block."""
        if not task_input:
            return
        subagent = self._subagent_display_label(subagent)
        block = self._subagent_blocks.get(subagent)
        if block is None:
            return
        if block.get("request"):
            return
        block["request"] = task_input
        self._update_subagent(subagent)

    def subagent_finished(self, subagent: str, result: str = "", task_input: str = "") -> None:
        """Mark a subagent block as done and attach its final output."""
        self.hide_waiting()
        subagent = self._subagent_display_label(subagent)
        block = self._subagent_blocks.setdefault(
            subagent,
            {
                "request": task_input,
                "status": "RUNNING",
                "output": "",
            },
        )
        if task_input and not block.get("request"):
            block["request"] = task_input
        block["status"] = "DONE"
        block["output"] = result
        self._update_subagent(subagent)

    def subagent_cancelled(self, subagent: str, result: str = "", task_input: str = "") -> None:
        """Mark a subagent block as cancelled."""
        self.hide_waiting()
        subagent = self._subagent_display_label(subagent)
        block = self._subagent_blocks.setdefault(
            subagent,
            {
                "request": task_input,
                "status": "RUNNING",
                "output": "",
            },
        )
        if task_input and not block.get("request"):
            block["request"] = task_input
        block["status"] = "CANCELLED"
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
        self._waiting_block = None
        self._activity_block = None
        self._reset_delegation_group()
        self._startup_block = None
        self._startup_state = "starting"
        self._startup_workspace = ""
        self._startup_spinner_index = 0
        self._subagent_labels = {}
        self._subagent_blocks = {}
        self._subagent_widgets = {}
        self._compaction_block = None
        self._compaction_spinner_index = 0
        self._compaction_running = False
        self._tool_blocks = {}
        self._tool_name_queues = defaultdict(deque)
        self._pending_tool_results_by_id = {}
        self._pending_tool_results_by_name = defaultdict(deque)
        self._subagent_aliases = {}
        for child in list(self.children):
            child.remove()

    def show_waiting(self) -> None:
        """Show the transient thinking status while MIRA is idle."""
        self.hide_model_activity()
        text = self._render_waiting()
        if self._waiting_block is None:
            self._waiting_block = self._add_block("mira", text, "message status")
            return
        self._waiting_block.update(text)
        self._scroll_to_end()

    def hide_waiting(self) -> None:
        """Remove the transient thinking status block."""
        if self._waiting_block is None:
            return
        self._waiting_block.remove()
        self._waiting_block = None

    def model_activity(self, text: str = "preparing tool call...") -> None:
        """Show transient model activity while tool-call JSON is streaming."""
        self.hide_waiting()
        renderable = Text(text, style="bold #DCE6FA")
        if self._activity_block is None:
            self._activity_block = self._add_block("mira", renderable, "message status")
            return
        self._activity_block.update(renderable)
        self._scroll_to_end()

    def hide_model_activity(self) -> None:
        """Remove the transient model-activity status block."""
        if self._activity_block is None:
            return
        self._activity_block.remove()
        self._activity_block = None

    def tick_waiting(self) -> None:
        """Advance the spinner on the transient thinking block."""
        if self._waiting_block is None:
            return
        self._waiting_spinner_index = (self._waiting_spinner_index + 1) % len(SPINNER_FRAMES)
        self._waiting_block.update(self._render_waiting())
        self._scroll_to_end()

    def tick_startup(self) -> None:
        """Advance the spinner on the startup splash."""
        if self._startup_block is None:
            return
        self._startup_spinner_index = (self._startup_spinner_index + 1) % len(SPINNER_FRAMES)
        self._startup_block.update(
            loading_splash_text(
                workspace=self._startup_workspace,
                state=self._startup_state,
                frame=SPINNER_FRAMES[self._startup_spinner_index],
            )
        )
        self._scroll_to_end()

    def truncate(self, value: Any) -> str:
        """Return a compact one-line display string."""
        text = re.sub(r"\s+", " ", str("" if value is None else value)).strip()
        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text
        return text[: self.tool_output_chars].rstrip() + " ... truncated ..."

    def truncate_multiline(self, value: Any) -> str:
        """Return text shortened to the configured display size, preserving line breaks."""
        text = str("" if value is None else value).strip()
        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text
        return text[: self.tool_output_chars].rstrip() + "\n... truncated ..."

    def _add_block(self, title: str, renderable: Any, classes: str) -> Static:
        """Mount one bordered transcript block."""
        block = Static(renderable, classes=classes)
        block.border_title = escape(title)
        self.mount(block)
        self._scroll_to_end()
        return block

    def _style_block(self, block: Static, *, title: str, classes: str) -> None:
        """Update a transcript block's title and style classes in place."""
        block.border_title = escape(title)
        block.set_classes(classes)

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

    def _new_delegation_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_calls = []
        for call in calls:
            key = self._delegation_key(call)
            if key in self._delegation_keys:
                continue
            self._delegation_keys.add(key)
            new_calls.append(call)
        return new_calls

    def _reset_delegation_group(self) -> None:
        self._delegation_block = None
        self._delegation_draft_block = None
        self._delegation_calls = []
        self._delegation_keys = set()

    def _delegation_key(self, call: dict[str, Any]) -> tuple[str, str]:
        call_id = str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or "")
        if call_id:
            return ("id", call_id)
        raw_args = call.get("args", call.get("input", {}))
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (TypeError, json.JSONDecodeError):
                raw_args = {"raw": raw_args}
        args = raw_args if isinstance(raw_args, dict) else {"raw": str(raw_args)}
        return (
            "request",
            json.dumps(
                [
                    str(call.get("name") or call.get("tool_name") or "task"),
                    str(args.get("description") or ""),
                    str(args.get("subagent_type") or ""),
                ],
                sort_keys=True,
            ),
        )

    def _render_delegation(self, calls: list[dict[str, Any]], *, draft: bool) -> Text | None:
        """Render a task delegation summary or live draft."""
        descriptions, errors = self._delegation_details(calls)
        if not descriptions and not errors:
            return None
        if draft and not descriptions:
            return None

        text = Text()
        label = "subagent" if len(descriptions) == 1 else "subagents"
        if descriptions:
            verb = "preparing" if draft else "delegating to"
            text.append(f"{verb} {len(descriptions)} {label}\n", style="bold yellow")
        for description in descriptions:
            text.append("request: ", style="bold cyan")
            text.append(self.truncate(description) + "\n")
        if draft:
            for _ in errors:
                text.append("request: ", style="bold cyan")
                text.append("drafting request...\n", style="dim")
        if not draft:
            for error in errors:
                text.append(f"failed: {error}\n", style="red")
        return text

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
            text.append(f"{SPINNER_FRAMES[self._subagent_spinner_index]} RUNNING", style="bold yellow")
        elif status == "CANCELLED":
            text.append("CANCELLED", style="bold yellow")
        else:
            text.append("DONE", style="bold green")

        if block.get("output"):
            text.append("\n\noutput:\n", style="bold cyan")
            text.append(self.truncate_multiline(block["output"]), style="dim")

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
        return generate_slug(fallback=self._fallback_suffixes)

    def _subagent_display_label(self, label: str) -> str:
        """Return a stable label, adding a nickname if the caller omitted one."""
        if "[" in label and "]" in label:
            return label
        if label not in self._subagent_aliases:
            self._subagent_aliases[label] = f"{label} [{self._next_suffix()}]"
        return self._subagent_aliases[label]

    def _render_waiting(self) -> Text:
        text = Text()
        text.append(f"{SPINNER_FRAMES[self._waiting_spinner_index]} ", style="bold yellow")
        text.append("working...", style="bold yellow")
        return text

    def _render_compaction(self) -> Text:
        text = Text()
        text.append(f"{SPINNER_FRAMES[self._compaction_spinner_index]} ", style="bold yellow")
        text.append("compacting context...", style="bold yellow")
        return text

    def _tool_key(self, name: str, call_id: str = "") -> str:
        if call_id:
            return f"id:{call_id}"
        return f"name:{name}:{next(self._tool_sequence)}"

    def _tool_update_key(self, name: str, call_id: str = "") -> str:
        if call_id:
            return f"id:{call_id}"
        draft_key = self._oldest_draft_tool_key(name)
        return draft_key or self._tool_key(name, call_id)

    def _oldest_draft_tool_key(self, name: str) -> str | None:
        queue = self._tool_name_queues.get(name)
        if not queue:
            return None
        for key in queue:
            block = self._tool_blocks.get(key)
            if block is not None and block.get("draft") and not block.get("result"):
                return key
        return None

    def _resolve_tool_key(self, name: str, call_id: str = "") -> str | None:
        if call_id:
            key = f"id:{call_id}"
            if key in self._tool_blocks:
                self._remove_tool_queue_key(name, key)
                return key
            return None

        queue = self._tool_name_queues.get(name)
        if not queue:
            return None
        while queue:
            key = queue.popleft()
            block = self._tool_blocks.get(key)
            if block is not None and not block.get("result"):
                return key
        return None

    def _queue_pending_tool_result(self, name: str, result: str, call_id: str = "") -> None:
        if call_id:
            self._pending_tool_results_by_id[call_id] = result
            return
        self._pending_tool_results_by_name[name].append(result)

    def _take_pending_tool_result(self, name: str, call_id: str = "") -> str:
        if call_id:
            result = self._pending_tool_results_by_id.pop(call_id, "")
            if result:
                return result
        queue = self._pending_tool_results_by_name.get(name)
        if not queue:
            return ""
        return queue.popleft()

    def _remove_tool_queue_key(self, name: str, key: str) -> None:
        queue = self._tool_name_queues.get(name)
        if not queue:
            return
        self._tool_name_queues[name] = deque(item for item in queue if item != key)

    def _update_tool_block(self, key: str) -> None:
        block = self._tool_blocks[key]
        text = Text()
        label = "draft" if block.get("draft") else "call"
        text.append(f"{label}: ", style="bold cyan")
        text.append(self.truncate(block["args"]))
        if block.get("result"):
            text.append("\n")
            text.append("-" * 12, style="dim")
            text.append("\noutput:\n", style="bold cyan")
            text.append(self.truncate_multiline(block["result"]), style="dim")
        block["widget"].update(text)
        self._scroll_to_end()
