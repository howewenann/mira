"""Scrollable chat output for the Textual TUI."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime
from itertools import count
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, DescendantFocus, Key
from textual.widgets import Button, Static

from runtime.output_events import normalize_response_delta
from runtime.rubric_events import rubric_result_text
from session.context import normalize_events
from ui.names import generate_slug
from ui.terminal_colors import RUBRIC_BODY_COLOR, RUBRIC_HEADER_COLOR
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
        self._startup_loading = False
        self._subagent_labels: dict[int, str] = {}
        self._subagent_blocks: dict[str, dict[str, str]] = {}
        self._subagent_widgets: dict[str, Static] = {}
        self._compaction_block: Static | None = None
        self._compaction_spinner_index = 0
        self._compaction_running = False
        self._plan_widgets: dict[str, PlanBubble] = {}
        self._proposal_widgets: dict[str, ProposalBubble] = {}
        self._rubric_widgets: dict[tuple[str, int], Static] = {}
        self._rubric_evaluations: dict[tuple[str, int], dict[str, Any]] = {}
        self._fallback_suffixes = count(1)
        self._waiting_spinner_index = 0
        self._waiting_label = "working..."
        self._subagent_spinner_index = 0
        self._tool_sequence = count(1)
        self._tool_blocks: dict[str, dict[str, Any]] = {}
        self._tool_name_queues: dict[str, deque[str]] = defaultdict(deque)
        self._pending_tool_results_by_id: dict[str, str] = {}
        self._pending_tool_results_by_name: dict[str, deque[str]] = defaultdict(deque)
        self._subagent_aliases: dict[str, deque[str]] = {}

    def startup(self, *, model_name: str, session_id: str, workspace: str) -> None:
        """Show session metadata when the app opens."""
        self._startup_loading = False
        text = splash_text(model_name=model_name, session_id=session_id, workspace=workspace)
        if self._startup_block is None:
            self._startup_block = self._add_block("mira", text, "message startup")
            return
        self._style_block(self._startup_block, title="mira", classes="message startup")
        self._startup_block.update(text)
        self._scroll_to_end()

    def startup_loading(self, *, workspace: str, state: str = "starting") -> None:
        """Show a startup splash before the session is ready."""
        self._startup_loading = True
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

    def timestamped_user_message(self, text: str, *, planning: bool = False, created_at: str = "") -> None:
        """Append a persisted user message with its session timestamp."""
        self._reset_delegation_group()
        title = "you (plan)" if planning else "you"
        self._add_block(title, Text(text), "message user", created_at=created_at)

    def assistant_message(self, text: str, *, created_at: str = "") -> None:
        """Append a completed assistant message."""
        self._add_block("mira", Text(text), "message assistant", created_at=created_at)

    def restore_session(self, session: dict[str, Any]) -> None:
        """Replay persisted visible session events."""
        self.finish_main()
        for event in normalize_events(session.get("events")):
            event_type = event["type"]
            created_at = str(event.get("created_at") or "")
            if event_type == "user":
                self.timestamped_user_message(event["text"], planning=event.get("mode") == "planning", created_at=created_at)
            elif event_type == "assistant":
                self.assistant_message(event["text"], created_at=created_at)
            elif event_type == "reasoning":
                text = str(event.get("text") or "")
                if text.strip():
                    self._add_block("thinking", Text(text), "message reasoning", created_at=created_at)
            elif event_type == "tool_call":
                self.tool_call(
                    event["name"],
                    event.get("args", {}),
                    call_id=str(event.get("call_id") or ""),
                    created_at=created_at,
                )
            elif event_type == "tool_result":
                self.tool_result(
                    event["name"],
                    event["output"],
                    call_id=str(event.get("call_id") or ""),
                    created_at=created_at,
                )
            elif event_type == "delegation":
                self.delegation_started(event["calls"], created_at=created_at)
            elif event_type == "subagent":
                if event.get("status") == "DONE":
                    self.subagent_finished(
                        event["name"],
                        event.get("output", ""),
                        event.get("task_input", ""),
                        origin=str(event.get("origin") or ""),
                        created_at=created_at,
                    )
                elif event.get("status") == "CANCELLED":
                    self.subagent_cancelled(
                        event["name"],
                        event.get("output", ""),
                        event.get("task_input", ""),
                        origin=str(event.get("origin") or ""),
                        created_at=created_at,
                    )
                else:
                    self.subagent_started(
                        event["name"],
                        event.get("task_input", ""),
                        origin=str(event.get("origin") or ""),
                        created_at=created_at,
                    )
            elif event_type == "compaction":
                self._add_block("session compacted", self._compaction_text(event), "message summary", created_at=created_at)
            elif event_type == "plan":
                self.present_plan(
                    event["plan"],
                    active=False,
                    status=str(event.get("status") or "resolved"),
                    created_at=created_at,
                )
            elif event_type == "proposal":
                self.present_proposal(
                    event["proposal"],
                    active=False,
                    status=str(event.get("status") or "resolved"),
                    created_at=created_at,
                )
            elif event_type == "rubric":
                self.rubric_evaluation_finished(
                    event["evaluation"],
                    int(event.get("max_iterations") or 1),
                    created_at=created_at,
                )
            elif event_type in {"info", "system_error", "interrupted"}:
                kind = {"info": "info", "system_error": "error", "interrupted": "warning"}[event_type]
                self.system_message(event["text"], kind=kind, created_at=created_at)

    def reasoning_delta(self, delta: str, *, created_at: str = "") -> None:
        """Append streamed reasoning text to the current reasoning block."""
        cleaned = re.sub(r"</?[^>]+>", "", delta)
        if not cleaned.strip() and not self._reasoning_text:
            return
        self.hide_waiting()
        self.hide_model_activity()
        self._close_assistant_phase()

        if self._reasoning_block is None:
            self._reasoning_text = ""
            self._reasoning_block = self._add_block("thinking", Text(""), "message reasoning", created_at=created_at)
        elif created_at:
            self._style_block(self._reasoning_block, title="thinking", classes="message reasoning", created_at=created_at)

        self._reasoning_text += cleaned
        self._reasoning_block.update(Text(self._reasoning_text))
        self._scroll_to_end()

    def text_delta(self, delta: str, *, created_at: str = "") -> None:
        """Append streamed assistant text to the current response block."""
        delta = normalize_response_delta(self._assistant_text, delta)
        if not delta:
            return
        self.hide_waiting()
        self.hide_model_activity()
        self._close_reasoning_phase()

        if self._assistant_block is None:
            self._assistant_text = ""
            self._assistant_block = self._add_block("mira", Text(""), "message assistant", created_at=created_at)

        self._assistant_text += delta
        self._assistant_block.update(Text(self._assistant_text))
        self._scroll_to_end()

    def finish_main(self) -> None:
        """Close the current streamed blocks so the next turn starts fresh."""
        self.hide_model_activity()
        self._close_assistant_phase()
        self._close_reasoning_phase()

    def finish_stream_phase(self) -> None:
        """Close the current model-message phase without ending the turn."""
        self._close_assistant_phase()
        self._close_reasoning_phase()

    def _close_assistant_phase(self) -> None:
        self._assistant_block = None
        self._assistant_text = ""

    def _close_reasoning_phase(self) -> None:
        self._reasoning_block = None
        self._reasoning_text = ""

    def finish_turn(self, *, cancelled: bool = False) -> None:
        """Close live turn state without clearing persisted transcript history."""
        self.finish_main()
        self.hide_waiting()
        self._discard_delegation_draft()
        if cancelled:
            self.subagents_cancelled()
            self._discard_tool_drafts()
            self._cancel_compaction()
            self._reset_delegation_group()

    def discard_reasoning(self) -> None:
        """Remove the current streamed reasoning block."""
        if self._reasoning_block is not None:
            self._reasoning_block.remove()
        self._reasoning_block = None
        self._reasoning_text = ""
        self._scroll_to_end()

    def system_message(self, text: str, *, kind: str = "system", created_at: str = "") -> None:
        """Append a system, info, status, warning, or error message."""
        self.finish_stream_phase()
        self.hide_model_activity()
        title = "mira" if kind == "startup" else kind
        self._add_block(title, Text(text), f"message {kind}", created_at=created_at)

    def command_output(self, renderable: Any) -> None:
        """Append command output, including Rich renderables such as tables."""
        self.finish_stream_phase()
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

    def compaction_finished(self, message: str = "context compacted", *, success: bool = True) -> None:
        """Mark the compaction status as complete."""
        if self._compaction_block is None:
            return
        self._compaction_running = False
        self._compaction_block.update(Text(message, style="bold green" if success else "dim"))
        self._compaction_block = None
        self._scroll_to_end()

    def tick_compaction(self) -> None:
        """Advance the spinner on an active compaction status."""
        if self._compaction_block is None or not self._compaction_running:
            return
        self._compaction_spinner_index = (self._compaction_spinner_index + 1) % len(SPINNER_FRAMES)
        self._compaction_block.update(self._render_compaction())
        self._scroll_to_end()

    def tool_call(self, name: str, args: Any, call_id: str = "", *, created_at: str = "") -> None:
        """Append a coordinator-level tool call in transcript order."""
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        key = self._tool_update_key(name, call_id)
        block = self._tool_blocks.get(key)
        if block is None:
            widget = self._add_block(f"tool - {name}", Text(""), "message tool-call", created_at=created_at)
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

    def tool_result(self, name: str, result: str, call_id: str = "", *, created_at: str = "") -> None:
        """Append a coordinator-level tool result in transcript order."""
        if not result:
            return
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        key = self._resolve_tool_key(name, call_id)
        if key is None:
            self._queue_pending_tool_result(name, result, call_id, created_at=created_at)
            return
        block = self._tool_blocks[key]
        block["result"] = result
        self._update_tool_block(key)

    def completed_tool_result(self, name: str, result: str, call_id: str = "", *, created_at: str = "") -> None:
        """Update an existing tool block without disturbing active model output."""
        if not result:
            return
        key = self._resolve_tool_key(name, call_id)
        if key is None:
            self._queue_pending_tool_result(name, result, call_id, created_at=created_at)
            return
        block = self._tool_blocks[key]
        block["result"] = result
        self._update_tool_block(key, scroll=False)

    def delegation_started(self, calls: list[dict[str, Any]], *, created_at: str = "") -> None:
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
        self._delegation_block = self._add_block("task", text, "message delegation", created_at=created_at)

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
        self.discard_delegation_summary()
        self._subagent_blocks = {}
        self._subagent_widgets = {}
        self._subagent_aliases = {}

    def discard_delegation_summary(self) -> None:
        """Remove the current live delegation summary from the transcript."""
        blocks = [self._delegation_draft_block, self._delegation_block]
        removed = False
        for block in blocks:
            if block is None or block not in self.children:
                continue
            block.remove()
            removed = True
        self._reset_delegation_group()
        if removed:
            self._scroll_to_end()

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

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        created_at: str = "",
    ) -> None:
        """Create or replace the block for a running subagent."""
        self.hide_waiting()
        self.hide_model_activity()
        self.finish_main()
        subagent = self._new_subagent_display_label(subagent)
        self._subagent_blocks[subagent] = {
            "request": task_input,
            "status": "RUNNING",
            "output": "",
            "origin": origin,
        }
        widget = self._add_block(
            self._subagent_title(subagent),
            self._render_subagent(subagent),
            "message subagent",
            created_at=created_at,
        )
        self._subagent_widgets[subagent] = widget

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Fill request text into an existing running subagent block."""
        if not task_input:
            return
        subagent = self._resolve_subagent_display_label(subagent, prefer_blank_request=True)
        block = self._subagent_blocks.get(subagent)
        if block is None:
            return
        if block.get("request"):
            return
        block["request"] = task_input
        block["origin"] = ""
        self._update_subagent(subagent)

    def subagent_finished(
        self,
        subagent: str,
        result: str = "",
        task_input: str = "",
        *,
        origin: str = "",
        created_at: str = "",
    ) -> None:
        """Mark a subagent block as done and attach its final output."""
        self.hide_waiting()
        subagent = self._resolve_subagent_display_label(subagent, consume=True, create=True)
        block = self._subagent_blocks.setdefault(
            subagent,
            {
                "request": task_input,
                "status": "RUNNING",
                "output": "",
                "origin": origin,
            },
        )
        if task_input and not block.get("request"):
            block["request"] = task_input
        if origin and not block.get("origin"):
            block["origin"] = origin
        block["status"] = "DONE"
        block["output"] = result
        self._update_subagent(subagent, created_at=created_at)

    def subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        task_input: str = "",
        *,
        origin: str = "",
        created_at: str = "",
    ) -> None:
        """Mark a subagent block as cancelled."""
        self.hide_waiting()
        subagent = self._resolve_subagent_display_label(subagent, consume=True, create=True)
        block = self._subagent_blocks.setdefault(
            subagent,
            {
                "request": task_input,
                "status": "RUNNING",
                "output": "",
                "origin": origin,
            },
        )
        if task_input and not block.get("request"):
            block["request"] = task_input
        if origin and not block.get("origin"):
            block["origin"] = origin
        block["status"] = "CANCELLED"
        block["output"] = result
        self._update_subagent(subagent, created_at=created_at)

    def present_plan(
        self,
        plan: dict[str, Any],
        *,
        active: bool,
        status: str = "pending",
        created_at: str = "",
    ) -> None:
        """Append a structured plan bubble."""
        self.finish_main()
        plan_id = str(plan.get("id") or "")
        bubble = PlanBubble(plan, active=active, status=status)
        if timestamp := timestamp_text(created_at):
            bubble.border_subtitle = escape(timestamp)
        self.mount(bubble)
        if plan_id:
            self._plan_widgets[plan_id] = bubble
        self._scroll_to_end()

    def resolve_plan(self, plan_id: str, status: str) -> None:
        """Mark an existing plan bubble as resolved."""
        bubble = self._plan_widgets.get(plan_id)
        if bubble is not None:
            bubble.resolve(status)
            self._scroll_to_end()

    def present_proposal(
        self,
        value: dict[str, Any],
        *,
        active: bool,
        status: str = "pending",
        created_at: str = "",
    ) -> None:
        """Append an explicit goal or rubric-enabled plan proposal."""
        self.finish_main()
        proposal_id = str(value.get("id") or "")
        bubble = ProposalBubble(value, active=active, status=status)
        if timestamp := timestamp_text(created_at):
            bubble.border_subtitle = escape(timestamp)
        self.mount(bubble)
        if proposal_id:
            self._proposal_widgets[proposal_id] = bubble
        self._scroll_to_end()

    def resolve_proposal(self, proposal_id: str, status: str) -> None:
        """Mark an existing goal proposal as resolved."""
        bubble = self._proposal_widgets.get(proposal_id)
        if bubble is not None:
            bubble.resolve(status)
            self._scroll_to_end()

    def rubric_evaluation_started(self, run_id: str, pass_number: int, max_iterations: int) -> None:
        """Show immediate rubric grading activity."""
        self.finish_main()
        key = (run_id, pass_number)
        body = Text(
            f"Reviewing completion criteria · pass {pass_number} of {max_iterations}…",
            style=RUBRIC_BODY_COLOR,
        )
        widget = self._add_block("rubric review", body, "message rubric")
        self._rubric_widgets[key] = widget

    def rubric_evaluation_finished(
        self,
        evaluation: dict[str, Any],
        max_iterations: int,
        *,
        created_at: str = "",
    ) -> None:
        """Replace live rubric activity with a concise evaluation result."""
        pass_number = int(evaluation.get("iteration") or 0) + 1
        key = (str(evaluation.get("grading_run_id") or ""), pass_number)
        self._rubric_evaluations[key] = dict(evaluation)
        widget = self._rubric_widgets.get(key)
        body = self._render_rubric(evaluation, max_iterations)
        if widget is None:
            widget = self._add_block("rubric review", body, "message rubric", created_at=created_at)
            self._rubric_widgets[key] = widget
            return
        if timestamp := timestamp_text(created_at):
            widget.border_subtitle = escape(timestamp)
        widget.update(body)
        self._scroll_to_end()

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:
        """Reconcile a streamed verdict with the completed checkpoint status."""
        key = (run_id, pass_number)
        evaluation = self._rubric_evaluations.get(key)
        if evaluation is None:
            return
        evaluation["result"] = status
        widget = self._rubric_widgets.get(key)
        if widget is not None:
            widget.update(self._render_rubric(evaluation, max_iterations))
            self._scroll_to_end()

    def _render_rubric(self, evaluation: dict[str, Any], max_iterations: int) -> Text:
        text = Text()
        lines = rubric_result_text(evaluation, max_iterations).splitlines()
        for index, line in enumerate(lines):
            if index:
                text.append("\n")
            if line.startswith("- "):
                text.append("✗ ", style="bold red")
                text.append(line[2:], style=RUBRIC_BODY_COLOR)
            elif index == 0:
                text.append(line, style=f"bold {RUBRIC_HEADER_COLOR}")
            else:
                text.append(line, style=RUBRIC_BODY_COLOR)
        return text

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
        self._startup_loading = False
        self._subagent_labels = {}
        self._subagent_blocks = {}
        self._subagent_widgets = {}
        self._compaction_block = None
        self._compaction_spinner_index = 0
        self._compaction_running = False
        self._plan_widgets = {}
        self._proposal_widgets = {}
        self._rubric_widgets = {}
        self._rubric_evaluations = {}
        self._tool_blocks = {}
        self._tool_name_queues = defaultdict(deque)
        self._pending_tool_results_by_id = {}
        self._pending_tool_results_by_name = defaultdict(deque)
        self._subagent_aliases = {}
        for child in list(self.children):
            child.remove()

    def show_waiting(self, label: str = "working...") -> None:
        """Show the transient thinking status while MIRA is idle."""
        self.hide_model_activity()
        self._waiting_label = label
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
        self.finish_stream_phase()
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
        if self._startup_block is None or not self._startup_loading:
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

    def _add_block(self, title: str, renderable: Any, classes: str, *, created_at: str = "") -> Static:
        """Mount one bordered transcript block."""
        block = Static(renderable, classes=classes)
        block.border_title = escape(title)
        if timestamp := timestamp_text(created_at):
            block.border_subtitle = escape(timestamp)
        self.mount(block)
        self._scroll_to_end()
        return block

    def _style_block(self, block: Static, *, title: str, classes: str, created_at: str = "") -> None:
        """Update a transcript block's title and style classes in place."""
        block.border_title = escape(title)
        if timestamp := timestamp_text(created_at):
            block.border_subtitle = escape(timestamp)
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

    def _discard_delegation_draft(self) -> None:
        if self._delegation_draft_block is not None:
            self._delegation_draft_block.remove()
            self._delegation_draft_block = None

    def _discard_tool_drafts(self) -> None:
        draft_keys = [
            key
            for key, block in self._tool_blocks.items()
            if block.get("draft") and not block.get("result")
        ]
        for key in draft_keys:
            block = self._tool_blocks.pop(key, None)
            if block is None:
                continue
            widget = block.get("widget")
            if widget is not None:
                widget.remove()
        if draft_keys:
            draft_key_set = set(draft_keys)
            for name, queue in list(self._tool_name_queues.items()):
                self._tool_name_queues[name] = deque(key for key in queue if key not in draft_key_set)

    def _cancel_compaction(self) -> None:
        if self._compaction_block is None or not self._compaction_running:
            return
        self._compaction_running = False
        self._compaction_block.update(Text("context compaction cancelled", style="bold yellow"))
        self._compaction_block = None
        self._scroll_to_end()

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

    def _update_subagent(self, label: str, *, created_at: str = "") -> None:
        """Update an existing subagent widget."""
        widget = self._subagent_widgets.get(label)
        if widget is None:
            widget = self._add_block(
                self._subagent_title(label),
                self._render_subagent(label),
                "message subagent",
                created_at=created_at,
            )
            self._subagent_widgets[label] = widget
            return
        if timestamp := timestamp_text(created_at):
            widget.border_subtitle = escape(timestamp)
        widget.border_title = escape(self._subagent_title(label))
        widget.update(self._render_subagent(label))
        self._scroll_to_end()

    def _next_suffix(self) -> str:
        """Return a short cute suffix for delegated workers."""
        return generate_slug(fallback=self._fallback_suffixes)

    def _new_subagent_display_label(self, label: str, *, track: bool = True) -> str:
        """Return a fresh visible label for a newly started subagent."""
        if self._has_subagent_suffix(label):
            return label
        display = f"{label} [{self._next_suffix()}]"
        if track:
            self._subagent_aliases.setdefault(label, deque()).append(display)
        return display

    def _resolve_subagent_display_label(
        self,
        label: str,
        *,
        consume: bool = False,
        create: bool = False,
        prefer_blank_request: bool = False,
    ) -> str:
        """Resolve an unsuffixed lifecycle update to its active display label."""
        if self._has_subagent_suffix(label):
            return label

        queue = self._subagent_aliases.get(label)
        if queue:
            fallback = ""
            for candidate in list(queue):
                block = self._subagent_blocks.get(candidate)
                if block is None:
                    continue
                if block.get("status") != "RUNNING":
                    continue
                if prefer_blank_request and block.get("request"):
                    fallback = fallback or candidate
                    continue
                if consume:
                    self._remove_subagent_alias(label, candidate)
                return candidate
            if fallback:
                return fallback

        if label in self._subagent_blocks:
            return label
        if create:
            return self._new_subagent_display_label(label, track=False)
        return label

    def _remove_subagent_alias(self, label: str, display: str) -> None:
        """Forget a consumed display label from an unsuffixed subagent queue."""
        queue = self._subagent_aliases.get(label)
        if not queue:
            return
        self._subagent_aliases[label] = deque(candidate for candidate in queue if candidate != display)

    def _has_subagent_suffix(self, label: str) -> bool:
        """Return whether a subagent label already includes a generated suffix."""
        return bool(re.search(r"\[[^\]]+\]\s*$", label))

    def _subagent_title(self, label: str) -> str:
        """Return a visible title for a subagent block."""
        return f"subagent - {label}"

    def _render_waiting(self) -> Text:
        text = Text()
        text.append(f"{SPINNER_FRAMES[self._waiting_spinner_index]} ", style="bold yellow")
        text.append(self._waiting_label, style="bold yellow")
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

    def _queue_pending_tool_result(self, name: str, result: str, call_id: str = "", *, created_at: str = "") -> None:
        value = json.dumps({"result": result, "created_at": created_at})
        if call_id:
            self._pending_tool_results_by_id[call_id] = value
            return
        self._pending_tool_results_by_name[name].append(value)

    def _take_pending_tool_result(self, name: str, call_id: str = "") -> str:
        if call_id:
            result = self._pending_tool_results_by_id.pop(call_id, "")
            if result:
                return pending_result_text(result)
        queue = self._pending_tool_results_by_name.get(name)
        if not queue:
            return ""
        return pending_result_text(queue.popleft())

    def _remove_tool_queue_key(self, name: str, key: str) -> None:
        queue = self._tool_name_queues.get(name)
        if not queue:
            return
        self._tool_name_queues[name] = deque(item for item in queue if item != key)

    def _update_tool_block(self, key: str, *, scroll: bool = True) -> None:
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
        if scroll:
            self._scroll_to_end()


def timestamp_text(value: Any) -> str:
    """Format a persisted event timestamp for chat bubble titles."""
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return ""
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def pending_result_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict):
        return str(payload.get("result") or "")
    return value


class PlanActionButton(Button):
    """Compact plan control with shortcuts matching its visible label."""

    SHORTCUT_ACTIONS = {"i": "implement", "r": "revise", "d": "discard"}

    def on_key(self, event: Key) -> None:
        bubble = self._plan_bubble()
        if event.key in {"left", "right"} and bubble is not None:
            event.stop()
            bubble.focus_action_offset(self, -1 if event.key == "left" else 1)
            return

        if event.key == "enter":
            event.stop()
            self.press()
            return

        action = self.SHORTCUT_ACTIONS.get(event.key.lower())
        if action is None:
            return

        bubble = self._plan_bubble()
        if bubble is None:
            return

        event.stop()
        bubble.query_one(f"#plan-{action}-{bubble.plan_id}", Button).press()

    def _plan_bubble(self) -> PlanBubble | None:
        """Return the plan bubble containing this action button."""
        bubble: Any = self.parent
        while bubble is not None and not isinstance(bubble, PlanBubble):
            bubble = bubble.parent
        return bubble


class PlanBubble(Vertical):
    """Structured plan transcript block with optional action buttons."""

    def __init__(self, plan: dict[str, Any], *, active: bool, status: str = "pending") -> None:
        super().__init__(classes="message plan")
        self.plan = plan
        self.status = status
        self.active = active
        self._last_action_id = f"plan-implement-{self.plan_id}"
        self.border_title = "plan" if status == "pending" else f"plan - {status}"

    def compose(self) -> Any:
        yield Static(self._render_plan(), classes="plan-body")
        if self.active:
            with Horizontal(classes="plan-actions"):
                yield PlanActionButton(
                    "Implement (i)",
                    id=f"plan-implement-{self.plan_id}",
                    classes="plan-action",
                    compact=True,
                )
                yield PlanActionButton(
                    "Revise (r)", id=f"plan-revise-{self.plan_id}", classes="plan-action", compact=True
                )
                yield PlanActionButton(
                    "Discard (d)", id=f"plan-discard-{self.plan_id}", classes="plan-action", compact=True
                )

    def on_mount(self) -> None:
        """Focus the first action after an active plan is fully mounted."""
        if self.active:
            self.call_after_refresh(self.query_one(f"#plan-implement-{self.plan_id}", Button).focus)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember the most recently focused plan action."""
        if isinstance(event.widget, PlanActionButton) and event.widget.id:
            self._last_action_id = event.widget.id

    def on_click(self, event: Click) -> None:
        """Restore the last plan action when the active bubble is clicked."""
        if not self.active or isinstance(event.widget, Button):
            return
        event.stop()
        self.query_one(f"#{self._last_action_id}", Button).focus()

    def focus_action_offset(self, focused: Button, offset: int) -> None:
        """Move focus through the one-row plan actions with wrapping."""
        buttons = list(self.query(PlanActionButton))
        if not buttons or focused not in buttons:
            return
        buttons[(buttons.index(focused) + offset) % len(buttons)].focus()

    @property
    def plan_id(self) -> str:
        return str(self.plan.get("id") or "plan")

    def resolve(self, status: str) -> None:
        """Disable actions and update the displayed status."""
        self.status = status
        self.active = False
        self.border_title = f"plan - {status}"
        for button in self.query(Button):
            button.disabled = True
            button.display = False
        body = self.query_one(".plan-body", Static)
        body.update(self._render_plan())

    def _render_plan(self) -> Text:
        text = Text()
        text.append(str(self.plan.get("title") or "Implementation Plan"), style="bold")
        text.append("\n\n")
        for heading, key in (
            ("Summary", "summary"),
            ("Key Changes", "key_changes"),
            ("Test Plan", "test_plan"),
            ("Assumptions", "assumptions"),
        ):
            items = self.plan.get(key)
            if not isinstance(items, list) or not items:
                continue
            text.append(f"{heading}\n", style="bold cyan")
            for item in items:
                text.append(f"- {item}\n")
            text.append("\n")
        if self.status != "pending":
            text.append(f"Status: {self.status}", style="bold yellow")
        return text


class ProposalActionButton(Button):
    """Compact proposal control with the same review shortcuts as plans."""

    SHORTCUT_ACTIONS = {"i": "implement", "r": "revise", "d": "discard"}

    def on_key(self, event: Key) -> None:
        bubble = self._proposal_bubble()
        if event.key in {"left", "right"} and bubble is not None:
            event.stop()
            bubble.focus_action_offset(self, -1 if event.key == "left" else 1)
            return
        if event.key == "enter":
            event.stop()
            self.press()
            return
        action = self.SHORTCUT_ACTIONS.get(event.key.lower())
        if action is None or bubble is None:
            return
        event.stop()
        bubble.query_one(f"#proposal-{action}-{bubble.proposal_id}", Button).press()

    def _proposal_bubble(self) -> ProposalBubble | None:
        bubble: Any = self.parent
        while bubble is not None and not isinstance(bubble, ProposalBubble):
            bubble = bubble.parent
        return bubble


class ProposalBubble(Vertical):
    """Goal or rubric-enabled plan with separately rendered criteria."""

    def __init__(self, value: dict[str, Any], *, active: bool, status: str = "pending") -> None:
        complete = isinstance(value.get("plan"), dict)
        kind = "plan-review" if complete else "goal"
        super().__init__(classes=f"message proposal {kind}")
        self.value = value
        self.status = status
        self.active = active
        self._last_action_id = f"proposal-implement-{self.proposal_id}"
        title = "plan + goal" if complete else "legacy goal"
        self.border_title = title if status == "pending" else f"{title} - {status}"

    @property
    def proposal_id(self) -> str:
        return str(self.value.get("id") or "proposal")

    def compose(self) -> Any:
        yield Static(self._render_proposal(), classes="proposal-body")
        if self.active:
            with Horizontal(classes="proposal-actions"):
                for action, label in (
                    ("implement", "Implement (i)"),
                    ("revise", "Revise (r)"),
                    ("discard", "Discard (d)"),
                ):
                    yield ProposalActionButton(
                        label,
                        id=f"proposal-{action}-{self.proposal_id}",
                        classes="proposal-action",
                        compact=True,
                    )

    def on_mount(self) -> None:
        if self.active:
            self.call_after_refresh(self.query_one(f"#proposal-implement-{self.proposal_id}", Button).focus)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        if isinstance(event.widget, ProposalActionButton) and event.widget.id:
            self._last_action_id = event.widget.id

    def on_click(self, event: Click) -> None:
        if not self.active or isinstance(event.widget, Button):
            return
        event.stop()
        self.query_one(f"#{self._last_action_id}", Button).focus()

    def focus_action_offset(self, focused: Button, offset: int) -> None:
        buttons = list(self.query(ProposalActionButton))
        if buttons and focused in buttons:
            buttons[(buttons.index(focused) + offset) % len(buttons)].focus()

    def resolve(self, status: str) -> None:
        self.status = status
        self.active = False
        base = "plan + goal" if isinstance(self.value.get("plan"), dict) else "legacy goal"
        self.border_title = f"{base} - {status}"
        for button in self.query(Button):
            button.disabled = True
            button.display = False
        self.query_one(".proposal-body", Static).update(self._render_proposal())

    def _render_proposal(self) -> Text:
        text = Text()
        plan = self.value.get("plan")
        if isinstance(plan, dict):
            text.append("PLAN\n", style="bold cyan")
            text.append(str(plan.get("title") or "Implementation Plan"), style="bold")
            text.append("\n\n")
            for heading, key in (
                ("Summary", "summary"),
                ("Key Changes", "key_changes"),
                ("Test Plan", "test_plan"),
                ("Assumptions", "assumptions"),
            ):
                items = plan.get(key)
                if not isinstance(items, list) or not items:
                    continue
                text.append(f"{heading}\n", style="bold cyan")
                for item in items:
                    text.append(f"- {item}\n")
                text.append("\n")
        else:
            text.append("Original objective\n", style=f"bold {RUBRIC_HEADER_COLOR}")
            text.append(str(self.value.get("original_objective") or ""), style=RUBRIC_BODY_COLOR)
            text.append("\n\n")

        text.append("GOAL / DEFINITION OF DONE\n", style=f"bold {RUBRIC_HEADER_COLOR}")
        text.append(str(self.value.get("criteria") or ""), style=RUBRIC_BODY_COLOR)
        text.append("\n\n")
        text.append(
            f"Rubric iterations: {int(self.value.get('rubric_iterations') or 3)}",
            style=f"bold {RUBRIC_HEADER_COLOR}",
        )
        if self.status != "pending":
            text.append(f"\n\nStatus: {self.status}", style="bold yellow")
        return text
