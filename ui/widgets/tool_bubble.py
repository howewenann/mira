"""Expandable tool-call transcript bubble for the Textual TUI."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Collapsible, Static, TextArea

from runtime.rubric_events import elapsed_ms, format_elapsed
from ui.spinners import SPINNER_FRAMES
from ui.terminal_colors import (
    TOOL_CANCELLED_COLOR,
    TOOL_COMPLETED_COLOR,
    TOOL_DURATION_COLOR,
    TOOL_FAILED_COLOR,
    TOOL_PREPARING_COLOR,
    TOOL_RUNNING_COLOR,
)


TOOL_ARGS_PREVIEW_CHARS = 112
TOOL_ARGS_MAX_ROWS = 12


def compact_tool_args(args: Any, limit: int = TOOL_ARGS_PREVIEW_CHARS) -> str:
    """Return a compact single-line preview without limiting the full arguments."""
    rendered = _json_text(args, indent=None)
    compact = re.sub(r"\s+", " ", rendered).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "\u2026"


def pretty_tool_args(args: Any) -> str:
    """Return readable complete arguments, preserving incomplete streamed JSON."""
    return _json_text(args, indent=2)


def tool_lifecycle_status(
    *,
    draft: bool,
    started_at: float | None,
    duration_ms: int | float | None,
    is_error: bool,
    terminal_status: str = "",
    frame: int = 0,
    clock: Any = time.monotonic,
) -> Text:
    """Render the shared MIRA tool status vocabulary for any tool container."""
    status = Text()
    if duration_ms is not None:
        if terminal_status == "cancelled":
            verb = "Cancelled after"
            verb_color = TOOL_CANCELLED_COLOR
        elif terminal_status == "interrupted":
            verb = "Interrupted after"
            verb_color = TOOL_FAILED_COLOR
        else:
            verb = "Failed after" if is_error else "Completed in"
            verb_color = TOOL_FAILED_COLOR if is_error else TOOL_COMPLETED_COLOR
        status.append(verb, style=verb_color)
        status.append(f" {format_elapsed(duration_ms)}", style=TOOL_DURATION_COLOR)
    elif started_at is not None:
        state = "Preparing" if draft else "Running"
        state_color = TOOL_PREPARING_COLOR if draft else TOOL_RUNNING_COLOR
        spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        status.append(f"{spinner} ", style=TOOL_DURATION_COLOR)
        status.append(state, style=state_color)
        status.append(
            f" · {format_elapsed(elapsed_ms(started_at, clock=clock))} elapsed",
            style=TOOL_DURATION_COLOR,
        )
    return status


def _json_text(value: Any, *, indent: int | None) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    separators = (", ", ": ") if indent is None else None
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


class ToolArgumentTextArea(TextArea):
    """Read-only native editor that grows to its wrapped content, then scrolls."""

    MAX_ROWS = TOOL_ARGS_MAX_ROWS
    WRAP_ESTIMATE = 96

    def __init__(self, text: str) -> None:
        super().__init__(
            text,
            language=None,
            soft_wrap=True,
            read_only=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            tab_behavior="focus",
            classes="tool-args-editor",
        )
        self._estimated_rows = self._estimate_rows(text)
        self.styles.height = min(self.MAX_ROWS, self._estimated_rows)

    def on_mount(self) -> None:
        self.call_after_refresh(self.fit_to_rendered_content)

    def on_resize(self) -> None:
        self.call_after_refresh(self.fit_to_rendered_content)

    def on_click(self, event: Click) -> None:
        """Keep a click in the editor focused for native selection and copying."""
        self.focus()
        event.stop()

    def replace_text(self, text: str) -> None:
        """Replace displayed arguments and refit without replacing the editor."""
        if text == self.text:
            return
        self.load_text(text)
        self._estimated_rows = self._estimate_rows(text)
        self.styles.height = min(self.MAX_ROWS, self._estimated_rows)
        if self.is_mounted:
            self.call_after_refresh(self.fit_to_rendered_content)

    def fit_to_rendered_content(self) -> None:
        """Use Textual's actual soft-wrapped height once width is available."""
        content_rows = (
            self.wrapped_document.height
            if self.wrap_width > 0
            else self._estimated_rows
        )
        self.styles.height = min(self.MAX_ROWS, max(1, content_rows))

    @classmethod
    def _estimate_rows(cls, text: str) -> int:
        lines = text.splitlines() or [""]
        return max(
            1,
            sum(
                max(1, math.ceil(len(line) / cls.WRAP_ESTIMATE))
                for line in lines
            ),
        )


class ToolBubble(Vertical):
    """A MIRA transcript tool call with collapsible, complete arguments."""

    def __init__(self, name: str, args: Any, *, draft: bool = False) -> None:
        super().__init__(classes="message tool-call")
        self._call_text = Text()
        self._output_text = Text()
        self._status_text = Text()
        self.args_editor = ToolArgumentTextArea(pretty_tool_args(args))
        self.args_collapsible = Collapsible(
            self.args_editor,
            title=self._title(args, draft),  # type: ignore[arg-type]
            collapsed=True,
            classes="tool-args-collapsible",
        )
        self.output = Static(classes="tool-output")
        self.status = Static(classes="tool-status")
        self.output.styles.display = "none"
        self.status.styles.display = "none"
        self.update_call(name, args, draft=draft)

    def compose(self) -> ComposeResult:
        yield self.args_collapsible
        yield self.output
        yield self.status

    def on_click(self, event: Click) -> None:
        """Keep native tool interactions from moving focus to the transcript."""
        event.stop()

    @property
    def renderable(self) -> Text:
        """Expose the visible transcript text to existing transcript consumers."""
        rendered = self._call_text.copy()
        for section in (self._output_text, self._status_text):
            if section.plain:
                rendered.append("\n")
                rendered.append_text(section)
        return rendered

    def update_call(self, name: str, args: Any, *, draft: bool) -> None:
        """Update a streamed or edited call while preserving expansion state."""
        self.border_title = escape(f"tool - {name}")
        self._call_text = self._title(args, draft)
        self.args_collapsible.title = self._call_text  # type: ignore[assignment]
        self.args_editor.replace_text(pretty_tool_args(args))

    def update_details(self, output: Text, status: Text) -> None:
        """Update compact output and lifecycle status independently of arguments."""
        self._output_text = output.copy()
        self._status_text = status.copy()
        self.output.update(self._output_text)
        self.status.update(self._status_text)
        self.output.styles.display = "block" if self._output_text.plain else "none"
        self.status.styles.display = "block" if self._status_text.plain else "none"

    @on(Collapsible.Expanded)
    def refit_expanded_arguments(self, event: Collapsible.Expanded) -> None:
        if event.collapsible is self.args_collapsible:
            self.args_editor.call_after_refresh(self.args_editor.fit_to_rendered_content)

    @staticmethod
    def _title(args: Any, draft: bool) -> Text:
        title = Text()
        title.append("draft: " if draft else "call: ", style="bold cyan")
        title.append(compact_tool_args(args))
        return title
