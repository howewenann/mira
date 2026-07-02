"""Live subagent execution panel for the Textual TUI."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

from ui.names import generate_slug
from ui.widgets.chat_log import SPINNER_FRAMES

STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_CANCELLED = "CANCELLED"
STATUS_ERROR = "ERROR"
TASKS_GROUP = "__regular_tasks__"
STATUS_COL = 11
TIME_COL = 7
TASK_MIN_COL = 48
MAX_LABEL_CHARS = 80
MAX_HINT_CHARS = 60
IDENTITY_STYLE = "bold #B7A4E8"


@dataclass
class SubagentRecord:
    """One live row in the subagent panel."""

    key: str
    name: str
    hint: str
    group_key: str = ""
    status: str = STATUS_RUNNING
    started: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    output: str = ""

    def elapsed_seconds(self) -> float:
        if self.duration_ms is not None:
            return max(0.0, self.duration_ms / 1000)
        return max(0.0, time.monotonic() - self.started)


@dataclass
class SubagentGroup:
    """A user-facing eval group backed by an internal eval id."""

    key: str
    index: int
    order: list[str] = field(default_factory=list)


class SubagentsPanel(Vertical):
    """Bottom panel for live subagent telemetry."""

    can_focus = False
    can_focus_children = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._records: dict[str, SubagentRecord] = {}
        self._order: list[str] = []
        self._regular_order: list[str] = []
        self._groups: dict[str, SubagentGroup] = {}
        self._group_order: list[str] = []
        self._eval_groups: dict[str, str] = {}
        self._selected_group: str = ""
        self._active_group: str = ""
        self._aliases: dict[str, deque[str]] = {}
        self._spinner_index = 0
        self._expanded = True
        self._closed = False
        self._pending_reset = False
        self._fallback_suffixes = count(1)
        self._last_header = ""
        self._last_groups = ""
        self._last_tasks = ""

    def compose(self) -> Any:
        with Horizontal(id="subagents-panel-header-row"):
            yield Static("[-]", id="subagents-panel-toggle", classes="subagents-panel-action")
            yield Static("", id="subagents-panel-header")
            yield Static("x", id="subagents-panel-close", classes="subagents-panel-action")
        with Horizontal(id="subagents-panel-body"):
            with VerticalScroll(id="subagents-groups-scroll"):
                yield Static("", id="subagents-groups")
            with VerticalScroll(id="subagents-tasks-scroll"):
                yield Static("", id="subagents-tasks")

    def prepare_turn(self) -> None:
        """Collapse completed panel state before the next prompt."""
        if self._records and not self.has_running_subagents():
            self.set_expanded(False)
            self._pending_reset = True

    def reset(self) -> None:
        """Clear panel state and hide it."""
        self._records = {}
        self._order = []
        self._regular_order = []
        self._groups = {}
        self._group_order = []
        self._eval_groups = {}
        self._selected_group = ""
        self._active_group = ""
        self._aliases = {}
        self._pending_reset = False
        self._closed = False
        self._last_header = ""
        self._last_groups = ""
        self._last_tasks = ""
        self.display = False
        self._refresh()

    def close(self) -> None:
        """Hide the current panel without deleting in-memory state."""
        self._closed = True
        self.display = False

    def toggle(self) -> None:
        """Toggle expanded/collapsed panel body."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._refresh_visibility()
        self._refresh_controls()
        self._refresh()

    def start_subagent(
        self,
        name: str,
        task: str = "",
        *,
        row_id: str = "",
        eval_id: str = "",
        label: str = "",
    ) -> None:
        """Add or update a running subagent row."""
        if self._pending_reset or self._closed:
            self.reset()

        eval_key = str(eval_id or "")
        group_key = self._group_key_for_eval(eval_key) if eval_key else ""
        key = str(row_id or "")
        display_name = self._display_name(name, key=key, track=not bool(eval_key), force=bool(eval_key))
        if not key:
            key = display_name

        if group_key:
            self._active_group = group_key
            self._selected_group = group_key
        elif self._group_order:
            self._selected_group = TASKS_GROUP

        if key not in self._records:
            self._order.append(key)
            if group_key:
                self._groups[group_key].order.append(key)
            else:
                self._regular_order.append(key)

        self._records[key] = SubagentRecord(
            key=key,
            name=sanitize(display_name, max_chars=MAX_LABEL_CHARS),
            hint=compact_hint(label or task),
            group_key=group_key,
        )
        self._show()

    def update_subagent_request(self, name: str, task: str) -> None:
        """Fill late-arriving task text for a running ungrouped row."""
        record = self._record_for_name(name)
        if record is None or not task:
            return
        record.hint = compact_hint(task)
        self._refresh()

    def finish_subagent(
        self,
        name: str,
        result: str = "",
        *,
        row_id: str = "",
        eval_id: str = "",
        status: str = STATUS_DONE,
        duration_ms: int | None = None,
    ) -> None:
        """Mark a row terminal."""
        key = str(row_id or "")
        record = self._records.get(key) if key else None
        record = record or self._record_for_name(name)
        if record is None:
            if status not in {STATUS_CANCELLED, STATUS_ERROR}:
                return
            group_key = self._orphan_error_group_key(str(eval_id or ""))
            if group_key:
                key = key or self._display_name(name, key=key, track=False)
                display_name = self._display_name(name, key=key, track=False, force=bool(group_key))
                self._records[key] = SubagentRecord(key=key, name=display_name, hint="", group_key=group_key)
                self._order.append(key)
                self._groups[group_key].order.append(key)
                record = self._records[key]
            else:
                return
        record.status = status
        record.output = sanitize(result, max_chars=MAX_HINT_CHARS)
        record.duration_ms = duration_ms if duration_ms is not None else int(record.elapsed_seconds() * 1000)
        self._refresh()

    def cancel_running(self) -> None:
        """Mark all running rows as cancelled."""
        changed = False
        for record in self._records.values():
            if record.status == STATUS_RUNNING:
                record.status = STATUS_CANCELLED
                record.duration_ms = int(record.elapsed_seconds() * 1000)
                changed = True
        if changed:
            self._refresh()

    def tick(self) -> None:
        """Advance the spinner on running rows."""
        if not self.has_running_subagents():
            return
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        self._refresh()

    def has_running_subagents(self) -> bool:
        return any(record.status == STATUS_RUNNING for record in self._records.values())

    def select_group(self, index: int) -> None:
        """Select a displayed group by zero-based index."""
        keys = self._display_group_keys()
        if index < 0 or index >= len(keys):
            return
        self._selected_group = keys[index]
        self._refresh()

    def select_group_line(self, line: int) -> None:
        """Select a group from a rendered group-list line."""
        if line <= 0:
            return
        self.select_group(line - 1)

    def _group_key_for_eval(self, eval_id: str) -> str:
        if eval_id in self._eval_groups:
            return self._eval_groups[eval_id]
        reusable = self._retry_group_key()
        if reusable:
            self._clear_group(reusable)
            self._eval_groups[eval_id] = reusable
            return reusable
        group_key = eval_id or f"eval-{len(self._group_order) + 1}"
        self._ensure_group(group_key)
        self._eval_groups[eval_id] = group_key
        return group_key

    def _retry_group_key(self) -> str:
        if not self._group_order:
            return ""
        group_key = self._active_group or self._group_order[-1]
        records = [self._records[key] for key in self._groups[group_key].order if key in self._records]
        if records and any(record.status == STATUS_ERROR for record in records) and not any(
            record.status == STATUS_RUNNING for record in records
        ):
            return group_key
        return ""

    def _clear_group(self, group_key: str) -> None:
        group = self._groups.get(group_key)
        if group is None:
            return
        remove = set(group.order)
        self._order = [key for key in self._order if key not in remove]
        for key in remove:
            self._records.pop(key, None)
        group.order = []

    def _orphan_error_group_key(self, eval_id: str) -> str:
        if eval_id and eval_id in self._eval_groups:
            return self._eval_groups[eval_id]
        if self._active_group:
            return self._active_group
        if self._group_order:
            return self._group_order[-1]
        return ""

    def _ensure_group(self, group_key: str) -> None:
        if group_key in self._groups:
            return
        self._groups[group_key] = SubagentGroup(group_key, len(self._group_order) + 1)
        self._group_order.append(group_key)

    def _display_name(self, name: str, *, key: str = "", track: bool = True, force: bool = False) -> str:
        if key and key in self._records:
            return self._records[key].name
        visible_base = base_name(name) if force else name
        if has_suffix(visible_base):
            return visible_base
        if key:
            return f"{visible_base} [{self._next_suffix()}]"
        display = f"{visible_base} [{self._next_suffix()}]"
        if track:
            self._aliases.setdefault(name, deque()).append(display)
        return display

    def _next_suffix(self) -> str:
        return generate_slug(fallback=self._fallback_suffixes)

    def _record_for_name(self, name: str) -> SubagentRecord | None:
        record = self._records.get(name)
        if record is not None:
            return record
        if has_suffix(name):
            for key in reversed(self._order):
                candidate = self._records[key]
                if candidate.name == name:
                    return candidate
            return None
        queue = self._aliases.get(name)
        if queue:
            for display in list(queue):
                record = self._records.get(display)
                if record is not None and record.status == STATUS_RUNNING:
                    return record
        for key in reversed(self._order):
            candidate = self._records[key]
            if candidate.status == STATUS_RUNNING and base_name(candidate.name) == name:
                return candidate
        return None

    def _show(self) -> None:
        self._closed = False
        self.display = True
        self.set_expanded(True)

    def _refresh_controls(self) -> None:
        try:
            self.query_one("#subagents-panel-toggle", Static).update("[-]" if self._expanded else "[+]")
        except NoMatches:
            return

    def _refresh_visibility(self) -> None:
        try:
            self.query_one("#subagents-panel-body").display = self._expanded
            self.query_one("#subagents-groups-scroll").display = self._expanded and bool(self._display_group_keys())
        except NoMatches:
            return

    def _refresh(self) -> None:
        self._refresh_visibility()
        self._refresh_controls()
        self._update_static("subagents-panel-header", self._render_header())
        self._update_static("subagents-groups", self._render_groups())
        self._update_static("subagents-tasks", self._render_tasks())

    def _update_static(self, widget_id: str, text: Text) -> None:
        plain = text.plain
        cache_name = f"_last_{widget_id.replace('subagents-panel-', '').replace('subagents-', '')}"
        if getattr(self, cache_name, "") == plain:
            return
        try:
            widget = self.query_one(f"#{widget_id}", Static)
        except NoMatches:
            return
        widget.update(text)
        setattr(self, cache_name, plain)

    def _render_header(self) -> Text:
        done, total, failed, cancelled = self._counts(self._records.values())
        text = Text()
        text.append("dynamic subagents" if self._eval_only() else "subagents", style="bold #ECE7FF")
        if total:
            text.append(f"  {done}/{total} done", style="dim")
        if self._group_order:
            label = "group" if len(self._group_order) == 1 else "groups"
            text.append(f"  {len(self._group_order)} {label}", style="dim")
        if failed:
            text.append(f"  {failed} failed", style="red")
        if cancelled:
            text.append(f"  {cancelled} cancelled", style="yellow")
        return text

    def _render_groups(self) -> Text:
        text = Text()
        keys = self._display_group_keys()
        if not keys:
            return text
        text.append("Groups\n", style="dim")
        selected = self._selected_group or self._active_group or keys[-1]
        for group_key in keys:
            records = self._records_for_group_key(group_key)
            done, total, failed, cancelled = self._counts(records)
            marker = ">" if group_key == selected else " "
            status, style = group_status_icon(done=done, total=total, failed=failed, cancelled=cancelled)
            elapsed = max((record.elapsed_seconds() for record in records), default=0.0)
            label = "Tasks" if group_key == TASKS_GROUP else f"Group {self._groups[group_key].index}"
            text.append(f"{marker} ")
            text.append(status, style=style)
            text.append(f" {label} {done}/{total}  {format_seconds(elapsed)}\n")
        return text

    def _render_tasks(self) -> Text:
        records = self._displayed_records()
        text = Text()
        if not records:
            return text
        task_col = self._task_col()
        text.append(" " * 3)
        text.append("TASK".ljust(task_col), style="dim")
        text.append("STATUS".ljust(STATUS_COL), style="dim")
        text.append("TIME".rjust(TIME_COL), style="dim")
        text.append("\n")
        for record in records:
            icon, style = status_icon(record.status, self._spinner_index)
            text.append(f" {icon} ", style=style)
            append_task_cell(text, record, task_col)
            text.append(record.status.ljust(STATUS_COL), style=style)
            text.append(format_seconds(record.elapsed_seconds()).rjust(TIME_COL), style="dim")
            text.append("\n")
        return text

    def _display_group_keys(self) -> list[str]:
        keys = []
        if self._regular_order and self._group_order:
            keys.append(TASKS_GROUP)
        keys.extend(self._group_order)
        return keys

    def _displayed_records(self) -> list[SubagentRecord]:
        if not self._group_order:
            return [self._records[key] for key in self._regular_order if key in self._records]
        group_key = self._selected_group or self._active_group or self._display_group_keys()[-1]
        return self._records_for_group_key(group_key)

    def _records_for_group_key(self, group_key: str) -> list[SubagentRecord]:
        if group_key == TASKS_GROUP:
            return [self._records[key] for key in self._regular_order if key in self._records]
        group = self._groups.get(group_key)
        if group is None:
            return []
        return [self._records[key] for key in group.order if key in self._records]

    def _task_col(self) -> int:
        width = max(60, self.size.width or 80)
        if self._display_group_keys():
            width -= 28
        return max(TASK_MIN_COL, width - STATUS_COL - TIME_COL - 8)

    def _counts(self, records: Any) -> tuple[int, int, int, int]:
        items = list(records)
        total = len(items)
        done = sum(1 for record in items if record.status != STATUS_RUNNING)
        failed = sum(1 for record in items if record.status == STATUS_ERROR)
        cancelled = sum(1 for record in items if record.status == STATUS_CANCELLED)
        return done, total, failed, cancelled

    def _eval_only(self) -> bool:
        return bool(self._group_order) and not self._regular_order


def status_icon(status: str, spinner_index: int) -> tuple[str, str]:
    if status == STATUS_RUNNING:
        return SPINNER_FRAMES[spinner_index], "bold yellow"
    if status == STATUS_DONE:
        return "v", "bold green"
    if status == STATUS_CANCELLED:
        return "-", "bold yellow"
    return "x", "bold red"


def group_status_icon(*, done: int, total: int, failed: int, cancelled: int) -> tuple[str, str]:
    """Return the aggregate group status icon and style."""
    if failed:
        return "x", "bold red"
    if cancelled and done == total:
        return "-", "bold yellow"
    if done == total and total:
        return "v", "bold green"
    return "*", "bold yellow"


def task_text(record: SubagentRecord, width: int) -> str:
    identity = record.name
    hint = terminal_hint(record)
    if not hint:
        return identity
    remaining = width - len(identity) - 2
    if remaining <= 0:
        return identity
    return f"{identity}  {sanitize(hint, max_chars=remaining)}"


def append_task_cell(text: Text, record: SubagentRecord, width: int) -> None:
    value = task_text(record, width)
    identity = record.name
    text.append(identity, style=IDENTITY_STYLE)
    rest = value[len(identity):]
    if rest:
        text.append(rest)
    padding = max(0, width - len(value))
    if padding:
        text.append(" " * padding)


def terminal_hint(record: SubagentRecord) -> str:
    if record.status in {STATUS_ERROR, STATUS_CANCELLED} and record.output:
        if record.hint:
            return f"{record.hint} - {record.output}"
        return record.output
    return record.hint


def has_suffix(label: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]\s*$", label))


def base_name(label: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", label).strip() or label


def compact_hint(value: Any) -> str:
    return sanitize(value, max_chars=MAX_HINT_CHARS)


def sanitize(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes}m{rest:02d}s"
