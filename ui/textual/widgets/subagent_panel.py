"""Live subagent execution panel for the Textual TUI."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from rich.cells import cell_len, set_cell_size
from rich.text import Text
from textual import events, on
from textual.containers import Grid, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Collapsible, DataTable, Label, OptionList
from textual.widgets.option_list import Option

from ui.shared.terminal.names import generate_slug
from ui.shared.terminal.spinners import SPINNER_FRAMES

STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_CANCELLED = "CANCELLED"
STATUS_ERROR = "ERROR"
TASKS_GROUP = "__regular_tasks__"
STATUS_COL = 11
TIME_COL = 7
TASK_MIN_COL = 4
PANEL_BODY_ROWS = 8
MAX_LABEL_CHARS = 80
FALLBACK_LABEL_CHARS = 60
MAX_OUTPUT_CHARS = 60
IDENTITY_STYLE = "bold #B7A4E8"


@dataclass
class SubagentRecord:
    """One live row in the subagent panel."""

    key: str
    name: str
    hint: str
    group_key: str = ""
    status: str = STATUS_RUNNING
    started: float = field(default_factory=lambda: time.monotonic())
    duration_ms: int | None = None
    output: str = ""
    finished_at: float | None = None

    def elapsed_seconds(self) -> float:
        if self.duration_ms is not None:
            return max(0.0, self.duration_ms / 1000)
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started)


@dataclass
class SubagentGroup:
    """A user-facing eval group backed by an internal eval id."""

    key: str
    index: int
    order: list[str] = field(default_factory=list)
    terminal_status: str = ""


class SubagentsPanel(Vertical):
    """Bottom panel for live subagent telemetry."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._records: dict[str, SubagentRecord] = {}
        self._order: list[str] = []
        self._regular_order: list[str] = []
        self._groups: dict[str, SubagentGroup] = {}
        self._group_order: list[str] = []
        self._eval_groups: dict[str, str] = {}
        self._eval_outcomes: dict[str, str] = {}
        self._retired_eval_ids: set[str] = set()
        self._selected_group: str = ""
        self._active_group: str = ""
        self._aliases: dict[str, deque[str]] = {}
        self._spinner_index = 0
        self._dismissed = False
        self._pending_reset = False
        self._fallback_suffixes = count(1)

    def compose(self) -> Any:
        groups = Vertical(
            Label("GROUPS", id="subagents-groups-label"),
            OptionList(id="subagents-groups", compact=True),
            id="subagents-groups-column",
        )
        tasks = DataTable(
            id="subagents-tasks",
            cursor_type="none",
            zebra_stripes=False,
            show_cursor=False,
            show_header=False,
        )
        task_header = Grid(
            Label("TASK", id="subagents-task-heading"),
            Label("STATUS", id="subagents-status-heading"),
            Label("TIME", id="subagents-time-heading"),
            id="subagents-tasks-header",
        )
        task_column = Vertical(task_header, tasks, id="subagents-tasks-column")
        body = Horizontal(groups, task_column, id="subagents-panel-body")
        with Horizontal(id="subagents-panel-row"):
            yield Collapsible(
                body,
                title="subagents",
                collapsed=False,
                id="subagents-collapsible",
            )
            with Vertical(id="subagents-close-cell"):
                yield Button("x", id="subagents-panel-close", compact=True)

    def on_mount(self) -> None:
        """Configure native table columns after the widget is mounted."""
        table = self.query_one("#subagents-tasks", DataTable)
        table.cell_padding = 1
        table.add_column("TASK", key="task", width=40)
        table.add_column("STATUS", key="status", width=STATUS_COL - 2 * table.cell_padding)
        table.add_column(Text("TIME", justify="center"), key="time", width=TIME_COL - 2 * table.cell_padding)
        self._refresh()
        self.call_after_refresh(self._align_task_column)

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
        self._eval_outcomes = {}
        self._retired_eval_ids = set()
        self._selected_group = ""
        self._active_group = ""
        self._aliases = {}
        self._pending_reset = False
        self._dismissed = False
        self.display = False
        self._refresh()

    def close(self) -> None:
        """Hide the current panel without deleting in-memory state."""
        if self.has_running_subagents():
            self.set_expanded(False)
            return
        self._dismissed = True
        self.display = False

    def set_expanded(self, expanded: bool) -> None:
        try:
            self.query_one("#subagents-collapsible", Collapsible).collapsed = not expanded
        except NoMatches:
            return
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
        if self._pending_reset:
            self.reset()

        eval_key = str(eval_id or "")
        if eval_key in self._retired_eval_ids:
            return
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

        record = SubagentRecord(
            key=key,
            name=sanitize(display_name, max_chars=MAX_LABEL_CHARS),
            hint=compact_subagent_hint(label, task),
            group_key=group_key,
        )
        self._records[key] = record
        if group_key and self._groups[group_key].terminal_status:
            self._cancel_record(record, time.monotonic())
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
        replaces_fallback = record.status == STATUS_CANCELLED and status != STATUS_CANCELLED
        record.status = status
        record.output = sanitize(result, max_chars=MAX_OUTPUT_CHARS)
        if record.finished_at is None or replaces_fallback:
            record.finished_at = (
                record.started + duration_ms / 1000
                if duration_ms is not None
                else time.monotonic()
            )
        record.duration_ms = (
            duration_ms if duration_ms is not None else int(max(0.0, record.finished_at - record.started) * 1000)
        )
        self._refresh()

    def finish_eval_group(self, eval_id: str, *, failed: bool = False) -> None:
        """Reconcile running rows after their parent eval becomes terminal."""
        eval_key = str(eval_id or "")
        if not eval_key or eval_key in self._retired_eval_ids:
            return
        terminal_status = STATUS_ERROR if failed else STATUS_DONE
        self._eval_outcomes[eval_key] = terminal_status
        group_key = self._eval_groups.get(eval_key)
        group = self._groups.get(group_key or "")
        if group is None:
            return

        group.terminal_status = terminal_status
        finished_at = time.monotonic()
        changed = False
        for key in group.order:
            record = self._records.get(key)
            if record is not None and record.status == STATUS_RUNNING:
                self._cancel_record(record, finished_at)
                changed = True
        if changed or failed:
            self._refresh()

    def cancel_running(self) -> None:
        """Mark all running rows as cancelled."""
        changed = False
        finished_at = time.monotonic()
        for record in self._records.values():
            if record.status == STATUS_RUNNING:
                self._cancel_record(record, finished_at)
                changed = True
        if changed:
            self._refresh()

    def tick(self) -> None:
        """Advance the spinner on running rows."""
        if not self.has_running_subagents():
            return
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        if not self.is_mounted:
            return
        self.query_one("#subagents-collapsible", Collapsible).title = self._render_header()
        self._refresh_running_groups()
        self._refresh_running_tasks()

    def on_resize(self, _event: events.Resize) -> None:
        """Rebuild fixed-width task rows after terminal layout changes."""
        self.call_after_refresh(self._align_task_column)

    def on_show(self, _event: events.Show) -> None:
        """Fit columns once a previously hidden panel has a real width."""
        self.call_after_refresh(self._align_task_column)

    @on(Button.Pressed, "#subagents-panel-close")
    def close_panel(self, event: Button.Pressed) -> None:
        """Close completed panel state from its native button."""
        event.stop()
        self.close()

    @on(OptionList.OptionSelected, "#subagents-groups")
    def select_group_option(self, event: OptionList.OptionSelected) -> None:
        """Select the group represented by a native option."""
        event.stop()
        keys = self._display_group_keys()
        if event.option_id not in keys:
            return
        previous = self._selected_group_key()
        if event.option_id == previous:
            return
        self._selected_group = event.option_id
        self._refresh_group_prompt(previous)
        self._refresh_group_prompt(event.option_id)
        self._refresh_tasks()
        self._refresh_body_height()

    def has_running_subagents(self) -> bool:
        return any(record.status == STATUS_RUNNING for record in self._records.values())

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
        self._groups[group_key].terminal_status = self._eval_outcomes.get(eval_id, "")
        return group_key

    def _retry_group_key(self) -> str:
        if not self._group_order:
            return ""
        group_key = self._active_group or self._group_order[-1]
        group = self._groups[group_key]
        records = [self._records[key] for key in group.order if key in self._records]
        failed = group.terminal_status == STATUS_ERROR or any(record.status == STATUS_ERROR for record in records)
        if records and failed and not any(record.status == STATUS_RUNNING for record in records):
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
        group.terminal_status = ""
        retired = [eval_id for eval_id, key in self._eval_groups.items() if key == group_key]
        for eval_id in retired:
            self._eval_groups.pop(eval_id, None)
            self._eval_outcomes.pop(eval_id, None)
            self._retired_eval_ids.add(eval_id)

    def _orphan_error_group_key(self, eval_id: str) -> str:
        if eval_id in self._retired_eval_ids:
            return ""
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

    @staticmethod
    def _cancel_record(record: SubagentRecord, finished_at: float) -> None:
        record.status = STATUS_CANCELLED
        record.finished_at = finished_at
        record.duration_ms = int(max(0.0, finished_at - record.started) * 1000)

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
        was_hidden = not self.display
        self._dismissed = False
        self.display = True
        if was_hidden:
            self.set_expanded(True)
        else:
            self._refresh()

    def _refresh(self) -> None:
        if not self.is_mounted:
            return
        close = self.query_one("#subagents-panel-close", Button)
        groups = self.query_one("#subagents-groups-column")
        collapsible = self.query_one("#subagents-collapsible", Collapsible)
        close_visible = not self.has_running_subagents()
        groups_visible = bool(self._display_group_keys())
        relayout = close.display != close_visible or groups.display != groups_visible
        close.display = close_visible
        groups.display = groups_visible
        collapsible.title = self._render_header()
        self._refresh_groups()
        self._refresh_tasks()
        self._refresh_body_height()
        if relayout:
            self.call_after_refresh(self._align_task_column)

    def _render_header(self) -> Text:
        done, total, failed, cancelled = self._counts(self._records.values())
        text = Text()
        if self.has_running_subagents():
            text.append(f"{SPINNER_FRAMES[self._spinner_index]} ", style="bold yellow")
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

    def _refresh_groups(self) -> None:
        keys = self._display_group_keys()
        groups = self.query_one("#subagents-groups", OptionList)
        if keys != [option.id for option in groups.options]:
            groups.set_options(Option("", id=key) for key in keys)
        if not keys:
            return
        for index, group_key in enumerate(keys):
            groups.replace_option_prompt_at_index(index, self._group_prompt(group_key))
        groups.highlighted = keys.index(self._selected_group_key())

    def _refresh_running_groups(self) -> None:
        """Update only group prompts whose status icon or clock is live."""
        for group_key in self._display_group_keys():
            if any(record.status == STATUS_RUNNING for record in self._records_for_group_key(group_key)):
                self._refresh_group_prompt(group_key)

    def _refresh_group_prompt(self, group_key: str) -> None:
        keys = self._display_group_keys()
        if group_key not in keys:
            return
        self.query_one("#subagents-groups", OptionList).replace_option_prompt_at_index(
            keys.index(group_key), self._group_prompt(group_key)
        )

    def _group_prompt(self, group_key: str) -> Text:
        records = self._records_for_group_key(group_key)
        done, total, failed, cancelled = self._counts(records)
        if group_key != TASKS_GROUP and self._groups[group_key].terminal_status == STATUS_ERROR:
            failed = max(1, failed)
        status, style = group_status_icon(done=done, total=total, failed=failed, cancelled=cancelled)
        label = "Tasks" if group_key == TASKS_GROUP else f"Group {self._groups[group_key].index}"
        text = Text()
        text.append("> " if group_key == self._selected_group_key() else "  ")
        text.append(status, style=style)
        text.append(f" {label} {done}/{total}  {format_seconds(group_elapsed_seconds(records))}")
        return text

    def _refresh_tasks(self) -> None:
        records = self._displayed_records()
        table = self.query_one("#subagents-tasks", DataTable)
        if len(table.ordered_columns) != 3:
            return
        keys = [record.key for record in records]
        rebuild = keys != [key.value for key in table.rows]
        if rebuild:
            table.clear(columns=False)
        task_width = table.ordered_columns[0].width
        for record in records:
            task, status, elapsed = self._row_cells(record, task_width)
            if rebuild:
                table.add_row(task, status, elapsed, key=record.key)
            else:
                table.update_cell(record.key, "task", task)
                table.update_cell(record.key, "status", status)
                table.update_cell(record.key, "time", elapsed)

    def _refresh_running_tasks(self) -> None:
        """Update live icons and clocks without rebuilding the selected table."""
        table = self.query_one("#subagents-tasks", DataTable)
        if len(table.ordered_columns) != 3:
            return
        task_width = table.ordered_columns[0].width
        for record in self._displayed_records():
            if record.status != STATUS_RUNNING or record.key not in table.rows:
                continue
            task, _, elapsed = self._row_cells(record, task_width)
            table.update_cell(record.key, "task", task)
            table.update_cell(record.key, "time", elapsed)

    def _row_cells(self, record: SubagentRecord, task_width: int) -> tuple[Text, Text, Text]:
        icon, style = status_icon(record.status, self._spinner_index)
        task = Text()
        task.append(f" {icon} ", style=style)
        append_task_cell(task, record, max(0, task_width - 3))
        status = Text(record.status, style=style)
        elapsed = Text(format_seconds(record.elapsed_seconds()), style="dim", justify="center")
        return task, status, elapsed

    def _refresh_body_height(self) -> None:
        body = self.query_one("#subagents-panel-body")
        body.styles.height = PANEL_BODY_ROWS

    def _align_task_column(self) -> None:
        table = self.query_one("#subagents-tasks", DataTable)
        if len(table.ordered_columns) != 3:
            return
        width = table.content_region.width or table.size.width
        if width <= 0:
            return
        fixed_width = STATUS_COL + TIME_COL
        task_width = max(TASK_MIN_COL, width - fixed_width - 2 * table.cell_padding - 1)
        column = table.ordered_columns[0]
        if column.width == task_width:
            return
        column.width = task_width
        self._refresh_tasks()
        table.refresh(layout=True)

    def _display_group_keys(self) -> list[str]:
        keys = []
        if self._regular_order and self._group_order:
            keys.append(TASKS_GROUP)
        keys.extend(self._group_order)
        return keys

    def _selected_group_key(self) -> str:
        keys = self._display_group_keys()
        if not keys:
            return ""
        selected = self._selected_group or self._active_group or keys[-1]
        return selected if selected in keys else keys[-1]

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


def group_elapsed_seconds(records: Any) -> float:
    """Return wall time from the first row start through the final terminal row."""
    items = list(records)
    if not items:
        return 0.0
    started = min(record.started for record in items)
    if any(record.status == STATUS_RUNNING for record in items):
        finished = time.monotonic()
    else:
        finished = max(record.finished_at if record.finished_at is not None else record.started for record in items)
    return max(0.0, finished - started)


def task_text(record: SubagentRecord, width: int) -> str:
    identity = record.name
    hint = terminal_hint(record)
    value = f"{identity}  {hint}" if hint else identity
    return truncate_cells(value, width)


def append_task_cell(text: Text, record: SubagentRecord, width: int) -> None:
    value = task_text(record, width)
    identity = record.name
    start = len(text)
    text.append(value)
    styled_length = len(identity) if value.startswith(identity) else len(value)
    text.stylize(IDENTITY_STYLE, start, start + min(styled_length, len(value)))
    padding = max(0, width - cell_len(value))
    if padding:
        text.append(" " * padding)


def truncate_cells(value: str, width: int) -> str:
    """Truncate text to an exact terminal-cell width with an ASCII marker."""
    width = max(0, width)
    if cell_len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    prefix = set_cell_size(value, width - 3).rstrip()
    return set_cell_size(prefix, width - 3) + "..."


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
    return sanitize(value)


def compact_subagent_hint(label: Any, task: Any) -> str:
    """Prefer full task text when an event label is its truncated fallback."""
    hint = compact_hint(label or task)
    task_hint = compact_hint(task)
    if label and task and len(hint) >= FALLBACK_LABEL_CHARS and task_hint.startswith(hint) and len(task_hint) > len(hint):
        return task_hint
    return hint


def sanitize(value: Any, *, max_chars: int | None = None) -> str:
    text = str(value or "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes}m{rest:02d}s"
