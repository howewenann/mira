"""Composite Rubric bubble with compact verifier tool lifecycles."""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, Resize
from textual.widgets import Collapsible, Static

from runtime.rubric_events import elapsed_ms, format_elapsed
from ui.spinners import SPINNER_FRAMES
from ui.terminal_colors import (
    RUBRIC_BODY_COLOR,
    RUBRIC_HEADER_COLOR,
    TOOL_COMPLETED_COLOR,
    TOOL_DURATION_COLOR,
    TOOL_FAILED_COLOR,
    TOOL_RUNNING_COLOR,
)
from ui.widgets.tool_bubble import (
    ToolArgumentTextArea,
    compact_tool_args,
    pretty_tool_args,
    tool_lifecycle_status,
)


class WidthAwareToolPreview(Static):
    """One visual output line that truncates to its current rendered width."""

    def __init__(self) -> None:
        super().__init__(classes="rubric-tool-output")
        self.raw_output = ""
        self.is_error = False

    def set_output(self, output: str, *, is_error: bool) -> None:
        self.raw_output = str(output)
        self.is_error = is_error
        self.refresh(layout=True)

    def on_resize(self, event: Resize) -> None:  # noqa: ARG002
        self.refresh()

    def render(self) -> Text:
        compact = re.sub(r"\s+", " ", self.raw_output).strip()
        width = max(1, self.content_size.width or self.size.width)
        preview = Text(compact, style="red" if self.is_error else "dim")
        preview.truncate(width, overflow="ellipsis")
        return preview


class RubricToolRow(Vertical):
    """Borderless ToolBubble-style row owned by one Rubric pass."""

    def __init__(self, name: str, args: Any, *, draft: bool) -> None:
        super().__init__(classes="rubric-tool-row")
        self.tool_name = str(name or "tool")
        self.args = args
        self.draft = draft
        self.result = ""
        self.is_error = False
        self.started_at: float | None = time.monotonic()
        self.duration_ms: int | None = None
        self.frame = 0
        self.args_editor = ToolArgumentTextArea(pretty_tool_args(args))
        self.args_collapsible = Collapsible(
            self.args_editor,
            title=self._title(),  # type: ignore[arg-type]
            collapsed=True,
            classes="tool-args-collapsible rubric-tool-args",
        )
        self.output = WidthAwareToolPreview()
        self.status = Static(classes="rubric-tool-status")
        self.output.styles.display = "none"
        self._refresh_status()

    def compose(self) -> ComposeResult:
        yield self.args_collapsible
        yield self.output
        yield self.status

    def on_click(self, event: Click) -> None:
        event.stop()

    @property
    def renderable(self) -> Text:
        rendered = self._title()
        if self.result:
            rendered.append("\n")
            rendered.append_text(self.output.render())
        status = self.status.content
        if isinstance(status, Text) and status.plain:
            rendered.append("\n")
            rendered.append_text(status)
        return rendered

    def update_call(self, name: str, args: Any, *, draft: bool) -> None:
        self.tool_name = str(name or "tool")
        self.args = args
        self.draft = draft
        self.args_collapsible.title = self._title()  # type: ignore[assignment]
        self.args_editor.replace_text(pretty_tool_args(args))
        self._refresh_status()

    def finish(self, output: str, *, is_error: bool, duration_ms: int | None) -> None:
        self.result = str(output)
        self.is_error = is_error
        self.draft = False
        if duration_ms is None and self.started_at is not None:
            duration_ms = elapsed_ms(self.started_at)
        self.duration_ms = duration_ms
        self.output.set_output(self.result, is_error=is_error)
        self.output.styles.display = "block" if self.result else "none"
        self._refresh_status()

    def tick(self) -> None:
        if self.duration_ms is not None or self.started_at is None:
            return
        self.frame += 1
        self._refresh_status()

    @on(Collapsible.Expanded)
    def refit_expanded_arguments(self, event: Collapsible.Expanded) -> None:
        if event.collapsible is self.args_collapsible:
            self.args_editor.call_after_refresh(self.args_editor.fit_to_rendered_content)

    def _title(self) -> Text:
        title = Text()
        title.append(f"tool - {self.tool_name} · ", style=f"bold {RUBRIC_HEADER_COLOR}")
        title.append("draft: " if self.draft else "call: ", style="bold cyan")
        title.append(compact_tool_args(self.args), style=RUBRIC_BODY_COLOR)
        return title

    def _refresh_status(self) -> None:
        self.status.update(
            tool_lifecycle_status(
                draft=self.draft,
                started_at=self.started_at,
                duration_ms=self.duration_ms,
                is_error=self.is_error,
                frame=self.frame,
                clock=time.monotonic,
            )
        )


class RubricBubble(Vertical):
    """One non-collapsible Rubric review with verifier and grader phases."""

    def __init__(
        self,
        run_id: str,
        pass_number: int,
        max_iterations: int,
        *,
        grader_model: str = "",
    ) -> None:
        super().__init__(classes="message rubric")
        self.run_id = run_id
        self.pass_number = pass_number
        self.max_iterations = max_iterations
        self.grader_model = grader_model
        self.header = Static(self._header_text(), classes="rubric-header")
        self.verifier_heading = Static(
            Text("Verifier", style=f"bold {RUBRIC_HEADER_COLOR}"),
            classes="rubric-phase-heading",
        )
        self.verifier_status = Static(classes="rubric-verifier-status")
        self.tools = Vertical(classes="rubric-tool-rows")
        self.no_tools = Static("No tools called.", classes="rubric-no-tools")
        self.grader_heading = Static(classes="rubric-phase-heading")
        self.grader_status = Static(classes="rubric-grader-status")
        self.result = Static(classes="rubric-result")
        self.no_tools.styles.display = "none"
        self.grader_heading.styles.display = "none"
        self.grader_status.styles.display = "none"
        self.result.styles.display = "none"
        self._tool_rows: dict[str, dict[str, Any]] = {}
        self._tool_name_queues: dict[str, deque[str]] = defaultdict(deque)
        self._pending_results: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._verifier_started_at: float | None = time.monotonic()
        self._verifier_frame = 0
        self._grader_started_at: float | None = None
        self._grader_frame = 0
        self._grader_seen = False
        self._refresh_verifier_status()

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.verifier_heading
        yield self.verifier_status
        yield self.tools
        yield self.no_tools
        yield self.grader_heading
        yield self.grader_status
        yield self.result

    def on_mount(self) -> None:
        pending = [
            block["widget"]
            for block in self._tool_rows.values()
            if block["widget"].parent is None
        ]
        if pending:
            self.tools.mount(*pending)

    @property
    def renderable(self) -> Text:
        sections = [self.header.content, self.verifier_heading.content]
        if self.verifier_status.styles.display != "none":
            sections.append(self.verifier_status.content)
        sections.extend(block["widget"].renderable for block in self._tool_rows.values())
        if self.no_tools.styles.display != "none":
            sections.append(Text("No tools called."))
        if self.grader_heading.styles.display != "none":
            sections.append(self.grader_heading.content)
        if self.grader_status.styles.display != "none":
            sections.append(self.grader_status.content)
        if self.result.styles.display != "none":
            sections.append(self.result.content)
        rendered = Text()
        for section in sections:
            if not isinstance(section, Text) or not section.plain:
                continue
            if rendered.plain:
                rendered.append("\n\n")
            rendered.append_text(section)
        return rendered

    def tool_delta(self, name: str, args: Any, call_id: str = "") -> None:
        key = self._tool_update_key(name, call_id)
        block = self._tool_rows.get(key)
        if block is None:
            block = self._add_tool(key, name, args, draft=True)
        else:
            block["name"] = name
            block["args"] = args
            block["draft"] = True
            block["widget"].update_call(name, args, draft=True)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        key = self._tool_update_key(name, call_id)
        block = self._tool_rows.get(key)
        if block is None:
            block = self._add_tool(key, name, args, draft=False)
        else:
            block["name"] = name
            block["args"] = args
            block["draft"] = False
            block["widget"].update_call(name, args, draft=False)
        pending = self._pending_results.pop(call_id, None) if call_id else None
        if pending is not None:
            self.tool_result(name, call_id=call_id, **pending)

    def tool_result(
        self,
        name: str,
        output: str,
        *,
        call_id: str = "",
        is_error: bool = False,
        duration_ms: int | None = None,
    ) -> None:
        key = f"id:{call_id}" if call_id else self._oldest_unfinished(name)
        if key is None or key not in self._tool_rows:
            if call_id:
                self._pending_results[call_id] = {
                    "output": output,
                    "is_error": is_error,
                    "duration_ms": duration_ms,
                }
            return
        block = self._tool_rows[key]
        block["result"] = output
        block["is_error"] = is_error
        block["duration_ms"] = duration_ms
        block["draft"] = False
        block["widget"].finish(output, is_error=is_error, duration_ms=duration_ms)
        self._remove_queue_key(name, key)

    def verifier_started(self) -> None:
        self.verifier_heading.update(
            Text("Verifier", style=f"bold {RUBRIC_HEADER_COLOR}")
        )
        self._verifier_started_at = time.monotonic()
        self._verifier_frame = 0
        self.verifier_status.styles.display = "block"
        self._refresh_verifier_status()
        self.no_tools.styles.display = "none"
        self.grader_heading.styles.display = "none"
        self.grader_status.styles.display = "none"
        self._grader_started_at = None
        self._grader_seen = False

    def verifier_finished(self, *, succeeded: bool, duration_ms: int | None = None) -> None:
        label = "Complete" if succeeded else "Failed"
        color = TOOL_COMPLETED_COLOR if succeeded else TOOL_FAILED_COLOR
        heading = Text("Verifier", style=f"bold {RUBRIC_HEADER_COLOR}")
        heading.append(" · ", style=RUBRIC_BODY_COLOR)
        heading.append(label, style=color)
        if duration_ms is not None:
            heading.append(f" · {format_elapsed(duration_ms)}", style=TOOL_DURATION_COLOR)
        self.verifier_heading.update(heading)
        self.verifier_status.styles.display = "none"
        self._verifier_started_at = None
        self.no_tools.styles.display = "block" if succeeded and not self._tool_rows else "none"

    def grader_started(self) -> None:
        self._grader_seen = True
        self._grader_started_at = time.monotonic()
        self._grader_frame = 0
        self.grader_heading.update(Text("Grader", style=f"bold {RUBRIC_HEADER_COLOR}"))
        self.grader_heading.styles.display = "block"
        self.grader_status.styles.display = "block"
        self._refresh_grader_status()

    def grader_finished(self, *, succeeded: bool, duration_ms: int | None = None) -> None:
        self._grader_seen = True
        label = "Complete" if succeeded else "Failed"
        color = TOOL_COMPLETED_COLOR if succeeded else TOOL_FAILED_COLOR
        heading = Text("Grader", style=f"bold {RUBRIC_HEADER_COLOR}")
        heading.append(" · ", style=RUBRIC_BODY_COLOR)
        heading.append(label, style=color)
        if duration_ms is not None:
            heading.append(f" · {format_elapsed(duration_ms)}", style=TOOL_DURATION_COLOR)
        self.grader_heading.update(heading)
        self.grader_heading.styles.display = "block"
        self.grader_status.styles.display = "none"
        self._grader_started_at = None

    def finish(self, evaluation: dict[str, Any], result: Text) -> None:
        tools = evaluation.get("verifier_tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name") or "tool")
                call_id = str(tool.get("call_id") or "")
                self.tool_call(name, tool.get("args", {}), call_id)
                self.tool_result(
                    name,
                    str(tool.get("output") or ""),
                    call_id=call_id,
                    is_error=bool(tool.get("is_error")),
                    duration_ms=tool.get("duration_ms"),
                )
        verifier_status = str(evaluation.get("verifier_status") or "complete")
        self.verifier_finished(
            succeeded=verifier_status != "failed",
            duration_ms=evaluation.get("verifier_duration_ms"),
        )
        grader_status = str(evaluation.get("grader_status") or "complete")
        if verifier_status != "failed" or self._grader_seen:
            self.grader_finished(
                succeeded=grader_status != "failed",
                duration_ms=evaluation.get("grader_duration_ms"),
            )
        self.result.update(result)
        self.result.styles.display = "block"

    def tick(self) -> None:
        if self._verifier_started_at is not None:
            self._verifier_frame += 1
            self._refresh_verifier_status()
        for block in self._tool_rows.values():
            block["widget"].tick()
        if self._grader_started_at is not None:
            self._grader_frame += 1
            self._refresh_grader_status()

    def interrupt(self) -> None:
        self.verifier_heading.update(Text("Review interrupted.", style=RUBRIC_BODY_COLOR))
        self.verifier_status.styles.display = "none"
        self._verifier_started_at = None
        self.grader_status.styles.display = "none"
        self._grader_started_at = None

    def _header_text(self) -> Text:
        text = Text(
            f"Rubric review · pass {self.pass_number} of {self.max_iterations}",
            style=f"bold {RUBRIC_HEADER_COLOR}",
        )
        if self.grader_model:
            text.append(f"\nModel: {self.grader_model}", style=RUBRIC_BODY_COLOR)
        return text

    def _refresh_grader_status(self) -> None:
        if self._grader_started_at is None:
            return
        elapsed = elapsed_ms(self._grader_started_at)
        frame = SPINNER_FRAMES[self._grader_frame % len(SPINNER_FRAMES)]
        status = Text(f"{frame} ", style=TOOL_DURATION_COLOR)
        status.append("Evaluating evidence", style=TOOL_RUNNING_COLOR)
        status.append(f" · {format_elapsed(elapsed)} elapsed", style=TOOL_DURATION_COLOR)
        self.grader_status.update(status)

    def _refresh_verifier_status(self) -> None:
        if self._verifier_started_at is None:
            return
        elapsed = elapsed_ms(self._verifier_started_at)
        frame = SPINNER_FRAMES[self._verifier_frame % len(SPINNER_FRAMES)]
        status = Text(f"{frame} ", style=TOOL_DURATION_COLOR)
        status.append("Verifying", style=TOOL_RUNNING_COLOR)
        status.append(f" · {format_elapsed(elapsed)} elapsed", style=TOOL_DURATION_COLOR)
        self.verifier_status.update(status)

    def _add_tool(self, key: str, name: str, args: Any, *, draft: bool) -> dict[str, Any]:
        widget = RubricToolRow(name, args, draft=draft)
        block = {
            "name": name,
            "args": args,
            "draft": draft,
            "result": "",
            "is_error": False,
            "duration_ms": None,
            "widget": widget,
        }
        self._tool_rows[key] = block
        self._tool_name_queues[name].append(key)
        if self.tools.is_mounted:
            self.tools.mount(widget)
        self.no_tools.styles.display = "none"
        return block

    def _tool_update_key(self, name: str, call_id: str) -> str:
        if call_id:
            key = f"id:{call_id}"
            if key in self._tool_rows:
                return key
            draft_key = self._oldest_draft(name)
            if draft_key is not None:
                block = self._tool_rows.pop(draft_key)
                self._tool_rows[key] = block
                queue = self._tool_name_queues.get(name)
                if queue:
                    self._tool_name_queues[name] = deque(
                        key if item == draft_key else item for item in queue
                    )
                return key
            return key
        draft_key = self._oldest_draft(name)
        if draft_key is not None:
            return draft_key
        self._sequence += 1
        return f"name:{name}:{self._sequence}"

    def _oldest_draft(self, name: str) -> str | None:
        for key in self._tool_name_queues.get(name, ()):
            block = self._tool_rows.get(key)
            if block is not None and block.get("draft"):
                return key
        return None

    def _oldest_unfinished(self, name: str) -> str | None:
        for key in self._tool_name_queues.get(name, ()):
            block = self._tool_rows.get(key)
            if block is not None and not block.get("result"):
                return key
        return None

    def _remove_queue_key(self, name: str, key: str) -> None:
        queue = self._tool_name_queues.get(name)
        if queue:
            self._tool_name_queues[name] = deque(item for item in queue if item != key)
