"""Compact, independently inspectable Rubric phase bubbles."""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from core.execution.streams.rubric import elapsed_ms, format_elapsed
from ui.shared.terminal.colors import (
    RUBRIC_BODY_COLOR,
    TOOL_CANCELLED_COLOR,
    TOOL_COMPLETED_COLOR,
    TOOL_DURATION_COLOR,
    TOOL_FAILED_COLOR,
    TOOL_RUNNING_COLOR,
)
from ui.shared.terminal.spinners import SPINNER_FRAMES


class RubricInspectionSelected(Message):
    """Request the existing generic Inspector for one live Rubric phase."""

    def __init__(self, inspection_id: str) -> None:
        super().__init__()
        self.inspection_id = inspection_id


class RubricInspectButton(Button):
    """Mouse-only Inspector affordance that never retains keyboard focus."""

    can_focus = False


class _RubricPhaseBubble(Vertical):
    """Shared lifecycle presentation for one Rubric phase."""

    phase_name = "Phase"
    running_label = "Running"

    def __init__(
        self,
        run_id: str,
        pass_number: int,
        max_iterations: int,
        *,
        inspection_id: str = "",
    ) -> None:
        super().__init__(classes="message rubric rubric-phase")
        self.run_id = run_id
        self.pass_number = pass_number
        self.max_iterations = max_iterations
        self.inspection_id = inspection_id
        self.status = Static(classes="rubric-phase-status")
        self.inspect = RubricInspectButton(
            "Inspect",
            compact=True,
            classes="rubric-inspect",
        )
        self.inspect.styles.display = "block" if inspection_id else "none"
        self._started_at: float | None = time.monotonic()
        self._duration_ms: int | None = None
        self._frame = 0
        self._terminal_label = ""
        self._refresh_status()

    def set_inspection_id(self, inspection_id: str) -> None:
        if inspection_id:
            self.inspection_id = inspection_id
            self.inspect.styles.display = "block"

    def start(self, inspection_id: str = "") -> None:
        self.set_inspection_id(inspection_id)
        self._started_at = time.monotonic()
        self._duration_ms = None
        self._frame = 0
        self._terminal_label = ""
        self._refresh_status()

    def finish(self, *, succeeded: bool, duration_ms: int | None = None) -> None:
        self._duration_ms = (
            duration_ms
            if duration_ms is not None
            else (elapsed_ms(self._started_at) if self._started_at is not None else None)
        )
        self._started_at = None
        self._terminal_label = "Complete" if succeeded else "Failed"
        self._refresh_status()

    def interrupt(self) -> None:
        if self._started_at is None:
            return
        self._duration_ms = elapsed_ms(self._started_at)
        self._started_at = None
        self._terminal_label = "Interrupted"
        self._refresh_status()

    def tick(self) -> None:
        if self._started_at is None:
            return
        self._frame += 1
        self._refresh_status()

    @on(Button.Pressed, ".rubric-inspect")
    def inspect_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.inspection_id:
            self.post_message(RubricInspectionSelected(self.inspection_id))

    def _refresh_status(self) -> None:
        value = Text()
        if self._started_at is not None:
            frame = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            value.append(f"{frame} ", style=TOOL_DURATION_COLOR)
            value.append(self.running_label, style=TOOL_RUNNING_COLOR)
            value.append(
                f" · {format_elapsed(elapsed_ms(self._started_at))} elapsed",
                style=TOOL_DURATION_COLOR,
            )
        else:
            color = {
                "Complete": TOOL_COMPLETED_COLOR,
                "Failed": TOOL_FAILED_COLOR,
                "Interrupted": TOOL_CANCELLED_COLOR,
            }.get(self._terminal_label, RUBRIC_BODY_COLOR)
            value.append(self._terminal_label or "Complete", style=color)
            if self._duration_ms is not None:
                value.append(
                    f" · {format_elapsed(self._duration_ms)}",
                    style=TOOL_DURATION_COLOR,
                )
        self.status.update(value)

    def _phase_border_title(self) -> str:
        return (
            f"{self.phase_name.lower()} · pass "
            f"{self.pass_number}/{self.max_iterations}"
        )


class RubricVerifierBubble(_RubricPhaseBubble):
    """Compact Verifier lifecycle and authoritative tool count."""

    phase_name = "Verifier"
    running_label = "Verifying"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = self._phase_border_title()
        self.tool_count = Static("0 tools called", classes="rubric-tool-count")
        self._tool_ids: set[str] = set()
        self._idless_tools = 0

    def compose(self) -> ComposeResult:
        yield self.status
        with Horizontal(classes="rubric-phase-actions"):
            yield self.tool_count
            yield self.inspect

    @property
    def renderable(self) -> Text:
        value = Text(str(self.border_title))
        value.append("\n")
        if isinstance(self.status.content, Text):
            value.append_text(self.status.content)
        value.append(f"\n{self.tool_count.content}")
        if self.inspection_id:
            value.append("\nInspect")
        return value

    def tool_started(self, call_id: str = "") -> None:
        if call_id:
            if call_id in self._tool_ids:
                return
            self._tool_ids.add(call_id)
        else:
            self._idless_tools += 1
        self._refresh_tool_count()

    def set_tool_count(self, count: int) -> None:
        self._tool_ids = {f"restored:{index}" for index in range(max(0, count))}
        self._idless_tools = 0
        self._refresh_tool_count()

    def _refresh_tool_count(self) -> None:
        count = len(self._tool_ids) + self._idless_tools
        noun = "tool" if count == 1 else "tools"
        self.tool_count.update(f"{count} {noun} called")


class RubricGraderBubble(_RubricPhaseBubble):
    """Grader lifecycle plus the existing human-friendly verdict."""

    phase_name = "Grader"
    running_label = "Evaluating evidence"

    def __init__(
        self,
        *args: Any,
        grader_model: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.grader_model = str(grader_model or "")
        self.border_title = self._phase_border_title()
        self.model = Static(
            Text(f"Model: {self.grader_model}", style=RUBRIC_BODY_COLOR),
            classes="rubric-grader-model",
        )
        self.result = Static(classes="rubric-result")
        self.model.styles.display = "block" if self.grader_model else "none"
        self.result.styles.display = "none"

    def compose(self) -> ComposeResult:
        yield self.model
        with Horizontal(classes="rubric-phase-actions"):
            yield self.status
            yield self.inspect
        yield self.result

    @property
    def renderable(self) -> Text:
        value = Text(str(self.border_title))
        if self.grader_model:
            value.append(f"\nModel: {self.grader_model}")
        value.append("\n")
        if isinstance(self.status.content, Text):
            value.append_text(self.status.content)
        if self.inspection_id:
            value.append("\nInspect")
        if self.result.styles.display != "none" and isinstance(self.result.content, Text):
            value.append("\n")
            value.append_text(self.result.content)
        return value

    def set_result(self, result: Text) -> None:
        self.result.update(result)
        self.result.styles.display = "block"


__all__ = [
    "RubricGraderBubble",
    "RubricInspectionSelected",
    "RubricVerifierBubble",
]
