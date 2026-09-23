"""Process-local completion bubble for a launched Workflow."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static


class WorkflowFinalStateSelected(Message):
    """Request the existing Inspector for one retained final state."""

    def __init__(self, workflow_name: str, final_state: Any) -> None:
        super().__init__()
        self.workflow_name = workflow_name
        self.final_state = final_state


class WorkflowCompletionBubble(Vertical):
    """Compact successful Workflow result with one process-local action."""

    def __init__(self, workflow_name: str, final_state: Any) -> None:
        super().__init__(classes="message workflow-completion")
        self.workflow_name = workflow_name
        self.final_state = final_state
        self.border_title = "workflow"

    def compose(self) -> ComposeResult:
        yield Static(
            Text(f"Workflow {self.workflow_name} completed"),
            classes="workflow-completion-body",
        )
        with Horizontal(classes="workflow-completion-actions"):
            yield Button(
                "Final state",
                classes="workflow-final-state",
                compact=True,
            )

    @on(Button.Pressed, ".workflow-final-state")
    def final_state_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(
            WorkflowFinalStateSelected(self.workflow_name, self.final_state)
        )


__all__ = ["WorkflowCompletionBubble", "WorkflowFinalStateSelected"]
