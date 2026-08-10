"""View-only screen for current startup and resource issues."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static

from runtime.issues import Issue, sort_issues


class IssuesScreen(ModalScreen[None]):
    """Show a stable, flat, initially collapsed list of current Issues."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, issues: list[Issue]) -> None:
        super().__init__()
        self.issues = sort_issues(issues)

    def compose(self) -> ComposeResult:
        with Vertical(id="issues-dialog"):
            with Horizontal(id="issues-title-row"):
                yield Static("ISSUES", id="issues-title")
                yield Button("x", id="issues-title-close", classes="panel-close")
            with VerticalScroll(id="issues-scroll"):
                for issue in self.issues:
                    body = "\n\n".join(
                        (
                            f"Location\n{issue.location}",
                            f"Details\n{issue.details}",
                            f"Guidance\n{issue.guidance}",
                        )
                    )
                    with Collapsible(title=f"[{issue.category}] {issue.summary}", collapsed=True):
                        yield Static(body, markup=False, classes="issue-details")

    def action_close(self) -> None:
        self.dismiss()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "issues-title-close":
            event.stop()
            self.dismiss()


__all__ = ["IssuesScreen"]
