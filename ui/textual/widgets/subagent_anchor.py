"""Small transcript anchor for retrospective subagent navigation."""

from __future__ import annotations

from textual.widgets import Button


class SubagentAnchor(Button):
    """A terminal tool's durable link to its task-created child runs."""

    can_focus = False

    def __init__(self, anchor_id: str, count: int, **kwargs: object) -> None:
        super().__init__(f"Subagents · {count}", classes="subagent-anchor", **kwargs)
        self.anchor_id = anchor_id
