"""Textual Tree cache probe: Tree.refresh() vs TreeNode.refresh().

Purpose
-------
Test the hypothesis that a Workflow timer can appear frozen because Textual's
Tree caches rendered line Strips using Tree/node update counters.

This probe has NO MIRA runtime, LangGraph, agents, LLM calls, or Workflow
events. It isolates Textual Tree repaint behavior.

Two rows render elapsed time from time.monotonic():

    tree.refresh only
        tick calls Tree.refresh()

    node.refresh
        tick calls TreeNode.refresh()

If the hypothesis is correct on the installed Textual version:

- "tree.refresh only" will remain visually stale or jump only after some
  unrelated invalidation.
- "node.refresh" will advance once per second.

Run from anywhere in the MIRA environment:

    python probe_textual_tree_refresh_cache.py

Press:
    q / Esc   quit
    r         force a structural invalidation of the whole Tree

The "r" key is useful: if the stale first row suddenly jumps to the current
elapsed time after pressing it, that is additional evidence that the rendered
value was correct but the cached Tree line was not being invalidated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import textual
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, Tree
from textual.widgets.tree import TreeNode


@dataclass(slots=True)
class TimerRow:
    name: str
    started_at: float
    mode: str


class TimerTree(Tree[TimerRow]):
    """Tree whose labels compute elapsed time dynamically during rendering."""

    def __init__(self) -> None:
        super().__init__("root", id="timer-tree")
        self.show_root = False
        self.show_guides = True
        self.auto_expand = False

        now = time.monotonic()

        self.tree_refresh_row = self.root.add_leaf(
            "tree.refresh only",
            data=TimerRow(
                name="tree.refresh only",
                started_at=now,
                mode="tree",
            ),
        )

        self.node_refresh_row = self.root.add_leaf(
            "node.refresh",
            data=TimerRow(
                name="node.refresh",
                started_at=now,
                mode="node",
            ),
        )

        self.root.expand()

    def render_label(
        self,
        node: TreeNode[TimerRow],
        base_style: Style,
        style: Style,
    ) -> Text:
        data = node.data

        if not isinstance(data, TimerRow):
            return super().render_label(node, base_style, style)

        elapsed = int(max(0.0, time.monotonic() - data.started_at))

        label = Text()
        label.append(f"{data.name:<24}", style="bold")
        label.append(f"{elapsed:02d}s", style="cyan")

        if data.mode == "tree":
            label.append("   <- only Tree.refresh()", style="dim")
        else:
            label.append("   <- TreeNode.refresh()", style="dim")

        return label


class TreeRefreshCacheProbe(App[None]):
    CSS = """
    Screen {
        background: #0c0f10;
        color: #e8edef;
    }

    #body {
        width: 1fr;
        height: auto;
        margin: 1 2;
        border: solid #5BB8B1;
        padding: 1 2;
    }

    #title {
        height: 1;
        text-style: bold;
        color: #5BB8B1;
        margin-bottom: 1;
    }

    #textual-version {
        height: 1;
        color: #82909a;
    }

    #instructions {
        height: auto;
        color: #c9d5d8;
        margin-bottom: 1;
    }

    #timer-tree {
        width: 1fr;
        height: 4;
        background: transparent;
        border: none;
        padding: 0;
    }

    #tick-count {
        height: 1;
        color: #82909a;
        margin-top: 1;
    }

    #verdict {
        height: auto;
        margin-top: 1;
        color: #e2b44f;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("r", "force_invalidate", "Force invalidate"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.tick_count = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield Static(
                "Textual Tree cache probe",
                id="title",
            )

            yield Static(
                f"Textual {textual.__version__}",
                id="textual-version",
            )

            yield Static(
                "Both rows calculate elapsed time from time.monotonic().\n"
                "Every second the app calls Tree.refresh() for the whole Tree, "
                "then TreeNode.refresh() ONLY for the second row.\n\n"
                "Expected if the cache hypothesis is correct:\n"
                "  tree.refresh only   -> visually stale / jumpy\n"
                "  node.refresh        -> advances every second\n\n"
                "Press R at any time to force a structural Tree invalidation. "
                "If the stale first row suddenly catches up, that is strong evidence "
                "of cached line reuse.",
                id="instructions",
            )

            yield TimerTree()

            yield Static(
                "animation ticks: 0",
                id="tick-count",
            )

            yield Static(
                "Watch the two elapsed values for ~10 seconds.",
                id="verdict",
            )

    def on_mount(self) -> None:
        # One-second cadence makes the effect unambiguous.
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        self.tick_count += 1

        tree = self.query_one("#timer-tree", TimerTree)

        # Control case:
        #
        # Ask the entire widget to repaint, but DO NOT change Tree/node update
        # counters. If Tree's line Strip cache key remains valid, render_label()
        # need not be called again for this row.
        tree.refresh()

        # Test case:
        #
        # Public TreeNode.refresh() increments this node's update counter before
        # repainting its line. That should invalidate this row's cached Strip.
        tree.node_refresh_row.refresh()

        self.query_one("#tick-count", Static).update(
            f"animation ticks: {self.tick_count}"
        )

        if self.tick_count == 3:
            self.query_one("#verdict", Static).update(
                "After 3 seconds: if the second value is advancing but the first "
                "is not, the cache hypothesis is reproduced."
            )

        if self.tick_count == 8:
            self.query_one("#verdict", Static).update(
                "Press R now. If the first row jumps to ~08s/current time, "
                "Tree.refresh() was repainting a cached line."
            )

    def action_force_invalidate(self) -> None:
        tree = self.query_one("#timer-tree", TimerTree)

        # Use public state to force Tree's normal structural invalidation path:
        # changing guide_depth invokes Tree.watch_guide_depth() -> _invalidate().
        #
        # Toggle it and restore immediately. This deliberately does NOT call the
        # private _invalidate() method from the probe.
        original = tree.guide_depth
        tree.guide_depth = 3 if original != 3 else 4
        tree.guide_depth = original

        self.query_one("#verdict", Static).update(
            "Forced Tree invalidation. Check whether 'tree.refresh only' "
            "immediately caught up to the current elapsed time."
        )


if __name__ == "__main__":
    TreeRefreshCacheProbe().run()
