"""Isolate MIRA f98e83b's two-line TurnSummary focus outline.
Run from probe/: python summary_focus_probe.py [--check]
Uses only Textual and Rich, no MIRA imports or custom border rendering.
"""
import argparse
import asyncio
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import __version__ as textual_version
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Static, Input


# The grid and focus/click behavior match the current MIRA implementation.
def activity_line(label: Text, duration: str, expanded: bool) -> Table:
    line = Table.grid(expand=True, padding=0)
    line.add_column(width=2, no_wrap=True)
    line.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    line.add_column(width=max(6, len(duration)), justify="right", no_wrap=True)
    line.add_row("▾ " if expanded else "▸ ", label, Text(duration, style="dim"))
    return line


class TurnSummary(Static, can_focus=True):
    BINDINGS = [("enter,space", "toggle", "Toggle activity")]

    def __init__(self, *, id: str) -> None:
        super().__init__(classes="turn-summary", id=id)

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus(scroll_visible=False)
        self.action_toggle()

    def action_toggle(self) -> None:
        self.parent.toggle_history()


class SampleTurn(Vertical):
    def __init__(self, variant: str) -> None:
        super().__init__(id=variant, classes="assistant-turn")
        self.border_title = "Assistant"
        self.summary = TurnSummary(id=f"{variant}-summary")
        self.history = Static("Example completed activity", classes="history")
        self.expanded = False
        self.tick()

    def compose(self) -> ComposeResult:
        yield self.summary
        yield self.history
        yield Static("Final response remains visible.", classes="answer")

    def toggle_history(self) -> None:
        self.expanded = not self.expanded
        self.tick()

    def tick(self) -> None:
        header = activity_line(Text("Worked · 5 steps"), "00:18", self.expanded)
        counts = Text("2 Thinking", style="#B8C2C9")
        counts.append(" · ")
        counts.append("2 Tools", style="#8FA3B8")
        counts.append(" · ")
        counts.append("1 Message", style="#5BB8B1")
        header.add_row("", counts, "")
        self.summary.update(header, layout=False)
        self.history.display = self.expanded


class SummaryProbe(App):
    CSS = """
    Screen { background: #0c0f10; color: #e8edef; padding: 1; }
    .assistant-turn { height: auto; border: solid #5bb8b1; padding: 0 1; margin-bottom: 1; }
    .turn-summary {
        height: 2;
        padding: 0 1;
        background: #17313a;
        color: #e8edef;
        text-style: none;
        pointer: pointer;
    }
    .turn-summary:hover, .turn-summary:focus {
        background: #1b3036;
        text-style: none;
    }
    /* A: historical f98e83b outline. B: validated correction, no overlay. */
    .turn-summary:focus { outline: solid #5bb8b1; }
    #fixed .turn-summary:focus { outline: none; }
    .history { height: auto; color: #8fa3b8; }
    .answer { height: auto; margin-top: 1; }
    .caption { height: 1; color: #b8c2c9; }
    Input { height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static(f"Textual {textual_version} | Click A and B; Tab / Enter / Space", classes="caption")
        yield Static("A: Historical MIRA CSS (focus outline)", classes="caption")
        yield SampleTurn("original")
        yield Static("B: Same widget, only focus outline removed", classes="caption")
        yield SampleTurn("fixed")
        yield Input(placeholder="Click here to clear summary focus", id="prompt")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()


async def check(directory: Path) -> None:
    """Export real Textual screenshots; geometry alone cannot prove visible text."""
    directory.mkdir(parents=True, exist_ok=True)
    for width in (60, 110):
        app = SummaryProbe()
        async with app.run_test(size=(width, 32)) as pilot:
            await pilot.pause()
            app.save_screenshot(f"{width}-unfocused.svg", path=str(directory))
            for variant in ("original", "fixed"):
                summary = app.query_one(f"#{variant}-summary", TurnSummary)
                turn = app.query_one(f"#{variant}", SampleTurn)
                before = summary.size.height
                assert before == 2, (variant, before)
                await pilot.click(summary)
                await pilot.pause()
                assert summary.has_focus, variant
                assert turn.expanded, variant
                assert summary.size.height == before, variant
                lines = [strip.text for strip in summary.render_lines(summary.size.region)]
                if variant == "original":
                    assert all("Worked" not in line and "Thinking" not in line for line in lines), lines
                else:
                    assert "Worked" in lines[0] and "Thinking" in lines[1], lines
                app.save_screenshot(f"{width}-{variant}-focused.svg", path=str(directory))
                print(f"{width} cols / {variant}: height={summary.size.height}, "
                      f"content={summary.content_size}, outline={summary.styles.outline}, "
                      f"focused={summary.has_focus}")
                await pilot.press("enter")
                await pilot.pause()
                assert not turn.expanded, variant
                assert summary.size.height == before, variant
                await pilot.press("space")
                await pilot.pause()
                assert turn.expanded, variant
                await pilot.press("enter")
                await pilot.pause()
                assert not turn.expanded, variant
                app.query_one("#prompt", Input).focus()
                await pilot.pause()
    print(f"Geometry and toggling passed. Inspect actual focused SVG frames: {directory}")
    print("Expected: A's outline obscures the two rows; B retains both lines.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path('artifacts/summary_focus'))
    args = parser.parse_args()
    if args.check:
        asyncio.run(check(args.out))
    else:
        SummaryProbe().run()
