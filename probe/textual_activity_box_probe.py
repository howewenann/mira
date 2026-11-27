"""Standalone Textual activity-row probe; no MIRA imports or modifications.
Run: python textual_activity_box_probe.py --mode native
     python textual_activity_box_probe.py --mode joined
     python textual_activity_box_probe.py --check
"""
import argparse
import asyncio
from importlib.metadata import version
from pathlib import Path

from rich.cells import cell_len
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

ROWS = [
    ("thinking", "", "00:12", "#82909A"),
    ("read_file", "/README.md", "00:01", "#8FA3B8"),
    ("message", "", "00:02", "#5BB8B1"),
    ("execute", "python -m pytest tests/core/" + "very_long_path/" * 10 + "test_operations.py --verbose", "00:08", "#8FA3B8"),
    ("rubric", "", "00:03", "#C58FD6"),
]

class ActivityRow(Static):
    can_focus = True

    def __init__(self, index, item, mode):
        super().__init__(id=f"row-{index}", classes=f"activity {mode}")
        self.item = item
        self.selected = False

    def render(self):
        name, args, duration, color = self.item
        width = max(0, self.content_size.width)
        left = Text(("▾" if self.selected else "▸") + " " + name, style=color)
        if args:
            left.append(" " + " ".join(args.split()), style="#9aaab0")
        room = max(0, width - cell_len(duration) - 1)
        left.truncate(room, overflow="ellipsis", pad=True)
        left.append(" " + duration, style="#82909A")
        left.truncate(width, overflow="crop")
        return left

    def on_resize(self):
        self.refresh()

    def on_click(self):
        self.app.select_row(self)
        self.focus()

    def on_key(self, event):
        if event.key in ("enter", "space"):
            self.app.select_row(self)
            event.stop()

class Separator(Static):
    """One literal terminal line; not a replacement rendering framework."""
    def __init__(self, position):
        super().__init__(classes="separator")
        self.position = position

    def render(self):
        left, right = {"top": ("┌", "┐"), "middle": ("├", "┤"), "bottom": ("└", "┘")}[self.position]
        width = self.content_size.width
        return Text(left + "─" * max(0, width - 2) + right, style="#51696c")

    def on_resize(self):
        self.refresh()

class Probe(App):
    CSS = """
    Screen { background: #0c0f10; color: #e8edef; padding: 1 2; }
    #heading, #status { height: 1; margin: 0; padding: 0; }
    #heading { background: #17313a; color: #5bb8b1; }
    #status { color: #82909a; }
    #activity-list { height: auto; width: 100%; padding: 0; margin: 0; }
    .activity {
        width: 100%; margin: 0; padding: 0 1;
        min-height: 1; background: #0c0f10;
        text-wrap: nowrap; text-style: none;
        border: none;
    }
    .activity.full { height: 3; border: solid #51696c; }
    .activity.native {
        height: 2;
        border-top: solid #51696c;
        border-left: solid #51696c;
        border-right: solid #51696c;
    }
    .activity.native.last {
        height: 3;
        border-bottom: solid #51696c;
    }
    .activity.joined, .activity.flat {
        height: 1;
        border-left: solid #51696c;
        border-right: solid #51696c;
    }
    .separator { height: 1; width: 100%; margin: 0; padding: 0; }
    .activity:hover, .activity:focus, .activity.selected {
        background: #17313a; text-style: none;
    }
    """
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def compose(self) -> ComposeResult:
        yield Static(f"Textual {version('textual')} | {self.mode} | Tab / Enter / click | Ctrl+C quits", id="heading")
        with Vertical(id="activity-list"):
            if self.mode == "joined":
                yield Separator("top")
            for i, item in enumerate(ROWS):
                if self.mode == "joined" and i:
                    yield Separator("middle")
                row = ActivityRow(i, item, self.mode)
                if self.mode == "native" and i == len(ROWS) - 1:
                    row.add_class("last")
                yield row
            if self.mode == "joined":
                yield Separator("bottom")
        yield Static("Select an activity to test stable glyphs and highlighting.", id="status")

    def select_row(self, row):
        selected = not row.selected
        for other in self.query(ActivityRow):
            other.selected = selected and other is row
            other.set_class(other.selected, "selected")
            other.refresh()
        self.query_one("#status", Static).update(f"Selected: {row.item[0]}" if selected else "No selection")

async def check(out):
    out.mkdir(parents=True, exist_ok=True)
    expected = {"full": 15, "native": 11, "joined": 11, "flat": 5}
    failures = []
    print("Textual:", version("textual"))
    for mode in expected:
        for width in (110, 60):
            app = Probe(mode)
            async with app.run_test(size=(width, 24)) as pilot:
                await pilot.pause()
                rows = list(app.query(ActivityRow))
                height = app.query_one("#activity-list").region.height
                before = [(r.region, r.content_size) for r in rows]
                print(f"{mode:7} {width:3} cols: height={height}, expected={expected[mode]}, content heights={[r.content_size.height for r in rows]}")
                if height != expected[mode] or any(r.content_size.height != 1 for r in rows):
                    failures.append(f"{mode}/{width}: incorrect geometry")
                (out / f"{mode}-{width}.svg").write_text(app.export_screenshot(), encoding="utf-8")
                await pilot.click("#row-3")
                await pilot.pause()
                if not rows[3].selected or before != [(r.region, r.content_size) for r in rows]:
                    failures.append(f"{mode}/{width}: selection or geometry changed")
                for r in rows:
                    if not r.render().plain.endswith(r.item[2]):
                        failures.append(f"{mode}/{width}: missing right-aligned time {r.id}")
                (out / f"{mode}-{width}-selected.svg").write_text(app.export_screenshot(), encoding="utf-8")
                await pilot.press("enter")
                await pilot.pause()
                if rows[3].selected:
                    failures.append(f"{mode}/{width}: Enter failed to deselect")
    print("SVG screenshots:", out.resolve())
    print("Inspect junctions and glyph appearance visually; geometry alone cannot prove those.")
    if failures:
        raise SystemExit("FAIL:\n" + "\n".join(failures))
    print("PASS: measured geometry, selection, keyboard toggle, and duration preservation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "native", "joined", "flat"), default="native")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("activity-box-probe-results"))
    args = parser.parse_args()
    if args.check:
        asyncio.run(check(args.out))
    else:
        Probe(args.mode).run()
