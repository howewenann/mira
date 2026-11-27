"""Standalone native Textual ListView + keyline probe (no custom widgets)."""
import argparse
import asyncio
from importlib.metadata import version
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Label, ListItem, ListView

STEPS = [
    ("thinking", "", "00:12", "#82909a"),
    ("read_file", "/README.md", "00:01", "#8fa3b8"),
    ("message", "", "00:02", "#5bb8b1"),
    ("execute", "python -m pytest tests/core/interface/test_operations.py --verbose", "00:08", "#8fa3b8"),
    ("rubric", "", "00:03", "#c58fd6"),
]


class KeylineProbe(App):
    CSS = """
    Screen { background: #0c0f10; }
    #page { padding: 1 2; }
    #activities {
        width: 1fr;
        height: auto;
        keyline: thin #51696c;
        background: #0c0f10;
        background-tint: transparent;
        overflow: hidden hidden;
    }
    #activities > ListItem {
        height: 1;
        min-height: 1;
        width: 1fr;
        margin: 1;
        padding: 0 1;
        border: none;
        background: #0c0f10;
        text-style: none;
    }
    #activities > ListItem.-hovered,
    #activities > ListItem.-highlight,
    #activities:focus > ListItem.-highlight {
        background: #17313a;
        text-style: none;
    }
    #activities:focus { background-tint: transparent; }
    .activity-content { height: 1; width: 1fr; }
    .glyph { width: 2; height: 1; text-style: none; }
    .name { width: auto; height: 1; text-style: none; }
    .preview {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #9aaab0;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    .duration { width: 5; height: 1; color: #82909a; text-align: right; }
    #status { margin-top: 1; height: auto; color: #82909a; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, rows=5):
        super().__init__()
        self.steps = [STEPS[i % len(STEPS)] for i in range(rows)]
        self.selected = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="page"):
            yield Label("Native ListView + keyline | Up/Down, Enter or click; q quits")
            with ListView(id="activities"):
                for i, (name, preview, duration, color) in enumerate(self.steps):
                    glyph = Label("▸", classes="glyph")
                    glyph.styles.color = color
                    label = Label(name, classes="name")
                    label.styles.color = color
                    yield ListItem(
                        Horizontal(glyph, label, Label(preview, markup=False, classes="preview"),
                                   Label(duration, classes="duration"), classes="activity-content"),
                        id=f"activity-{i}",
                    )
            yield Label("Select a row to test its disclosure glyph.", id="status")

    def on_mount(self):
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected):
        i = self.query_one(ListView).index
        if i is None:
            return
        self.selected = None if self.selected == i else i
        for j, item in enumerate(self.query(ListItem)):
            item.query_one(".glyph", Label).update("▾" if j == self.selected else "▸")
        self.query_one("#status", Label).update(
            "No selection" if self.selected is None else f"Selected: {self.steps[i][0]}"
        )


async def check(out):
    out.mkdir(parents=True, exist_ok=True)
    print("Textual", version("textual"))
    for width in (60, 110):
        app = KeylineProbe()
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            view = app.query_one(ListView)
            items = list(view.query(ListItem))
            print(f"Width {width}: list height={view.size.height}; row heights={[i.size.height for i in items]}; y={[i.region.y for i in items]}")
            await pilot.press("down", "enter")
            await pilot.pause()
            print("Selected index:", app.selected, "expected: 1")
            print("Duration right edges:", [i.query_one('.duration').region.right for i in items])
            path = out / f"keyline_probe_{width}.svg"
            path.write_text(app.export_screenshot(), encoding="utf-8")
            print("Screenshot:", path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path('.'))
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    if args.check:
        asyncio.run(check(args.out))
    else:
        KeylineProbe(rows=max(1, args.rows)).run()
