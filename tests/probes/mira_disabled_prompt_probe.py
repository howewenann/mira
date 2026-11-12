#!/usr/bin/env python
"""Reproduce MIRA PromptBox editing while disabled.

Uses MIRA's ACTUAL PromptBox, not a stand-in, in a minimal Textual app.
Runs a headless before/after comparison without editing repository files.

Run from the MIRA repository root:
    python /path/to/mira_disabled_prompt_probe.py
    python /path/to/mira_disabled_prompt_probe.py --interactive

The comparison's "patched" variant applies Widget.watch_disabled()
before MIRA's current watcher, mirroring the proposed super() fix.
The interactive mode tests an actual mouse and keyboard in your terminal.

No model, MCP, network, project or settings changes.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def locate_repository() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for root in candidates:
        if (root / "ui" / "textual" / "widgets" / "prompt_box.py").is_file():
            return root
    raise SystemExit(
        "Cannot find the MIRA source. Run from the repository root, e.g. "
        r"cd D:\Projects\mira"
    )


sys.path.insert(0, str(locate_repository()))

try:
    import textual
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widget import Widget
    from textual.widgets import Button, RichLog, Static, TextArea
    from ui.textual.widgets.prompt_box import PromptBox
except ImportError as exc:
    raise SystemExit(
        f"Missing MIRA / Textual dependency: {exc}\n"
        "Activate the MIRA Conda environment and run from the repo root."
    ) from exc


class ProposedFixPromptBox(PromptBox):
    """Only change the original watch_disabled behavior for comparison."""

    def watch_disabled(self, disabled: bool) -> None:
        Widget.watch_disabled(self, disabled)
        super().watch_disabled(disabled)


class ProbeApp(App[None]):
    CSS = """
    Screen { background: #0c0f10; color: #e8edef; }
    #explanation { height: auto; padding: 1; }
    #prompt { height: 6; width: 100%; border: heavy #5bb8b1; }
    #outside { width: 26; height: 3; }
    #state { height: 2; padding: 0 1; color: #7ce3dc; }
    #events { height: 1fr; border: solid #51696c; }
    """
    BINDINGS = [
        Binding("f2", "disable_prompt", "Disable"),
        Binding("f3", "enable_prompt", "Enable"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, prompt_class: type[PromptBox], interactive: bool = False) -> None:
        super().__init__()
        self.prompt_class = prompt_class
        self.interactive = interactive
        self.submissions: list[str] = []
        self.history: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"MIRA DISABLED-PROMPT PROBE — {self.prompt_class.__name__}\n"
                "F2: disable | F3: enable | Click the button, then click the prompt "
                "| Ctrl+Q: quit\n"
                "After disabling, try typing and pressing Enter. Log below records edits.",
                id="explanation",
            )
            yield self.prompt_class()
            yield Button("Click here to blur prompt", id="outside")
            yield Static("", id="state")
            yield RichLog(id="events", wrap=True, markup=False)

    def on_mount(self) -> None:
        self.query_one(PromptBox).focus()
        if self.interactive:
            self.set_interval(0.15, self.update_state)
        self.update_state()
        self.log_event("Started: prompt enabled.")

    def update_state(self) -> None:
        prompt = self.query_one(PromptBox)
        self.query_one("#state", Static).update(
            f"disabled={prompt.disabled} | focus={self.screen.focused is prompt}"
            f" | text={prompt.value!r} | submissions={len(self.submissions)}"
        )

    def log_event(self, value: str) -> None:
        self.history.append(value)
        self.query_one("#events", RichLog).write(value)
        self.update_state()

    @on(TextArea.Changed, "#prompt")
    def record_edit(self, event: TextArea.Changed) -> None:
        self.log_event(f"EDIT while disabled={event.text_area.disabled}: "
                       f"text now {event.text_area.text!r}")

    def on_prompt_box_submitted(self, message: PromptBox.Submitted) -> None:
        self.submissions.append(message.value)
        self.log_event(f"ENTER submitted while disabled={message.prompt.disabled}: "
                       f"text={message.value!r}")

    def action_disable_prompt(self) -> None:
        self.query_one(PromptBox).disabled = True
        self.log_event("F2: set disabled=True")

    def action_enable_prompt(self) -> None:
        self.query_one(PromptBox).disabled = False
        self.query_one(PromptBox).focus()
        self.log_event("F3: set disabled=False and focus prompt")


def focus_name(app: ProbeApp) -> str:
    widget = app.screen.focused
    return getattr(widget, "id", None) or type(widget).__name__ if widget else "(none)"


async def run_case(label: str, prompt_type: type[PromptBox]) -> dict:
    app = ProbeApp(prompt_type)
    async with app.run_test(size=(85, 24)) as pilot:
        prompt = app.query_one(PromptBox)

        # Establish that simulated typing really reaches the enabled prompt.
        prompt.focus()
        await pilot.pause()
        initially_focused = app.screen.focused is prompt
        await pilot.press("a")
        await pilot.pause()
        enabled_input = prompt.value == "a"

        # Disable WHILE the prompt is focused (the user's reload scenario).
        prompt.disabled = True
        await pilot.pause()
        focus_after_disable = focus_name(app)
        retained_focus = app.screen.focused is prompt
        before = prompt.value
        before_submits = len(app.submissions)
        await pilot.press("b")
        await pilot.pause()
        typed_while_disabled = prompt.value != before
        await pilot.press("enter")
        await pilot.pause()
        submitted_while_disabled = len(app.submissions) > before_submits

        # Now explicitly leave the prompt and try to RE-enter with the mouse.
        outside = app.query_one("#outside", Button)
        await pilot.click("#outside")
        # Ensure focus has actually been moved, to isolate mouse re-entry.
        outside.focus()
        await pilot.pause()
        outside_focus = focus_name(app)
        click_landed = await pilot.click("#prompt", offset=(3, 2))
        await pilot.pause()
        focus_after_click = focus_name(app)
        mouse_refocused = app.screen.focused is prompt
        before = prompt.value
        before_submits = len(app.submissions)
        await pilot.press("c")
        await pilot.press("enter")
        await pilot.pause()
        typed_after_mouse_click = prompt.value != before
        submitted_after_mouse_click = len(app.submissions) > before_submits

        # Enabled editing must still work after restoring availability.
        prompt.disabled = False
        prompt.focus()
        await pilot.pause()
        before = prompt.value
        await pilot.press("d")
        await pilot.pause()
        enabled_again = prompt.value == before + "d"

        return {
            "label": label,
            "initially_focused": initially_focused,
            "enabled_input": enabled_input,
            "focus_after_disable": focus_after_disable,
            "retained_focus": retained_focus,
            "typed_while_disabled": typed_while_disabled,
            "submitted_while_disabled": submitted_while_disabled,
            "outside_focus": outside_focus,
            "click_landed": click_landed,
            "focus_after_click": focus_after_click,
            "mouse_refocused": mouse_refocused,
            "typed_after_mouse_click": typed_after_mouse_click,
            "submitted_after_mouse_click": submitted_after_mouse_click,
            "enabled_again": enabled_again,
        }


async def run_automatic() -> None:
    print(f"Textual version: {getattr(textual, '__version__', 'see pip show textual')}")
    print(f"MIRA PromptBox: {Path(sys.modules[PromptBox.__module__].__file__).resolve()}")
    for label, cls in (
        ("CURRENT MIRA", PromptBox),
        ("WITH PROPOSED FOCUS FIX", ProposedFixPromptBox),
    ):
        out = await run_case(label, cls)
        print(f"\n===== {label} =====")
        print(f"Baseline enabled input works:      {out['enabled_input']}")
        print(f"Originally focused:               {out['initially_focused']}")
        print(f"Focus after disabled=True:       {out['focus_after_disable']}")
        print(f"Focus was retained:               {out['retained_focus']}")
        print(f"Typing while disabled changed text: {out['typed_while_disabled']}")
        print(f"Enter while disabled submitted:   {out['submitted_while_disabled']}")
        print(f"Moved focus outside:              {out['outside_focus']}")
        print(f"Mouse click landed in prompt:     {out['click_landed']}")
        print(f"Focus after clicking disabled:    {out['focus_after_click']}")
        print(f"Mouse refocused disabled prompt:  {out['mouse_refocused']}")
        print(f"Typing after mouse click changed text: {out['typed_after_mouse_click']}")
        print(f"Enter after mouse click submitted: {out['submitted_after_mouse_click']}")
        print(f"Editing works again when enabled: {out['enabled_again']}")
        if not out["enabled_input"] or not out["enabled_again"]:
            print("RESULT: INVALID BASELINE — check Textual test keyboard event delivery.")
        elif any(out[k] for k in (
            "typed_while_disabled",
            "submitted_while_disabled",
            "typed_after_mouse_click",
            "submitted_after_mouse_click",
        )):
            print("RESULT: DISABLED INPUT BUG REPRODUCED.")
        else:
            print("RESULT: No disabled input observed in this scenario.")
    print("\nNote: Pilot.click simulates pointer events in the Textual test driver.")
    print("For real terminal mouse verification, run with --interactive.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true",
                        help="Open terminal UI to click and type manually")
    parser.add_argument("--patched", action="store_true",
                        help="With --interactive, use the proposed watch_disabled fix")
    args = parser.parse_args()
    if args.interactive:
        cls = ProposedFixPromptBox if args.patched else PromptBox
        ProbeApp(cls, interactive=True).run()
    else:
        asyncio.run(run_automatic())


if __name__ == "__main__":
    main()
