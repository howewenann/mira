"""Disposable Textual preview for assistant-bubble Copy button styles."""

from __future__ import annotations

from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static


SAMPLE_RESPONSE = (
    "I’ve updated the configuration. Everything is ready to use."
)

VARIANTS = (
    ("1 · Exact new+ baseline", "baseline"),
    ("2 · Subtle background", "subtle"),
    ("3 · Very-light border", "light-border"),
    ("4 · Balanced recommendation", "balanced"),
)


class AssistantBubble(Vertical):
    """One MIRA-like assistant bubble containing a real Copy button."""

    def __init__(self, label: str, variant: str, timestamp: str) -> None:
        super().__init__(classes="message assistant")
        self.variant = variant
        self.variant_label = label
        self.border_title = "mira"
        self.border_subtitle = timestamp

    def compose(self) -> ComposeResult:
        yield Static(SAMPLE_RESPONSE, classes="response")
        with Horizontal(classes="copy-actions"):
            yield Button(
                "Copy",
                classes=f"copy-button {self.variant}",
                compact=True,
            )
            yield Static(self.variant_label, classes="button-caption")


class CopyButtonPreview(App[None]):
    """Compare four compact Copy treatments in realistic assistant bubbles."""

    CSS = """
    Screen {
        background: #0c0f10;
        color: #e8edef;
    }

    #preview-title {
        height: 2;
        padding: 0 2;
        content-align: left middle;
        color: #eef7f8;
        text-style: bold;
        background: #101516;
    }

    #chat-log {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        border: solid #2d5661;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    .message {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: solid #3c4a50;
        color: #e8edef;
    }

    .message.assistant {
        border: solid #5BB8B1;
    }

    .button-caption {
        width: 1fr;
        height: 1;
        margin-left: 1;
        color: #82909A;
        text-style: italic;
    }

    .response {
        height: auto;
        color: #e8edef;
    }

    .copy-actions {
        width: 1fr;
        height: 1;
        min-height: 1;
        margin-top: 1;
        align-horizontal: left;
    }

    Button.copy-button {
        width: 10;
        min-width: 10;
        max-width: 10;
        height: 1;
        min-height: 1;
        margin: 0;
        content-align: center middle;
        text-align: center;
    }

    /* Variant 1: the current #new-chat treatment, adapted from "+ New". */
    Button.copy-button.baseline {
        padding: 0;
        border: none;
        background: transparent;
        color: #7ce3dc;
        text-style: bold;
    }

    Button.copy-button.baseline:focus,
    Button.copy-button.baseline:hover {
        background: #1b3036;
        color: #eef7f8;
    }

    /* Variant 2: always-visible but deliberately low-contrast fill. */
    Button.copy-button.subtle {
        padding: 0 1;
        border: none;
        background: #151f22;
        color: #b8d6d5;
        text-style: none;
    }

    Button.copy-button.subtle:focus,
    Button.copy-button.subtle:hover {
        background: #1b3036;
        color: #eef7f8;
        text-style: none;
    }

    /* Variant 3: a one-row outline using only the side cells. */
    Button.copy-button.light-border {
        padding: 0;
        border: none;
        border-left: tall #56616a;
        border-right: tall #56616a;
        background: transparent;
        color: #b8c1c7;
    }

    Button.copy-button.light-border:focus,
    Button.copy-button.light-border:hover {
        border-left: tall #5BB8B1;
        border-right: tall #5BB8B1;
        background: #122023;
        color: #eef7f8;
    }

    /* Variant 4: a quiet teal fill makes the action legible at rest. */
    Button.copy-button.balanced {
        padding: 0 1;
        border: none;
        background: #17313a;
        color: #d6fff6;
        text-style: bold;
    }

    Button.copy-button.balanced:focus,
    Button.copy-button.balanced:hover {
        background: #24505a;
        color: #ffffff;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._feedback_versions: dict[Button, int] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "Assistant Copy button preview · click or Tab/Enter to compare",
            id="preview-title",
        )
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        with VerticalScroll(id="chat-log"):
            for label, variant in VARIANTS:
                yield AssistantBubble(label, variant, timestamp)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        version = self._feedback_versions.get(button, 0) + 1
        self._feedback_versions[button] = version
        button.label = "Copied"
        self.set_timer(
            1.5,
            lambda: self._restore_button(button, version),
        )

    def _restore_button(self, button: Button, version: int) -> None:
        """Restore Copy only if this is still the newest click timer."""
        if self._feedback_versions.get(button) == version:
            button.label = "Copy"


if __name__ == "__main__":
    CopyButtonPreview().run()
