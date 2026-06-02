"""Modal dialogs used by the Textual app."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea


class ChoiceScreen(ModalScreen[str]):
    """Small modal choice prompt."""

    def __init__(self, title: str, message: str, choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.choices = choices
        self.keys = {value.lower(): value for value, _ in choices}

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Static(self.title, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            with Horizontal(classes="dialog-buttons"):
                for value, label in self.choices:
                    yield Button(label, id=f"choice-{value}")

    def on_mount(self) -> None:
        """Focus the first button."""
        buttons = self.query(Button)
        if len(buttons) > 0:
            buttons.first().focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        """Return the selected value."""
        self.dismiss((event.button.id or "").removeprefix("choice-"))

    def on_key(self, event: Key) -> None:
        """Allow direct keyboard selection."""
        value = self.keys.get(event.key.lower())
        if value is None:
            return
        event.stop()
        self.dismiss(value)


class TextPromptScreen(ModalScreen[str | None]):
    """Modal freeform text prompt."""

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Static(self.title, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            yield Input(id="answer")
            with Horizontal(classes="dialog-buttons"):
                yield Button("submit", id="submit")
                yield Button("cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the text input."""
        self.query_one("#answer", Input).focus()

    @on(Input.Submitted)
    def submit_input(self, event: Input.Submitted) -> None:
        """Return the entered text."""
        self.dismiss(event.value)

    @on(Button.Pressed)
    def submit_button(self, event: Button.Pressed) -> None:
        """Return or cancel from a button."""
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self.dismiss(self.query_one("#answer", Input).value)

    def on_key(self, event: Key) -> None:
        """Close the prompt with Escape."""
        if event.key == "escape":
            event.stop()
            self.dismiss(None)


class JSONEditScreen(ModalScreen[str | None]):
    """Modal JSON editor used for edited tool approvals."""

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.title = title
        self.text = text

    def compose(self) -> ComposeResult:
        with Container(classes="dialog json-dialog"):
            yield Static(self.title, classes="dialog-title")
            yield TextArea(
                self.text,
                language="json",
                show_line_numbers=False,
                id="json-editor",
            )
            with Horizontal(classes="dialog-buttons"):
                yield Button("save", id="save")
                yield Button("cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the JSON editor."""
        self.query_one("#json-editor", TextArea).focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        """Return edited JSON or cancel."""
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self.dismiss(self.query_one("#json-editor", TextArea).text)

    def on_key(self, event: Key) -> None:
        """Close the editor with Escape."""
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
