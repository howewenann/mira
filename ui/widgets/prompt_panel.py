"""In-window prompt panel for interactive user decisions."""

from __future__ import annotations

import asyncio
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Button, Input, Static, TextArea


class PromptPanel(Vertical):
    """Focused prompt surface mounted in the main app layout."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="prompt-panel", **kwargs)
        self.display = False
        self._future: asyncio.Future[str | None] | None = None
        self._mode = ""
        self._shortcuts: dict[str, str] = {}
        self._button_values: dict[str, str] = {}

    @property
    def active(self) -> bool:
        """Return whether the panel is waiting for an answer."""
        return self._future is not None and not self._future.done()

    def compose(self) -> ComposeResult:
        """Compose the reusable prompt shell."""
        yield Static("", id="prompt-panel-title")
        with VerticalScroll(id="prompt-panel-body"):
            yield Static("", id="prompt-panel-message")
        yield Input(id="prompt-panel-input")
        yield TextArea("", show_line_numbers=False, id="prompt-panel-editor")
        yield Horizontal(id="prompt-panel-buttons")

    def on_mount(self) -> None:
        """Hide controls that only appear for some prompt modes."""
        self.query_one("#prompt-panel-input", Input).display = False
        self.query_one("#prompt-panel-editor", TextArea).display = False

    async def choose(self, title: str, message: str, choices: list[tuple[str, str]]) -> str | None:
        """Show a choice prompt and return the selected value."""
        await self._open("choice", title, message)
        self._shortcuts = choice_shortcuts(choices)
        buttons = self.query_one("#prompt-panel-buttons", Horizontal)
        for index, (value, label) in enumerate(choices):
            button_id = f"prompt-choice-{index}"
            self._button_values[button_id] = value
            buttons.mount(Button(label, id=button_id, classes="prompt-panel-button"))
        self.call_after_refresh(self._focus_first_button)
        return await self._wait()

    async def ask_text(self, title: str, message: str) -> str | None:
        """Show a freeform text prompt and return the entered value."""
        await self._open("text", title, message)
        answer = self.query_one("#prompt-panel-input", Input)
        answer.value = ""
        answer.display = True
        self._mount_action_buttons(("prompt-submit", "submit"), ("prompt-cancel", "cancel"))
        self.call_after_refresh(answer.focus)
        return await self._wait()

    async def edit_json(self, title: str, text: str) -> str | None:
        """Show a JSON editor prompt and return the edited text."""
        await self._open("json", title, "")
        self.query_one("#prompt-panel-body", VerticalScroll).display = False
        editor = self.query_one("#prompt-panel-editor", TextArea)
        editor.text = text
        editor.display = True
        self._mount_action_buttons(("prompt-save", "save"), ("prompt-cancel", "cancel"))
        self.call_after_refresh(editor.focus)
        return await self._wait()

    async def _open(self, mode: str, title: str, message: str) -> None:
        """Prepare the prompt shell for one active prompt."""
        if self._future is not None and not self._future.done():
            raise RuntimeError("prompt already active")

        self._mode = mode
        self._future = asyncio.get_running_loop().create_future()
        self._shortcuts = {}
        self._button_values = {}

        self.query_one("#prompt-panel-title", Static).update(title)
        self.query_one("#prompt-panel-message", Static).update(message)
        self.query_one("#prompt-panel-body", VerticalScroll).display = True
        self.query_one("#prompt-panel-input", Input).display = False
        self.query_one("#prompt-panel-editor", TextArea).display = False
        await self._clear_buttons()
        self.display = True

    async def _wait(self) -> str | None:
        """Wait for a prompt result and clean up the panel."""
        try:
            if self._future is None:
                return None
            return await self._future
        finally:
            await self._close()

    async def _close(self) -> None:
        """Hide the panel and clear prompt-specific state."""
        self.display = False
        self._mode = ""
        self._shortcuts = {}
        self._button_values = {}
        if not self.is_mounted:
            self._future = None
            return
        try:
            self.query_one("#prompt-panel-title", Static).update("")
            self.query_one("#prompt-panel-message", Static).update("")
            self.query_one("#prompt-panel-input", Input).value = ""
            self.query_one("#prompt-panel-editor", TextArea).text = ""
            self.query_one("#prompt-panel-body", VerticalScroll).display = True
            self.query_one("#prompt-panel-input", Input).display = False
            self.query_one("#prompt-panel-editor", TextArea).display = False
        except NoMatches:
            self._future = None
            return
        await self._clear_buttons()
        self._future = None

    def _mount_action_buttons(self, *buttons: tuple[str, str]) -> None:
        """Mount a stable footer button row."""
        container = self.query_one("#prompt-panel-buttons", Horizontal)
        for button_id, label in buttons:
            container.mount(Button(label, id=button_id, classes="prompt-panel-button"))

    async def _clear_buttons(self) -> None:
        """Remove all footer buttons from the previous prompt."""
        buttons = self.query_one("#prompt-panel-buttons", Horizontal)
        await buttons.remove_children()

    def _focus_first_button(self) -> None:
        """Focus the first footer button if one exists."""
        buttons = self.query(Button)
        if len(buttons) > 0:
            buttons.first().focus()

    def _focus_button_offset(self, offset: int) -> None:
        """Move focus across footer buttons by offset."""
        buttons = list(self.query(Button))
        if not buttons:
            return

        focused_index = next((index for index, button in enumerate(buttons) if button.has_focus), None)
        if focused_index is None:
            target_index = 0 if offset >= 0 else len(buttons) - 1
        else:
            target_index = (focused_index + offset) % len(buttons)
        buttons[target_index].focus()

    def _focused_button_value(self) -> str | None:
        """Return the choice value for the focused footer button."""
        for button in self.query(Button):
            if button.has_focus:
                return self._button_values.get(button.id or "")
        return None

    def _resolve(self, value: str | None) -> None:
        """Resolve the active prompt once."""
        if self._future is not None and not self._future.done():
            self._future.set_result(value)

    @on(Button.Pressed)
    def press_button(self, event: Button.Pressed) -> None:
        """Resolve prompts from footer button presses."""
        button_id = event.button.id or ""
        event.stop()

        if self._mode == "choice":
            self._resolve(self._button_values.get(button_id))
            return

        if button_id == "prompt-cancel":
            self._resolve(None)
            return

        if self._mode == "text" and button_id == "prompt-submit":
            self._resolve(self.query_one("#prompt-panel-input", Input).value)
            return

        if self._mode == "json" and button_id == "prompt-save":
            self._resolve(self.query_one("#prompt-panel-editor", TextArea).text)

    @on(Input.Submitted, "#prompt-panel-input")
    def submit_input(self, event: Input.Submitted) -> None:
        """Submit the freeform prompt with Enter."""
        event.stop()
        if self._mode == "text":
            self._resolve(event.value)

    def on_key(self, event: Key) -> None:
        """Handle direct shortcuts and cancel keys."""
        if self._mode == "choice":
            if event.key == "escape":
                event.stop()
                self._resolve(None)
                return

            if event.key in {"left", "up"}:
                event.stop()
                self._focus_button_offset(-1)
                return

            if event.key in {"right", "down"}:
                event.stop()
                self._focus_button_offset(1)
                return

            if event.key == "enter":
                value = self._focused_button_value()
                if value is not None:
                    event.stop()
                    self._resolve(value)
                return

            value = self._shortcuts.get(event.key.lower())
            if value is None:
                return
            event.stop()
            self._resolve(value)
            return

        if self._mode in {"text", "json"} and event.key == "escape":
            event.stop()
            self._resolve(None)


def choice_shortcuts(choices: list[tuple[str, str]]) -> dict[str, str]:
    """Return keyboard shortcuts for choice values and short labels."""
    shortcuts: dict[str, str] = {}
    for value, label in choices:
        value_key = value.lower()
        if value_key:
            shortcuts[value_key] = value
        label_key = label.strip().lower()
        if label_key:
            shortcuts.setdefault(label_key, value)
            shortcuts.setdefault(label_key[0], value)
    return shortcuts
