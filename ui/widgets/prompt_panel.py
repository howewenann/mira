"""In-window prompt panel for interactive user decisions."""

from __future__ import annotations

import asyncio
from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Button, Input, Static, TextArea


class PromptPanel(Vertical):
    """Focused prompt surface mounted in the main app layout."""

    MIN_BUTTON_WIDTH = 12
    MAX_BUTTON_WIDTH = 32
    BUTTON_GAP = 1

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="prompt-panel", **kwargs)
        self.can_focus = True
        self.display = False
        self._future: asyncio.Future[str | None] | None = None
        self._mode = ""
        self._shortcuts: dict[str, str] = {}
        self._button_values: dict[str, str] = {}
        self._button_rows: list[list[str]] = []
        self._button_positions: dict[str, tuple[int, int]] = {}
        self._button_specs: list[tuple[str, str]] = []
        self._last_button_width = 0
        self._reflow_running = False
        self._reflow_generation = 0

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
        yield Vertical(id="prompt-panel-buttons")

    def on_mount(self) -> None:
        """Hide controls that only appear for some prompt modes."""
        self.query_one("#prompt-panel-input", Input).display = False
        self.query_one("#prompt-panel-editor", TextArea).display = False

    async def choose(self, title: str, message: str, choices: list[tuple[str, str]]) -> str | None:
        """Show a choice prompt and return the selected value."""
        await self._open("choice", title, message)
        self.focus()
        self._shortcuts = choice_shortcuts(choices)
        self._button_specs = []
        for index, (value, label) in enumerate(choices):
            button_id = f"prompt-choice-{index}"
            self._button_values[button_id] = value
            self._button_specs.append((button_id, label))
        await self._start_button_reflow(focus_first=True)
        if not self._button_rows:
            self._schedule_button_reflow(focus_first=True)
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
        self._button_rows = []
        self._button_positions = {}
        self._button_specs = []
        self._last_button_width = 0
        self._reflow_running = False
        self._reflow_generation += 1

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
        self._button_specs = []
        self._last_button_width = 0
        self._reflow_running = False
        self._reflow_generation += 1
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
        self._button_specs = list(buttons)
        self._schedule_button_reflow()

    async def _mount_button_rows(self, buttons: list[tuple[str, str]], available_width: int) -> None:
        """Mount prompt buttons in rows that fit the measured panel width."""
        container = self.query_one("#prompt-panel-buttons", Vertical)
        await container.remove_children()
        self._button_rows = []
        self._button_positions = {}
        for row_index, row in enumerate(wrapped_button_rows(buttons, available_width)):
            row_id = f"prompt-panel-button-row-{row_index}"
            row_ids = []
            row_buttons = []
            for column_index, (button_id, label) in enumerate(row):
                button = Button(
                    visible_button_label(label),
                    id=button_id,
                    classes="prompt-panel-button",
                )
                button.styles.width = button_width(label)
                row_buttons.append(button)
                row_ids.append(button_id)
                self._button_positions[button_id] = (row_index, column_index)
            self._button_rows.append(row_ids)
            row_widget = Horizontal(*row_buttons, id=row_id, classes="prompt-panel-button-row")
            await container.mount(row_widget)

    def _available_button_width(self) -> tuple[int, bool]:
        """Return the content width for prompt button rows and whether it was measured."""
        try:
            container = self.query_one("#prompt-panel-buttons", Vertical)
        except NoMatches:
            return 0, False
        width = container.region.width
        if width > 0:
            return max(0, width - 2), True
        width = container.size.width
        if width > 0:
            return max(0, width - 2), True
        if self.parent is not None:
            width = self.parent.region.width or self.parent.size.width
        if width <= 0 and self.app is not None:
            width = self.app.size.width
        return max(0, width - 2), False

    def _schedule_button_reflow(self, *, focus_first: bool = False) -> None:
        """Rebuild choice rows after Textual has measured the prompt width."""
        self.call_after_refresh(self._start_button_reflow, self._reflow_generation, focus_first)

    async def _start_button_reflow(self, generation: int | None = None, focus_first: bool = False) -> None:
        """Run an async button reflow after layout has measured width."""
        if generation is not None and generation != self._reflow_generation:
            return
        if not self.is_mounted or not self._button_specs:
            return
        if self._reflow_running:
            return
        await self._reflow_button_rows(focus_first=focus_first)

    async def _reflow_button_rows(self, *, focus_first: bool = False) -> None:
        """Rebuild button rows from the currently measured width."""
        self._reflow_running = True
        try:
            await asyncio.sleep(0)
            if not self.is_mounted or not self._button_specs:
                return
            available_width = 0
            measured = False
            for _ in range(4):
                available_width, measured = self._available_button_width()
                if available_width > 0:
                    break
                await asyncio.sleep(0)
            if available_width <= 0:
                return
            if available_width == self._last_button_width and self._button_rows:
                if focus_first:
                    self.call_after_refresh(self._focus_first_button)
                return
            self._last_button_width = available_width
            focused_id = next((button.id for button in self.query(Button) if button.has_focus), None)
            await self._mount_button_rows(self._button_specs, available_width)
            await asyncio.sleep(0)
            self._restore_button_focus(focused_id, focus_first)
            if not measured:
                self.call_after_refresh(self._start_button_reflow, self._reflow_generation, False)
        finally:
            self._reflow_running = False

    def _restore_button_focus(self, focused_id: str | None, focus_first: bool) -> None:
        """Restore button focus after a row rebuild."""
        if focused_id:
            try:
                self.query_one(f"#{focused_id}", Button).focus()
                return
            except NoMatches:
                pass
        if focus_first:
            self._focus_first_button()
            if not any(button.has_focus for button in self.query(Button)):
                self.call_after_refresh(self._focus_first_button)

    async def _clear_buttons(self) -> None:
        """Remove all footer buttons from the previous prompt."""
        buttons = self.query_one("#prompt-panel-buttons", Vertical)
        await buttons.remove_children()

    def _focus_first_button(self) -> None:
        """Focus the first footer button if one exists."""
        buttons = self.query(Button)
        if len(buttons) > 0:
            buttons.first().focus()

    def _focus_button_offset(self, offset: int) -> None:
        """Move focus across buttons in document order by offset."""
        buttons = list(self.query(Button))
        if not buttons:
            return

        focused_index = next((index for index, button in enumerate(buttons) if button.has_focus), None)
        if focused_index is None:
            target_index = 0 if offset >= 0 else len(buttons) - 1
        else:
            target_index = (focused_index + offset) % len(buttons)
        buttons[target_index].focus()

    def _focus_button_grid(self, row_delta: int, column_delta: int) -> None:
        """Move focus through wrapped button rows."""
        focused = next((button for button in self.query(Button) if button.has_focus), None)
        if focused is None or focused.id not in self._button_positions:
            self._focus_first_button()
            return

        row_index, column_index = self._button_positions[focused.id]
        if row_delta:
            target_row_index = (row_index + row_delta) % len(self._button_rows)
            target_row = self._button_rows[target_row_index]
            target_column_index = min(column_index, len(target_row) - 1)
        else:
            target_row_index = row_index
            target_row = self._button_rows[target_row_index]
            target_column_index = (column_index + column_delta) % len(target_row)

        target_id = target_row[target_column_index]
        self.query_one(f"#{target_id}", Button).focus()

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
                if event.key == "up":
                    self._focus_button_grid(-1, 0)
                else:
                    self._focus_button_grid(0, -1)
                return

            if event.key in {"right", "down"}:
                event.stop()
                if event.key == "down":
                    self._focus_button_grid(1, 0)
                else:
                    self._focus_button_grid(0, 1)
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

    def on_resize(self, event: events.Resize) -> None:
        """Reflow prompt buttons when the available panel width changes."""
        if self._button_specs:
            self._schedule_button_reflow()


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


def button_width(label: str) -> int:
    """Return a compact fixed width for a prompt button."""
    return max(PromptPanel.MIN_BUTTON_WIDTH, min(PromptPanel.MAX_BUTTON_WIDTH, len(label) + 2))


def visible_button_label(label: str) -> str:
    """Return a display label that fits the prompt button width."""
    width = button_width(label)
    if len(label) <= width - 2:
        return label
    return f"{label[: width - 5].rstrip()}..."


def wrapped_button_rows(buttons: list[tuple[str, str]], available_width: int) -> list[list[tuple[str, str]]]:
    """Group buttons into rows that fit the current prompt width."""
    rows: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    row_width = 0
    for button_id, label in buttons:
        width = button_width(label)
        next_width = width if not row else row_width + PromptPanel.BUTTON_GAP + width
        if row and next_width > available_width:
            rows.append(row)
            row = []
            row_width = 0
            next_width = width

        row.append((button_id, label))
        row_width = next_width

    if row:
        rows.append(row)
    return rows
