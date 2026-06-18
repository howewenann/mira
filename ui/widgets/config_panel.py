"""Interactive HITL configuration panel."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Static

from config.settings import (
    DEFAULT_APPROVAL_TOOLS,
    git_protection_enabled,
    set_git_protection,
    set_tool_always_allow,
    tool_always_allow,
)

ToggleKind = Literal["git", "tool"]
ToggleCallback = Callable[[dict[str, Any]], Awaitable[tuple[bool, str]]]
CloseCallback = Callable[[], None]


@dataclass(frozen=True)
class ToggleCell:
    """One togglable config value."""

    kind: ToggleKind
    name: str


class ConfigPanel(Vertical):
    """Flat button-driven config menu."""

    can_focus = True

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        apply_change: ToggleCallback,
        close_panel: CloseCallback | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="config-panel", **kwargs)
        self.settings = settings
        self.apply_change = apply_change
        self.close_panel = close_panel
        self._button_cells: dict[str, ToggleCell] = {}

    def compose(self) -> ComposeResult:
        """Compose a compact grouped settings panel."""
        with Horizontal(classes="config-top-row"):
            yield Static("System", classes="config-section system")
            yield ConfigCloseButton("x", panel=self, id="config-close", classes="config-close")
        with Horizontal(classes="config-row"):
            yield Static("Git Protection", classes="config-label")
            yield self._toggle_button(ToggleCell("git", "git_protection"), git_protection_enabled(self.settings))

        yield Static("Tools", classes="config-section tools")
        for tool_name in config_tool_names(self.settings):
            with Horizontal(classes="config-row"):
                yield Static(tool_name, classes="config-label")
                yield self._toggle_button(ToggleCell("tool", tool_name), tool_always_allow(self.settings, tool_name))
        yield Static("", id="config-status", classes="config-status")

    def on_mount(self) -> None:
        """Focus the first toggle when the panel appears."""
        self.call_after_refresh(self._focus_first_toggle)

    async def on_key(self, event: Key) -> None:
        """Handle direct yes/no and close shortcuts."""
        key = event.key.lower()
        if key in {"escape", "q"}:
            event.stop()
            self._close()
            return
        if key in {"y", "n"}:
            event.stop()
            await self._set_focused(key == "y")

    @on(Button.Pressed, ".config-toggle")
    async def press_toggle(self, event: Button.Pressed) -> None:
        """Toggle the clicked or keyboard-activated setting."""
        event.stop()
        button = event.button
        cell = self._button_cells.get(button.id or "")
        if cell is None:
            return
        await self._set_cell(cell, not selected_value(self.settings, cell))

    @on(Button.Pressed, "#config-close")
    def press_close(self, event: Button.Pressed) -> None:
        """Close the config panel from the visible close button."""
        event.stop()
        self._close()

    async def _set_focused(self, value: bool) -> None:
        button = self._focused_toggle()
        if button is None:
            return
        cell = self._button_cells.get(button.id or "")
        if cell is None:
            return
        await self._set_cell(cell, value)

    async def _set_cell(self, cell: ToggleCell, value: bool) -> None:
        updated = set_git_protection(self.settings, value) if cell.kind == "git" else set_tool_always_allow(
            self.settings,
            cell.name,
            value,
        )
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if not ok:
            return
        self.settings = updated
        self._refresh_buttons()

    def _toggle_button(self, cell: ToggleCell, value: bool) -> Button:
        button_id = button_id_for(cell)
        self._button_cells[button_id] = cell
        return ConfigToggleButton(
            button_label(cell, value),
            panel=self,
            id=button_id,
            classes=toggle_classes(value),
        )

    def _refresh_buttons(self) -> None:
        for button_id, cell in self._button_cells.items():
            button = self.query_one(f"#{button_id}", Button)
            value = selected_value(self.settings, cell)
            button.label = button_label(cell, value)
            button.set_classes(toggle_classes(value))

    def _set_status(self, message: str) -> None:
        self.query_one("#config-status", Static).update(message)

    def _focused_toggle(self) -> Button | None:
        for button in self.query(Button):
            if button.has_focus and "config-toggle" in button.classes:
                return button
        return None

    def _focus_first_toggle(self) -> None:
        buttons = [button for button in self.query(Button) if "config-toggle" in button.classes]
        if buttons:
            buttons[0].focus()

    def _close(self) -> None:
        self.remove()
        if self.close_panel is not None:
            self.close_panel()


def config_tool_names(settings: dict[str, Any]) -> list[str]:
    """Return tool names shown in the HITL settings menu."""
    names = list(DEFAULT_APPROVAL_TOOLS)
    hitl = settings.get("hitl") if isinstance(settings, dict) else None
    tools = hitl.get("tools") if isinstance(hitl, dict) else None
    if isinstance(tools, dict):
        names.extend(name for name in tools if isinstance(name, str))
    return sorted(set(names))


def selected_value(settings: dict[str, Any], cell: ToggleCell) -> bool:
    """Return the boolean value for a togglable cell."""
    if cell.kind == "git":
        return git_protection_enabled(settings)
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a config button."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", cell.name)
    return f"config-toggle-{cell.kind}-{safe_name}"


def button_label(cell: ToggleCell, value: bool) -> str:
    """Return display text for a config toggle."""
    if cell.kind == "git":
        return "enabled" if value else "disabled"
    return "allow" if value else "ask"


def toggle_classes(value: bool) -> str:
    """Return CSS classes for a toggle button."""
    state = "on" if value else "off"
    return f"config-mode config-toggle {state}"


class ConfigToggleButton(Button):
    """Toggle button that forwards config shortcuts to its panel."""

    def __init__(self, *args: Any, panel: ConfigPanel, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel

    async def on_key(self, event: Key) -> None:
        """Support q/Esc/y/n while focus is on a button."""
        key = event.key.lower()
        if key in {"escape", "q"}:
            event.stop()
            self.panel._close()
            return
        if key in {"y", "n"}:
            event.stop()
            cell = self.panel._button_cells.get(self.id or "")
            if cell is not None:
                await self.panel._set_cell(cell, key == "y")


class ConfigCloseButton(Button):
    """Close button that keeps config keyboard shortcuts local."""

    def __init__(self, *args: Any, panel: ConfigPanel, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel

    async def on_key(self, event: Key) -> None:
        """Support q/Esc while focus is on the close button."""
        key = event.key.lower()
        if key in {"escape", "q"}:
            event.stop()
            self.panel._close()
