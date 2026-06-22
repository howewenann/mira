"""Interactive settings overlay."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Button, Static

from config.settings import (
    EXECUTE_TOOL,
    INBUILT_DANGEROUS_TOOLS,
    git_protection_enabled,
    set_git_protection,
    set_tool_always_allow,
    set_tool_enabled,
    tool_always_allow,
    tool_enabled,
)

ToggleKind = Literal["git", "enabled", "always_allow"]
ToggleCallback = Callable[[dict[str, Any]], Awaitable[tuple[bool, str]]]
CloseCallback = Callable[[], None]


@dataclass(frozen=True)
class ToggleCell:
    """One togglable settings value."""

    kind: ToggleKind
    name: str
    locked: bool = False


class SettingsPanel(Vertical):
    """Focused settings overlay for HITL and tool controls."""

    can_focus = True

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        tool_metadata: list[dict[str, str]] | None = None,
        apply_change: ToggleCallback,
        close_panel: CloseCallback | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="settings-overlay", **kwargs)
        self.settings = settings
        self.tool_metadata = tool_metadata or []
        self.apply_change = apply_change
        self.close_panel = close_panel
        self._button_cells: dict[str, ToggleCell] = {}

    def compose(self) -> ComposeResult:
        """Compose a scrollable settings window."""
        with Vertical(id="settings-window"):
            with Horizontal(classes="settings-top-row"):
                yield Static("Settings", classes="settings-title")
                yield SettingsCloseButton("x", panel=self, id="settings-close", classes="settings-close")
            with VerticalScroll(id="settings-body"):
                yield Static("System", classes="settings-section system")
                yield SettingsHeaderRow("Setting")
                with Horizontal(classes="settings-row"):
                    yield Static("Git Protection", classes="settings-label")
                    yield self._toggle_button(ToggleCell("git", "git_protection"), git_protection_enabled(self.settings))
                    yield Static("-", classes="settings-placeholder")

                yield Static("Inbuilt Tools", classes="settings-section inbuilt")
                yield SettingsHeaderRow("Tool")
                for tool_name in INBUILT_DANGEROUS_TOOLS:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(
                            ToggleCell("enabled", tool_name, locked=tool_name != EXECUTE_TOOL),
                            enabled,
                        )
                        yield self._toggle_button(
                            ToggleCell("always_allow", tool_name),
                            tool_always_allow(self.settings, tool_name),
                        )

                yield Static("Custom Tools", classes="settings-section custom")
                yield SettingsHeaderRow("Tool")
                custom_names = custom_tool_names(self.tool_metadata)
                if not custom_names:
                    yield Static("No custom tools loaded", classes="settings-empty")
                for tool_name in custom_names:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(ToggleCell("enabled", tool_name), enabled)
                        yield self._toggle_button(
                            ToggleCell("always_allow", tool_name),
                            tool_always_allow(self.settings, tool_name) if enabled else False,
                        )

            yield Static("", id="settings-status", classes="settings-status")

    def on_mount(self) -> None:
        """Focus the first editable toggle when the panel appears."""
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

    @on(Button.Pressed, ".settings-toggle")
    async def press_toggle(self, event: Button.Pressed) -> None:
        """Toggle the clicked or keyboard-activated setting."""
        event.stop()
        button = event.button
        cell = self._button_cells.get(button.id or "")
        if cell is None or self._cell_locked(cell):
            return
        await self._set_cell(cell, not selected_value(self.settings, cell))

    @on(Button.Pressed, "#settings-close")
    def press_close(self, event: Button.Pressed) -> None:
        """Close the settings panel from the visible close button."""
        event.stop()
        self._close()

    async def _set_focused(self, value: bool) -> None:
        button = self._focused_toggle()
        if button is None:
            return
        cell = self._button_cells.get(button.id or "")
        if cell is None or self._cell_locked(cell):
            return
        await self._set_cell(cell, value)

    async def _set_cell(self, cell: ToggleCell, value: bool) -> None:
        if cell.kind == "git":
            updated = set_git_protection(self.settings, value)
        elif cell.kind == "enabled":
            updated = set_tool_enabled(self.settings, cell.name, value)
        else:
            updated = set_tool_always_allow(self.settings, cell.name, value)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if not ok:
            return
        self.settings = updated
        self._refresh_buttons()

    def _toggle_button(self, cell: ToggleCell, value: bool) -> Button:
        button_id = button_id_for(cell)
        self._button_cells[button_id] = cell
        locked = self._cell_locked(cell)
        label = button_label(cell, value, locked=locked, enabled=tool_enabled(self.settings, cell.name))
        button = SettingsToggleButton(
            label,
            panel=self,
            id=button_id,
            classes=toggle_classes(value, locked=locked),
        )
        if locked:
            button.disabled = True
        return button

    def _refresh_buttons(self) -> None:
        for button_id, cell in self._button_cells.items():
            button = self.query_one(f"#{button_id}", Button)
            locked = cell.locked
            if cell.kind == "always_allow":
                locked = not tool_enabled(self.settings, cell.name)
            value = selected_value(self.settings, cell)
            button.label = button_label(cell, value, locked=locked, enabled=tool_enabled(self.settings, cell.name))
            button.disabled = locked
            button.set_classes(toggle_classes(value, locked=locked))

    def _cell_locked(self, cell: ToggleCell) -> bool:
        if cell.kind == "always_allow":
            return not tool_enabled(self.settings, cell.name)
        return cell.locked

    def _set_status(self, message: str) -> None:
        self.query_one("#settings-status", Static).update(message)

    def _focused_toggle(self) -> Button | None:
        for button in self.query(Button):
            if button.has_focus and "settings-toggle" in button.classes:
                return button
        return None

    def _focus_first_toggle(self) -> None:
        buttons = [
            button
            for button in self.query(Button)
            if "settings-toggle" in button.classes and not button.disabled
        ]
        if buttons:
            buttons[0].focus()

    def _close(self) -> None:
        self.remove()
        if self.close_panel is not None:
            self.close_panel()


def custom_tool_names(metadata: list[dict[str, str]]) -> list[str]:
    """Return project tool names shown in the custom tools section."""
    return sorted({item["name"] for item in metadata if item.get("source") == "project" and item.get("name")})


def selected_value(settings: dict[str, Any], cell: ToggleCell) -> bool:
    """Return the boolean value for a settings cell."""
    if cell.kind == "git":
        return git_protection_enabled(settings)
    if cell.kind == "enabled":
        return tool_enabled(settings, cell.name)
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a settings button."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", cell.name)
    return f"settings-toggle-{cell.kind}-{safe_name}"


def button_label(cell: ToggleCell, value: bool, *, locked: bool = False, enabled: bool = True) -> str:
    """Return display text for a settings toggle."""
    if locked and cell.kind == "always_allow" and cell.name != EXECUTE_TOOL and not enabled:
        return "-"
    return "yes" if value else "no"


def toggle_classes(value: bool, *, locked: bool = False) -> str:
    """Return CSS classes for a toggle button."""
    state = "on" if value else "off"
    locked_class = " locked" if locked else ""
    return f"settings-mode settings-toggle {state}{locked_class}"


class SettingsToggleButton(Button):
    """Toggle button that forwards settings shortcuts to its panel."""

    def __init__(self, *args: Any, panel: SettingsPanel, **kwargs: Any) -> None:
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
            if cell is not None and not self.panel._cell_locked(cell):
                await self.panel._set_cell(cell, key == "y")


class SettingsHeaderRow(Horizontal):
    """Column header row for settings tables."""

    def __init__(self, name_label: str) -> None:
        super().__init__(classes="settings-row settings-header-row")
        self.name_label = name_label

    def compose(self) -> ComposeResult:
        yield Static(self.name_label, classes="settings-column-label name")
        yield Static("enabled", classes="settings-column-label enabled")
        yield Static("always allow", classes="settings-column-label always")


class SettingsCloseButton(Button):
    """Close button that keeps settings keyboard shortcuts local."""

    def __init__(self, *args: Any, panel: SettingsPanel, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel

    async def on_key(self, event: Key) -> None:
        """Support q/Esc while focus is on the close button."""
        key = event.key.lower()
        if key in {"escape", "q"}:
            event.stop()
            self.panel._close()
