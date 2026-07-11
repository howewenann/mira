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
from textual.widgets import Button, Input, Static

from config.settings import (
    DYNAMIC_SUBAGENTS,
    DYNAMIC_SUBAGENT_RESPONSE_SCHEMA,
    EXECUTE_TOOL,
    EXECUTE_ENV_MODES,
    INBUILT_DANGEROUS_TOOLS,
    dynamic_subagent_response_schema_enabled,
    dynamic_subagents_enabled,
    execute_env_settings,
    git_protection_enabled,
    set_dynamic_subagent_response_schema,
    set_dynamic_subagents,
    set_execute_env_allow,
    set_execute_env_mode,
    set_execute_env_value,
    set_git_protection,
    set_tool_always_allow,
    set_tool_enabled,
    tool_always_allow,
    tool_enabled,
)

ToggleKind = Literal["git", "system", "response_schema", "enabled", "always_allow"]
EXECUTE_ENV_LABELS = {
    "system": "system shell",
    "conda_name": "conda env name",
    "conda_prefix": "conda env path",
    "venv": "venv path",
}
EXECUTE_ENV_FIELDS = {
    "conda_name": ("name", "Conda env name", "my_project_env"),
    "conda_prefix": ("prefix", "Conda env path", r"C:\Users\me\.conda\envs\my_project_env"),
    "venv": ("path", "Venv location", r".venv or .venv\Scripts\python.exe"),
}
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
                yield Static("System Settings", classes="settings-section system")
                yield SettingsHeaderRow("", show_always=False)
                with Horizontal(classes="settings-row"):
                    yield Static("Git Protection", classes="settings-label")
                    yield self._toggle_button(ToggleCell("git", "git_protection"), git_protection_enabled(self.settings))
                with Horizontal(classes="settings-row"):
                    yield Static("Dynamic subagents", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("system", DYNAMIC_SUBAGENTS),
                        dynamic_subagents_enabled(self.settings),
                    )
                with Horizontal(classes="settings-row settings-child-row"):
                    yield Static("Response schemas", classes="settings-label settings-child-label")
                    yield self._toggle_button(
                        ToggleCell("response_schema", DYNAMIC_SUBAGENT_RESPONSE_SCHEMA),
                        dynamic_subagent_response_schema_enabled(self.settings),
                    )

                yield Static("Inbuilt Tools", classes="settings-section inbuilt")
                yield SettingsHeaderRow("")
                for tool_name in INBUILT_DANGEROUS_TOOLS:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(
                            ToggleCell("enabled", tool_name),
                            enabled,
                        )
                        yield self._toggle_button(
                            ToggleCell("always_allow", tool_name),
                            tool_always_allow(self.settings, tool_name),
                        )

                yield Static("Execute Environment", classes="settings-section execute-env")
                execute_env = execute_env_settings(self.settings)
                execute_env_mode = str(execute_env.get("mode") or "system")
                with Horizontal(classes="settings-row settings-wide-row"):
                    yield Static("Run commands in", classes="settings-label")
                    yield Button(
                        execute_env_mode_label(execute_env_mode),
                        id="settings-execute-env-mode",
                        classes="settings-value-button",
                    )
                yield Static("Press Enter/click to change", classes="settings-help")

                for mode, (key, label, placeholder) in EXECUTE_ENV_FIELDS.items():
                    with Horizontal(id=f"settings-execute-env-{key}-row", classes="settings-row settings-wide-row") as row:
                        row.display = execute_env_mode == mode
                        yield Static(label, classes="settings-label")
                        yield Input(
                            value=str(execute_env.get(key) or ""),
                            placeholder=f"<{placeholder}>",
                            id=f"settings-execute-env-{key}",
                            classes="settings-input",
                        )
                preview = Static(
                    execute_env_preview(execute_env),
                    id="settings-execute-env-preview",
                    classes="settings-help",
                )
                preview.display = execute_env_mode in EXECUTE_ENV_FIELDS
                yield preview

                with Horizontal(classes="settings-row settings-wide-row"):
                    yield Static("Additional env var names", classes="settings-label")
                    yield Input(
                        value=", ".join(execute_env.get("allow") or []),
                        placeholder="<CUDA_HOME, HF_HOME, REQUESTS_CA_BUNDLE>",
                        id="settings-execute-env-allow",
                        classes="settings-input",
                    )
                yield Static("Examples only. Use comma-separated names.", classes="settings-help")

                yield Static("Custom Tools", classes="settings-section custom")
                yield SettingsHeaderRow("")
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

    @on(Button.Pressed, "#settings-execute-env-mode")
    async def press_execute_env_mode(self, event: Button.Pressed) -> None:
        """Cycle the execute environment mode."""
        event.stop()
        current = execute_env_settings(self.settings).get("mode", "system")
        modes = list(EXECUTE_ENV_MODES)
        next_mode = modes[(modes.index(current) + 1) % len(modes)] if current in modes else "system"
        updated = set_execute_env_mode(self.settings, next_mode)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            self._refresh_execute_env_section("settings-execute-env-mode")

    @on(Input.Submitted, ".settings-input")
    async def submit_execute_env_input(self, event: Input.Submitted) -> None:
        """Save execute environment text fields on Enter."""
        event.stop()
        input_id = event.input.id or ""
        value = event.value
        if input_id == "settings-execute-env-allow":
            updated = set_execute_env_allow(self.settings, value)
        elif input_id.startswith("settings-execute-env-"):
            updated = set_execute_env_value(self.settings, input_id.removeprefix("settings-execute-env-"), value)
        else:
            return
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            self._refresh_execute_env_section(input_id)

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
        elif cell.kind == "system":
            updated = set_dynamic_subagents(self.settings, value)
        elif cell.kind == "response_schema":
            updated = set_dynamic_subagent_response_schema(self.settings, value)
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
            locked = self._cell_locked(cell)
            value = selected_value(self.settings, cell)
            button.label = button_label(cell, value, locked=locked, enabled=tool_enabled(self.settings, cell.name))
            button.disabled = locked
            button.set_classes(toggle_classes(value, locked=locked))

    def _cell_locked(self, cell: ToggleCell) -> bool:
        if cell.kind == "always_allow":
            return not tool_enabled(self.settings, cell.name)
        if cell.kind == "response_schema":
            return not dynamic_subagents_enabled(self.settings)
        return cell.locked

    def _set_status(self, message: str) -> None:
        self.query_one("#settings-status", Static).update(message)

    def _refresh_execute_env_section(self, focus_id: str | None = None) -> None:
        execute_env = execute_env_settings(self.settings)
        mode = str(execute_env.get("mode") or "system")
        self.query_one("#settings-execute-env-mode", Button).label = execute_env_mode_label(mode)
        for field_mode, (key, _, _) in EXECUTE_ENV_FIELDS.items():
            self.query_one(f"#settings-execute-env-{key}-row", Horizontal).display = mode == field_mode
            self.query_one(f"#settings-execute-env-{key}", Input).value = str(execute_env.get(key) or "")
        preview = self.query_one("#settings-execute-env-preview", Static)
        preview.update(execute_env_preview(execute_env))
        preview.display = mode in EXECUTE_ENV_FIELDS
        self.query_one("#settings-execute-env-allow", Input).value = ", ".join(execute_env.get("allow") or [])
        if focus_id is not None:
            try:
                self.query_one(f"#{focus_id}", Button).focus(scroll_visible=False)
            except Exception:
                pass

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


def execute_env_mode_label(mode: str) -> str:
    """Return a discoverable label for the execute environment mode control."""
    return f"{EXECUTE_ENV_LABELS.get(mode, 'system shell')} >"


def execute_env_preview(settings: dict[str, Any]) -> str:
    """Return preview or validation text for the current execute environment mode."""
    mode = str(settings.get("mode") or "system")
    if mode == "conda_name" and settings.get("name"):
        return f"Preview: conda run -n {settings['name']} <command>"
    if mode == "conda_name":
        return "Conda env name required before commands are wrapped."
    if mode == "conda_prefix" and settings.get("prefix"):
        return f"Preview: conda run -p {settings['prefix']} <command>"
    if mode == "conda_prefix":
        return "Conda env path required before commands are wrapped."
    if mode == "venv" and settings.get("path"):
        return f"Preview: {settings['path']}"
    if mode == "venv":
        return "Venv location required before PATH is adjusted."
    return "Placeholder examples are not saved or applied."


def selected_value(settings: dict[str, Any], cell: ToggleCell) -> bool:
    """Return the boolean value for a settings cell."""
    if cell.kind == "git":
        return git_protection_enabled(settings)
    if cell.kind == "system":
        return dynamic_subagents_enabled(settings)
    if cell.kind == "response_schema":
        return dynamic_subagent_response_schema_enabled(settings)
    if cell.kind == "enabled":
        return tool_enabled(settings, cell.name)
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a settings button."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", cell.name)
    return f"settings-toggle-{cell.kind}-{safe_name}"


def button_label(cell: ToggleCell, value: bool, *, locked: bool = False, enabled: bool = True) -> str:
    """Return display text for a settings toggle."""
    if locked and cell.kind == "always_allow" and not enabled:
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

    def __init__(self, name_label: str, *, show_always: bool = True) -> None:
        super().__init__(classes="settings-row settings-header-row")
        self.name_label = name_label
        self.show_always = show_always

    def compose(self) -> ComposeResult:
        yield Static(self.name_label, classes="settings-column-label name")
        yield Static("enable", classes="settings-column-label enabled")
        if self.show_always:
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
