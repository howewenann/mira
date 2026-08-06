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
from textual.widgets import Button, Input, Select, Static

from config.settings import (
    DYNAMIC_SUBAGENTS,
    DYNAMIC_SUBAGENT_RESPONSE_SCHEMA,
    EXECUTE_TOOL,
    EXECUTE_ENV_MODES,
    INBUILT_DANGEROUS_TOOLS,
    PLANNING_RESPONSE_STATUS_MAX_RETRIES_LIMIT,
    PLANNING_TODOS,
    RUBRIC,
    RUBRIC_MAX_ITERATIONS_LIMIT,
    dynamic_subagent_response_schema_enabled,
    dynamic_subagents_enabled,
    execute_env_settings,
    git_protection_enabled,
    planning_todos_enabled,
    planning_response_status_max_retries,
    rubric_enabled,
    rubric_max_iterations,
    set_dynamic_subagent_response_schema,
    set_dynamic_subagents,
    set_execute_env_allow,
    set_execute_env_mode,
    set_execute_env_value,
    set_git_protection,
    set_planning_todos,
    set_planning_response_status_max_retries,
    set_rubric_enabled,
    set_rubric_max_iterations,
    set_tool_always_allow,
    set_tool_enabled,
    set_tool_plan_access,
    tool_always_allow,
    tool_enabled,
    tool_plan_access,
    mcp_server_always_allow,
    mcp_server_enabled,
    mcp_tool_policy,
    set_mcp_tool_policy_value,
)

ToggleKind = Literal[
    "git", "system", "response_schema", "todos", "rubric", "enabled", "always_allow", "plan_access",
    "mcp_server_enabled", "mcp_server_allow", "mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan",
]
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
    server: str = ""


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
        mcp_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="settings-overlay", **kwargs)
        self.settings = settings
        self.tool_metadata = tool_metadata or []
        self.apply_change = apply_change
        self.close_panel = close_panel
        self.mcp_manager = mcp_manager
        self._button_cells: dict[str, ToggleCell] = {}

    def compose(self) -> ComposeResult:
        """Compose a scrollable settings window."""
        with Vertical(id="settings-window"):
            with Horizontal(classes="settings-top-row"):
                yield Static("Settings", classes="settings-title")
                yield SettingsCloseButton("x", panel=self, id="settings-close", classes="panel-close")
            with Horizontal(id="settings-tabs"):
                yield Button("General", id="settings-tab-general", classes="settings-tab active")
                yield Button("Custom Tools", id="settings-tab-custom", classes="settings-tab")
                yield Button("MCP", id="settings-tab-mcp", classes="settings-tab")
            with VerticalScroll(id="settings-body", classes="settings-body settings-general-body"):
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
                with Horizontal(classes="settings-row"):
                    yield Static("Planning todos", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("todos", PLANNING_TODOS),
                        planning_todos_enabled(self.settings),
                    )
                with Horizontal(classes="settings-row settings-child-row planning-response-status-retries-row"):
                    yield Static("Response-status retries", classes="settings-label settings-child-label")
                    yield Input(
                        value=str(planning_response_status_max_retries(self.settings)),
                        id="settings-planning-response-status-max-retries",
                        classes="settings-input planning-response-status-retries-input",
                    )
                with Horizontal(classes="settings-row"):
                    yield Static("Rubric Middleware", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("rubric", RUBRIC),
                        rubric_enabled(self.settings),
                    )
                with Horizontal(classes="settings-row settings-child-row rubric-iterations-row"):
                    yield Static("Maximum iterations", classes="settings-label settings-child-label")
                    yield Input(
                        value=str(rubric_max_iterations(self.settings)),
                        id="settings-rubric-max-iterations",
                        classes="settings-input rubric-iterations-input",
                        disabled=not rubric_enabled(self.settings),
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
                    yield Select(
                        ((label, mode) for mode, label in EXECUTE_ENV_LABELS.items()),
                        value=execute_env_mode,
                        allow_blank=False,
                        id="settings-execute-env-mode",
                        classes="settings-select",
                    )

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

                with Horizontal(classes="settings-row settings-wide-row"):
                    yield Static("Additional env var names", classes="settings-label")
                    yield Input(
                        value=", ".join(execute_env.get("allow") or []),
                        placeholder="<CUDA_HOME, HF_HOME, REQUESTS_CA_BUNDLE>",
                        id="settings-execute-env-allow",
                        classes="settings-input",
                    )

            with VerticalScroll(id="settings-custom-body", classes="settings-body"):
                yield Static("Custom Tools", classes="settings-section custom")
                yield SettingsHeaderRow("", show_plan=True)
                custom_names = custom_tool_names(self.tool_metadata)
                if not custom_names:
                    yield Static("No custom tools loaded", classes="settings-empty")
                for tool_name in custom_names:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(ToggleCell("enabled", tool_name), enabled)
                        yield self._toggle_button(ToggleCell("always_allow", tool_name), tool_always_allow(self.settings, tool_name))
                        yield self._toggle_button(ToggleCell("plan_access", tool_name), tool_plan_access(self.settings, tool_name))

            with VerticalScroll(id="settings-mcp-body", classes="settings-body"):
                yield Static("MCP", classes="settings-section mcp")
                yield SettingsHeaderRow("", show_plan=True)
                states = list(getattr(self.mcp_manager, "servers", {}).values())
                if not states:
                    yield Static("No MCP servers configured", classes="settings-empty")
                for state in states:
                    enabled = mcp_server_enabled(self.settings, state.name)
                    with Horizontal(classes="settings-row settings-mcp-server-row"):
                        yield Static(f"{state.name} MCP · {state.transport}", classes="settings-label settings-mcp-server-label")
                        yield self._toggle_button(ToggleCell("mcp_server_enabled", state.name, server=state.name), enabled)
                        yield self._toggle_button(
                            ToggleCell("mcp_server_allow", state.name, server=state.name),
                            mcp_server_always_allow(self.settings, state.name),
                        )
                        yield Static("-", classes="settings-mode settings-mcp-dash")
                    if not state.tool_metadata:
                        yield Static("  No tools", classes="settings-empty settings-mcp-tool-empty")
                    for item in state.tool_metadata:
                        original = item.get("original_name", "")
                        policy = mcp_tool_policy(self.settings, state.name, original)
                        with Horizontal(classes="settings-row settings-mcp-tool-row"):
                            yield Static(original, classes="settings-label")
                            yield self._toggle_button(ToggleCell("mcp_tool_enabled", original, not enabled, state.name), policy.enabled)
                            yield self._toggle_button(ToggleCell("mcp_tool_allow", original, not enabled or not policy.enabled, state.name), policy.always_allow)
                            yield self._toggle_button(ToggleCell("mcp_tool_plan", original, not enabled or not policy.enabled, state.name), policy.plan_access)
                    yield Static("-" * 64, classes="settings-mcp-divider")

            yield Static("", id="settings-status", classes="settings-status")

    def on_mount(self) -> None:
        """Focus the first editable toggle when the panel appears."""
        self.query_one("#settings-custom-body").display = False
        self.query_one("#settings-mcp-body").display = False
        self.call_after_refresh(self._focus_first_toggle)

    @on(Button.Pressed, ".settings-tab")
    def press_tab(self, event: Button.Pressed) -> None:
        event.stop()
        selected = (event.button.id or "").removeprefix("settings-tab-")
        for name in ("general", "custom", "mcp"):
            selector = "#settings-body" if name == "general" else f"#settings-{name}-body"
            self.query_one(selector).display = name == selected
            self.query_one(f"#settings-tab-{name}", Button).set_class(name == selected, "active")
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

    @on(Select.Changed, "#settings-execute-env-mode")
    async def change_execute_env_mode(self, event: Select.Changed) -> None:
        """Save the selected execute environment mode."""
        event.stop()
        current = str(execute_env_settings(self.settings).get("mode") or "system")
        if event.value not in EXECUTE_ENV_MODES or event.value == current:
            return
        updated = set_execute_env_mode(self.settings, event.value)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            self._refresh_execute_env_section("settings-execute-env-mode")
        else:
            event.select.value = current

    @on(Input.Submitted, ".settings-input")
    async def submit_execute_env_input(self, event: Input.Submitted) -> None:
        """Save execute environment text fields on Enter."""
        event.stop()
        input_id = event.input.id or ""
        if input_id == "settings-planning-response-status-max-retries":
            await self._submit_planning_response_status_retries(event.input)
            return
        if input_id == "settings-rubric-max-iterations":
            await self._submit_rubric_iterations(event.input)
            return
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
        elif cell.kind == "todos":
            updated = set_planning_todos(self.settings, value)
        elif cell.kind == "rubric":
            updated = set_rubric_enabled(self.settings, value)
        elif cell.kind == "enabled":
            updated = set_tool_enabled(self.settings, cell.name, value)
        elif cell.kind == "plan_access":
            updated = set_tool_plan_access(self.settings, cell.name, value)
        elif cell.kind == "mcp_server_enabled":
            if self.mcp_manager is None or not await self.mcp_manager.set_server_enabled(cell.server, value):
                self._set_status("could not update MCP server")
                return
            from config.settings import load_settings

            self.settings = load_settings(self.mcp_manager.workspace)
            self._set_status("MCP server updated")
            self._refresh_buttons()
            return
        elif cell.kind == "mcp_server_allow":
            if self.mcp_manager is None or not await self.mcp_manager.set_server_always_allow(cell.server, value):
                self._set_status("could not update MCP server approval")
                return
            from config.settings import load_settings

            self.settings = load_settings(self.mcp_manager.workspace)
            self._set_status("MCP server approval updated")
            self._refresh_buttons()
            return
        elif cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan"}:
            key = {"mcp_tool_enabled": "enabled", "mcp_tool_allow": "always_allow", "mcp_tool_plan": "plan_access"}[cell.kind]
            updated = set_mcp_tool_policy_value(self.settings, cell.server, cell.name, key, value)
        else:
            updated = set_tool_always_allow(self.settings, cell.name, value)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if not ok:
            return
        self.settings = updated
        self._refresh_buttons()
        self._refresh_rubric_control()

    async def _submit_rubric_iterations(self, input_widget: Input) -> None:
        """Persist a valid rubric cap and restore the last value after invalid input."""
        try:
            value = int(input_widget.value.strip())
        except ValueError:
            value = 0
        updated = set_rubric_max_iterations(self.settings, value)
        if rubric_max_iterations(updated) != value:
            input_widget.value = str(rubric_max_iterations(self.settings))
            self._set_status(f"maximum iterations must be from 1 to {RUBRIC_MAX_ITERATIONS_LIMIT}")
            return
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
        input_widget.value = str(rubric_max_iterations(self.settings))

    async def _submit_planning_response_status_retries(self, input_widget: Input) -> None:
        """Persist a valid Plan/Goal response-status retry cap."""
        try:
            value = int(input_widget.value.strip())
        except ValueError:
            value = 0
        updated = set_planning_response_status_max_retries(self.settings, value)
        if planning_response_status_max_retries(updated) != value:
            input_widget.value = str(planning_response_status_max_retries(self.settings))
            self._set_status(
                "response-status retries must be from 1 to "
                f"{PLANNING_RESPONSE_STATUS_MAX_RETRIES_LIMIT}"
            )
            return
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
        input_widget.value = str(planning_response_status_max_retries(self.settings))

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
        if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan"}:
            if not mcp_server_enabled(self.settings, cell.server):
                return True
            if cell.kind != "mcp_tool_enabled" and not mcp_tool_policy(self.settings, cell.server, cell.name).enabled:
                return True
            return False
        if cell.kind in {"always_allow", "plan_access"}:
            return not tool_enabled(self.settings, cell.name)
        if cell.kind == "response_schema":
            return not dynamic_subagents_enabled(self.settings)
        return cell.locked

    def _set_status(self, message: str) -> None:
        self.query_one("#settings-status", Static).update(message)

    def _refresh_execute_env_section(self, focus_id: str | None = None) -> None:
        execute_env = execute_env_settings(self.settings)
        mode = str(execute_env.get("mode") or "system")
        select = self.query_one("#settings-execute-env-mode", Select)
        if select.value != mode:
            select.value = mode
        for field_mode, (key, _, _) in EXECUTE_ENV_FIELDS.items():
            self.query_one(f"#settings-execute-env-{key}-row", Horizontal).display = mode == field_mode
            self.query_one(f"#settings-execute-env-{key}", Input).value = str(execute_env.get(key) or "")
        self.query_one("#settings-execute-env-allow", Input).value = ", ".join(execute_env.get("allow") or [])
        if focus_id is not None:
            try:
                self.query_one(f"#{focus_id}").focus(scroll_visible=False)
            except Exception:
                pass

    def _refresh_rubric_control(self) -> None:
        """Refresh dependent rubric controls after the parent toggle changes."""
        control = self.query_one("#settings-rubric-max-iterations", Input)
        control.disabled = not rubric_enabled(self.settings)
        control.value = str(rubric_max_iterations(self.settings))

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
    if cell.kind == "system":
        return dynamic_subagents_enabled(settings)
    if cell.kind == "response_schema":
        return dynamic_subagent_response_schema_enabled(settings)
    if cell.kind == "todos":
        return planning_todos_enabled(settings)
    if cell.kind == "rubric":
        return rubric_enabled(settings)
    if cell.kind == "enabled":
        return tool_enabled(settings, cell.name)
    if cell.kind == "plan_access":
        return tool_plan_access(settings, cell.name)
    if cell.kind == "mcp_server_enabled":
        return mcp_server_enabled(settings, cell.server)
    if cell.kind == "mcp_server_allow":
        return mcp_server_always_allow(settings, cell.server)
    if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan"}:
        policy = mcp_tool_policy(settings, cell.server, cell.name)
        return {
            "mcp_tool_enabled": policy.enabled,
            "mcp_tool_allow": policy.always_allow,
            "mcp_tool_plan": policy.plan_access,
        }[cell.kind]
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a settings button."""
    identity = f"{cell.server}-{cell.name}" if cell.server else cell.name
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", identity)
    return f"settings-toggle-{cell.kind}-{safe_name}"


def button_label(cell: ToggleCell, value: bool, *, locked: bool = False, enabled: bool = True) -> str:
    """Return display text for a settings toggle."""
    if locked and cell.kind == "always_allow":
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

    def __init__(self, name_label: str, *, show_always: bool = True, show_plan: bool = False) -> None:
        super().__init__(classes="settings-row settings-header-row")
        self.name_label = name_label
        self.show_always = show_always
        self.show_plan = show_plan

    def compose(self) -> ComposeResult:
        yield Static(self.name_label, classes="settings-column-label name")
        yield Static("enable", classes="settings-column-label enabled")
        if self.show_always:
            yield Static("always allow", classes="settings-column-label always")
        if self.show_plan:
            yield Static("plan access", classes="settings-column-label plan")


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
