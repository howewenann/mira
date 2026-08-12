"""Interactive settings overlay."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from rich.text import Text
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
    PTC_INAPPLICABLE_TOOLS,
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
    set_tool_ptc,
    tool_always_allow,
    tool_enabled,
    tool_plan_access,
    tool_ptc,
    mcp_server_always_allow,
    mcp_server_enabled,
    mcp_tool_policy,
    set_mcp_tool_policy_value,
    MAIN_MODEL,
    RUBRIC_MODEL,
    SUMMARIZATION_MODEL,
    context_limit_tokens,
    model_assignment,
    set_context_limit_tokens,
    set_model_assignment,
    set_subagent_enabled,
    set_subagent_model_assignment,
    subagent_enabled,
    subagent_model_assignment,
)

ToggleKind = Literal[
    "git", "system", "response_schema", "todos", "rubric", "enabled", "always_allow", "plan_access", "ptc",
    "mcp_server_enabled", "mcp_server_allow", "mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc",
    "subagent_enabled",
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
INHERIT_VALUE = "__mira_default__"


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
        model_registry: Any = None,
        subagent_metadata: list[dict[str, str]] | None = None,
        initial_tab: str = "general",
        **kwargs: Any,
    ) -> None:
        super().__init__(id="settings-overlay", **kwargs)
        self.settings = settings
        self.tool_metadata = tool_metadata or []
        self.apply_change = apply_change
        self.close_panel = close_panel
        self.mcp_manager = mcp_manager
        self.model_registry = model_registry
        self.subagent_metadata = list(subagent_metadata or [])
        self.initial_tab = initial_tab if initial_tab in {"general", "models", "custom", "mcp"} else "general"
        self._model_controls_ready = False
        self._button_cells: dict[str, ToggleCell] = {}
        if self.initial_tab != "general":
            self.styles.visibility = "hidden"

    def compose(self) -> ComposeResult:
        """Compose a scrollable settings window."""
        with Vertical(id="settings-window"):
            with Horizontal(classes="settings-top-row"):
                yield Static("Settings", classes="settings-title")
                yield SettingsCloseButton("x", panel=self, id="settings-close", classes="panel-close")
            with Horizontal(id="settings-tabs"):
                yield Button(
                    "General",
                    id="settings-tab-general",
                    classes=f"settings-tab{' active' if self.initial_tab == 'general' else ''}",
                )
                yield Button(
                    "Models",
                    id="settings-tab-models",
                    classes=f"settings-tab{' active' if self.initial_tab == 'models' else ''}",
                )
                yield Button(
                    "Custom Tools",
                    id="settings-tab-custom",
                    classes=f"settings-tab{' active' if self.initial_tab == 'custom' else ''}",
                )
                yield Button(
                    "MCP",
                    id="settings-tab-mcp",
                    classes=f"settings-tab{' active' if self.initial_tab == 'mcp' else ''}",
                )
            with VerticalScroll(
                id="settings-body",
                classes="settings-body settings-general-body",
            ):
                yield Static("System Settings", classes="settings-section system")
                yield SettingsHeaderRow("", show_always=False)
                with Horizontal(classes="settings-row settings-policy-row"):
                    yield Static("Git Protection", classes="settings-label")
                    yield self._toggle_button(ToggleCell("git", "git_protection"), git_protection_enabled(self.settings))
                with Horizontal(classes="settings-row settings-policy-row"):
                    yield Static("Dynamic eval subagents", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("system", DYNAMIC_SUBAGENTS),
                        dynamic_subagents_enabled(self.settings),
                    )
                with Horizontal(classes="settings-row settings-child-row settings-policy-row"):
                    yield Static("Response schemas", classes="settings-label settings-child-label")
                    yield self._toggle_button(
                        ToggleCell("response_schema", DYNAMIC_SUBAGENT_RESPONSE_SCHEMA),
                        dynamic_subagent_response_schema_enabled(self.settings),
                    )
                with Horizontal(classes="settings-row settings-policy-row"):
                    yield Static("Planning todos", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("todos", PLANNING_TODOS),
                        planning_todos_enabled(self.settings),
                    )
                with Horizontal(
                    classes="settings-row settings-value-row planning-response-status-retries-row"
                ):
                    yield Static("Response-status retries", classes="settings-label")
                    yield Input(
                        value=str(planning_response_status_max_retries(self.settings)),
                        id="settings-planning-response-status-max-retries",
                        classes="settings-input planning-response-status-retries-input",
                    )
                with Horizontal(classes="settings-row settings-policy-row"):
                    yield Static("Rubric Middleware", classes="settings-label")
                    yield self._toggle_button(
                        ToggleCell("rubric", RUBRIC),
                        rubric_enabled(self.settings),
                    )
                with Horizontal(
                    classes="settings-row settings-child-row settings-value-row rubric-iterations-row"
                ):
                    yield Static("Maximum iterations", classes="settings-label settings-child-label")
                    yield Input(
                        value=str(rubric_max_iterations(self.settings)),
                        id="settings-rubric-max-iterations",
                        classes="settings-input rubric-iterations-input",
                        disabled=not rubric_enabled(self.settings),
                    )

                yield Static("Inbuilt Tools", classes="settings-section inbuilt")
                yield SettingsHeaderRow("", show_ptc=True)
                for tool_name in INBUILT_DANGEROUS_TOOLS:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row settings-policy-row settings-inbuilt-tool-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(
                            ToggleCell("enabled", tool_name),
                            enabled,
                        )
                        yield self._toggle_button(
                            ToggleCell("always_allow", tool_name),
                            tool_always_allow(self.settings, tool_name),
                        )
                        yield self._toggle_button(
                            ToggleCell("ptc", tool_name),
                            tool_ptc(self.settings, tool_name),
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

            with VerticalScroll(id="settings-models-body", classes="settings-body"):
                yield Static("Model Context", classes="settings-section model-context")
                with Horizontal(classes="settings-row settings-wide-row settings-value-row settings-context-row"):
                    yield Static("Context limit tokens", classes="settings-label")
                    yield Input(
                        str(context_limit_tokens(self.settings)),
                        id="settings-model-context-limit",
                        classes="settings-input settings-context-limit-input",
                    )

                yield Static("Model Assignments", classes="settings-section model-assignments")
                for role, label in (
                    (MAIN_MODEL, "Main"),
                    (RUBRIC_MODEL, "Rubric"),
                    (SUMMARIZATION_MODEL, "Summarization"),
                ):
                    with Horizontal(classes="settings-row settings-wide-row settings-model-row"):
                        yield Static(label, classes="settings-label")
                        yield Select(
                            self._role_options(role),
                            value=model_assignment(self.settings, role) or INHERIT_VALUE,
                            allow_blank=False,
                            id=f"settings-model-{role}",
                            classes="settings-select settings-model-select",
                        )

                yield Static("Subagents", classes="settings-section subagents")
                yield SettingsHeaderRow(
                    "",
                    show_always=False,
                    show_model=True,
                    row_class="settings-subagent-header",
                )
                for item in self.subagent_metadata:
                    name = str(item.get("name") or "")
                    if not name:
                        continue
                    kind = str(item.get("kind") or "raw")
                    with Horizontal(classes="settings-row settings-wide-row settings-subagent-row"):
                        yield Static(name, classes="settings-label")
                        yield self._toggle_button(
                            ToggleCell("subagent_enabled", name, locked=name == "general-purpose"),
                            subagent_enabled(self.settings, name),
                        )
                        yield Select(
                            self._subagent_options(item),
                            value=subagent_model_assignment(self.settings, name) or INHERIT_VALUE,
                            allow_blank=False,
                            disabled=kind != "raw",
                            id=f"settings-subagent-model-{safe_id(name)}",
                            classes="settings-select settings-subagent-model-select",
                        )

            with VerticalScroll(id="settings-custom-body", classes="settings-body"):
                yield Static("Custom Tools", classes="settings-section custom")
                yield SettingsHeaderRow(
                    "",
                    show_plan=True,
                    show_ptc=True,
                    row_class="settings-custom-tool-header",
                )
                custom_names = custom_tool_names(self.tool_metadata)
                if not custom_names:
                    yield Static("No custom tools loaded", classes="settings-empty")
                for tool_name in custom_names:
                    enabled = tool_enabled(self.settings, tool_name)
                    with Horizontal(classes="settings-row settings-policy-row settings-custom-tool-row"):
                        yield Static(tool_name, classes="settings-label")
                        yield self._toggle_button(ToggleCell("enabled", tool_name), enabled)
                        yield self._toggle_button(
                            ToggleCell("always_allow", tool_name),
                            tool_always_allow(self.settings, tool_name),
                        )
                        yield self._toggle_button(
                            ToggleCell("plan_access", tool_name),
                            tool_plan_access(self.settings, tool_name),
                        )
                        yield self._toggle_button(ToggleCell("ptc", tool_name), tool_ptc(self.settings, tool_name))

            with VerticalScroll(id="settings-mcp-body", classes="settings-body"):
                states = list(getattr(self.mcp_manager, "servers", {}).values())
                if not states:
                    yield Static("No MCP servers configured", classes="settings-empty")
                for index, state in enumerate(states):
                    enabled = mcp_server_enabled(self.settings, state.name)
                    group_classes = "settings-mcp-server-group"
                    if index == 0:
                        group_classes += " first"
                    with Vertical(classes=group_classes):
                        yield Static(
                            f"{state.name} [{state.transport.upper()}]",
                            classes="settings-mcp-server-title",
                            markup=False,
                        )
                        yield SettingsHeaderRow(
                            "",
                            row_class="settings-mcp-server-header",
                        )
                        with Horizontal(
                            classes="settings-row settings-policy-row settings-mcp-server-policy-row"
                        ):
                            yield Static("Server", classes="settings-label")
                            yield self._toggle_button(
                                ToggleCell("mcp_server_enabled", state.name, server=state.name),
                                enabled,
                            )
                            yield self._toggle_button(
                                ToggleCell("mcp_server_allow", state.name, server=state.name),
                                mcp_server_always_allow(self.settings, state.name),
                            )
                        yield SettingsHeaderRow(
                            "Tool",
                            show_plan=True,
                            show_ptc=True,
                            row_class="settings-mcp-tool-header",
                        )
                        if not state.tool_metadata:
                            yield Static("No tools", classes="settings-empty settings-mcp-tool-empty")
                        for item in state.tool_metadata:
                            original = item.get("original_name", "")
                            policy = mcp_tool_policy(self.settings, state.name, original)
                            with Horizontal(
                                classes="settings-row settings-policy-row settings-mcp-tool-row"
                            ):
                                yield Static(original, classes="settings-label")
                                yield self._toggle_button(
                                    ToggleCell("mcp_tool_enabled", original, not enabled, state.name),
                                    policy.enabled,
                                )
                                yield self._toggle_button(
                                    ToggleCell(
                                        "mcp_tool_allow",
                                        original,
                                        not enabled or not policy.enabled,
                                        state.name,
                                    ),
                                    policy.always_allow,
                                )
                                yield self._toggle_button(
                                    ToggleCell(
                                        "mcp_tool_plan",
                                        original,
                                        not enabled or not policy.enabled,
                                        state.name,
                                    ),
                                    policy.plan_access,
                                )
                                yield self._toggle_button(
                                    ToggleCell(
                                        "mcp_tool_ptc",
                                        original,
                                        not enabled or not policy.enabled,
                                        state.name,
                                    ),
                                    policy.ptc,
                                )

            yield Static("", id="settings-status", classes="settings-status")

    def on_mount(self) -> None:
        """Focus the first editable toggle when the panel appears."""
        # Select composes an internal overlay. Keep the whole panel invisible
        # until nested selectors mount and the requested tab is ready.
        self.call_after_refresh(self._activate_initial_tab)

    def _activate_initial_tab(self) -> None:
        self._show_tab(self.initial_tab)
        self._model_controls_ready = True
        self.styles.visibility = "visible"
        if not any(widget.has_focus for widget in self.query("Button, Input, Select")):
            self._focus_first_toggle()

    @on(Button.Pressed, ".settings-tab")
    def press_tab(self, event: Button.Pressed) -> None:
        event.stop()
        selected = (event.button.id or "").removeprefix("settings-tab-")
        self.initial_tab = selected
        self._show_tab(selected)
        self.call_after_refresh(self._focus_first_toggle)

    def _show_tab(self, selected: str) -> None:
        for name in ("general", "models", "custom", "mcp"):
            selector = "#settings-body" if name == "general" else f"#settings-{name}-body"
            self.query_one(selector).display = name == selected
            self.query_one(f"#settings-tab-{name}", Button).set_class(name == selected, "active")

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

    @on(Select.Changed, ".settings-model-select")
    async def change_model_assignment(self, event: Select.Changed) -> None:
        """Persist an assignment while keeping the inheritance option null."""
        event.stop()
        if not self._model_controls_ready:
            return
        role = (event.select.id or "").removeprefix("settings-model-")
        selected = "" if event.value == INHERIT_VALUE else str(event.value or "")
        if (selected or None) == model_assignment(self.settings, role):
            return
        if selected and selected not in self._profile_names():
            self._set_status(f"model profile '{selected}' is unavailable")
            self._restore_select_value(
                event.select,
                model_assignment(self.settings, role) or INHERIT_VALUE,
            )
            return
        updated = set_model_assignment(self.settings, role, selected or None)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            if role == MAIN_MODEL:
                self._replace_model_select_options(
                    event.select,
                    self._role_options(MAIN_MODEL),
                    selected,
                )
                self._refresh_inherited_model_labels()
        else:
            self._restore_select_value(
                event.select,
                model_assignment(self.settings, role) or INHERIT_VALUE,
            )

    @on(Select.Changed, ".settings-subagent-model-select")
    async def change_subagent_model(self, event: Select.Changed) -> None:
        """Persist a MIRA-owned raw-subagent override."""
        event.stop()
        if not self._model_controls_ready:
            return
        suffix = (event.select.id or "").removeprefix("settings-subagent-model-")
        name = next(
            (str(item.get("name")) for item in self.subagent_metadata if safe_id(str(item.get("name") or "")) == suffix),
            "",
        )
        selected = "" if event.value == INHERIT_VALUE else str(event.value or "")
        if name and (selected or None) == subagent_model_assignment(self.settings, name):
            return
        if not name or (selected and selected not in self._profile_names()):
            self._set_status("selected model profile is unavailable")
            if name:
                self._restore_select_value(
                    event.select,
                    subagent_model_assignment(self.settings, name) or INHERIT_VALUE,
                )
            return
        updated = set_subagent_model_assignment(self.settings, name, selected or None)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
        else:
            self._restore_select_value(
                event.select,
                subagent_model_assignment(self.settings, name) or INHERIT_VALUE,
            )

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
        if input_id == "settings-model-context-limit":
            try:
                limit = int(event.value.strip())
            except ValueError:
                limit = 0
            updated = set_context_limit_tokens(self.settings, limit)
            if context_limit_tokens(updated) != limit:
                event.input.value = str(context_limit_tokens(self.settings))
                self._set_status("context limit must be a positive integer")
                return
            ok, message = await self.apply_change(updated)
            self._set_status(message)
            if ok:
                self.settings = updated
            else:
                event.input.value = str(context_limit_tokens(self.settings))
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
            override = model_assignment(self.settings, RUBRIC_MODEL)
            if value and override and override not in self._profile_names():
                self._set_status(f"model profile '{override}' is unavailable")
                return
            updated = set_rubric_enabled(self.settings, value)
        elif cell.kind == "subagent_enabled":
            override = subagent_model_assignment(self.settings, cell.name)
            if value and override and override not in self._profile_names():
                self._set_status(f"model profile '{override}' is unavailable")
                return
            updated = set_subagent_enabled(self.settings, cell.name, value)
        elif cell.kind == "enabled":
            updated = set_tool_enabled(self.settings, cell.name, value)
        elif cell.kind == "plan_access":
            updated = set_tool_plan_access(self.settings, cell.name, value)
        elif cell.kind == "ptc":
            updated = set_tool_ptc(self.settings, cell.name, value)
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
        elif cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc"}:
            key = {
                "mcp_tool_enabled": "enabled",
                "mcp_tool_allow": "always_allow",
                "mcp_tool_plan": "plan_access",
                "mcp_tool_ptc": "ptc",
            }[cell.kind]
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
        label = button_label(cell, value)
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
            button.label = button_label(cell, value)
            button.disabled = locked
            button.set_classes(toggle_classes(value, locked=locked))

    def _cell_locked(self, cell: ToggleCell) -> bool:
        if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc"}:
            if not mcp_server_enabled(self.settings, cell.server) or not self._mcp_tool_available(cell):
                return True
            if cell.kind != "mcp_tool_enabled" and not mcp_tool_policy(self.settings, cell.server, cell.name).enabled:
                return True
            if cell.kind == "mcp_tool_ptc" and not tool_enabled(self.settings, "eval"):
                return True
            return False
        if cell.kind in {"always_allow", "plan_access", "ptc"}:
            if cell.kind == "ptc" and (
                cell.name in PTC_INAPPLICABLE_TOOLS or not tool_enabled(self.settings, "eval")
            ):
                return True
            return not tool_enabled(self.settings, cell.name)
        if cell.kind == "system":
            return not tool_enabled(self.settings, "eval") or not tool_enabled(self.settings, "task")
        if cell.kind == "response_schema":
            return not dynamic_subagents_enabled(self.settings)
        return cell.locked

    def _mcp_tool_available(self, cell: ToggleCell) -> bool:
        state = getattr(self.mcp_manager, "servers", {}).get(cell.server)
        if state is None or not bool(getattr(state, "usable", False)):
            return False
        metadata = list(getattr(state, "tool_metadata", []) or [])
        tools = getattr(state, "tools", None)
        if not isinstance(tools, list):
            return any(item.get("original_name") == cell.name for item in metadata)
        return any(
            item.get("original_name") == cell.name and index < len(tools)
            for index, item in enumerate(metadata)
        )

    def _set_status(self, message: str) -> None:
        self.query_one("#settings-status", Static).update(message)

    def _restore_select_value(self, select: Select, value: str) -> None:
        """Restore a rejected model selection without firing another save."""
        with select.prevent(Select.Changed):
            select.value = value

    def _replace_model_select_options(
        self,
        select: Select,
        options: list[tuple[Any, str]],
        value: str,
    ) -> None:
        """Replace options and repaint a value whose identity did not change."""
        with select.prevent(Select.Changed):
            select.set_options(options)
            select.value = value
        label = next((label for label, option_value in options if option_value == value), value)
        select.query_one("SelectCurrent").update(label)

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

    def _profile_names(self) -> list[str]:
        return list(getattr(self.model_registry, "profiles", {}))

    def _role_options(self, role: str) -> list[tuple[str, str]]:
        assigned = model_assignment(self.settings, role)
        names = self._profile_names()
        options: list[tuple[str, str]] = []
        if role != MAIN_MODEL:
            options.append((f"{model_assignment(self.settings, MAIN_MODEL) or 'unset'} (default)", INHERIT_VALUE))
        elif not assigned:
            options.append(("unset", INHERIT_VALUE))
        if assigned and assigned not in names:
            options.append((f"{assigned} (missing)", assigned))
        options.extend((name, name) for name in names)
        return options

    def _subagent_options(self, item: dict[str, str]) -> list[tuple[Any, str]]:
        name = str(item.get("name") or "")
        kind = str(item.get("kind") or "raw")
        if kind == "compiled":
            return [(Text("[compiled]"), INHERIT_VALUE)]
        if kind == "async":
            graph_id = str(item.get("graph_id") or name)
            return [(Text(f"[async] {graph_id}"), INHERIT_VALUE)]
        source = str(item.get("source_model") or "")
        inherited = source or f"{model_assignment(self.settings, MAIN_MODEL) or 'unset'} (default)"
        assigned = subagent_model_assignment(self.settings, name)
        names = self._profile_names()
        inherited_label: Any = Text(inherited) if inherited.startswith("[") else inherited
        options = [(inherited_label, INHERIT_VALUE)]
        if assigned and assigned not in names:
            options.append((f"{assigned} (missing)", assigned))
        options.extend((profile, profile) for profile in names)
        return options

    def _refresh_inherited_model_labels(self) -> None:
        for selector in self.query(".settings-model-select"):
            role = (selector.id or "").removeprefix("settings-model-")
            if role != MAIN_MODEL and not model_assignment(self.settings, role):
                self._replace_model_select_options(
                    selector,
                    self._role_options(role),
                    INHERIT_VALUE,
                )
        for item in self.subagent_metadata:
            name = str(item.get("name") or "")
            if not name or subagent_model_assignment(self.settings, name):
                continue
            selector = self.query_one(f"#settings-subagent-model-{safe_id(name)}", Select)
            self._replace_model_select_options(
                selector,
                self._subagent_options(item),
                INHERIT_VALUE,
            )


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
    if cell.kind == "subagent_enabled":
        return subagent_enabled(settings, cell.name)
    if cell.kind == "enabled":
        return tool_enabled(settings, cell.name)
    if cell.kind == "plan_access":
        return tool_plan_access(settings, cell.name)
    if cell.kind == "ptc":
        return tool_ptc(settings, cell.name)
    if cell.kind == "mcp_server_enabled":
        return mcp_server_enabled(settings, cell.server)
    if cell.kind == "mcp_server_allow":
        return mcp_server_always_allow(settings, cell.server)
    if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc"}:
        policy = mcp_tool_policy(settings, cell.server, cell.name)
        return {
            "mcp_tool_enabled": policy.enabled,
            "mcp_tool_allow": policy.always_allow,
            "mcp_tool_plan": policy.plan_access,
            "mcp_tool_ptc": policy.ptc,
        }[cell.kind]
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a settings button."""
    identity = f"{cell.server}-{cell.name}" if cell.server else cell.name
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", identity)
    return f"settings-toggle-{cell.kind}-{safe_name}"


def button_label(cell: ToggleCell, value: bool) -> str:
    """Return display text for a settings toggle."""
    if cell.kind == "ptc" and cell.name in PTC_INAPPLICABLE_TOOLS and not cell.server:
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

    def __init__(
        self,
        name_label: str,
        *,
        show_always: bool = True,
        show_plan: bool = False,
        show_ptc: bool = False,
        show_model: bool = False,
        row_class: str = "",
    ) -> None:
        classes = "settings-row settings-header-row"
        if row_class:
            classes += f" {row_class}"
        super().__init__(classes=classes)
        self.name_label = name_label
        self.show_always = show_always
        self.show_plan = show_plan
        self.show_ptc = show_ptc
        self.show_model = show_model

    def compose(self) -> ComposeResult:
        yield Static(self.name_label, classes="settings-column-label name")
        yield Static("enable", classes="settings-column-label enabled")
        if self.show_always:
            yield Static("always allow", classes="settings-column-label always")
        if self.show_plan:
            yield Static("plan access", classes="settings-column-label plan")
        if self.show_ptc:
            yield Static("PTC", classes="settings-column-label ptc")
        if self.show_model:
            yield Static("model", classes="settings-column-label model")


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value)


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
