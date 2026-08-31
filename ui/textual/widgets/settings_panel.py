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
from textual.widgets import Button, ContentSwitcher, Input, Select, Static

from agent.planning.policy import PLAN_DISABLED_TOOLS
from config.settings import (
    DYNAMIC_SUBAGENTS,
    DYNAMIC_SUBAGENT_RESPONSE_SCHEMA,
    EXECUTE_TOOL,
    EXECUTE_ENV_MODES,
    INBUILT_DANGEROUS_TOOLS,
    INBUILT_TOOLS,
    PTC_INAPPLICABLE_TOOLS,
    READ_ONLY_BUILTIN_TOOLS,
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
    set_tool_rubric_access,
    tool_always_allow,
    tool_enabled,
    tool_plan_access,
    tool_ptc,
    tool_rubric_access,
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
    set_tracing_enabled,
    set_middleware_span_mode,
    set_tracing_profile,
    tracing_enabled,
    tracing_settings,
)
from config.tracing import TracingRegistry
from tracing.bootstrap import tracing_yaml_fragment

ToggleKind = Literal[
    "git", "system", "response_schema", "todos", "rubric", "enabled", "always_allow", "plan_access", "ptc",
    "rubric_access", "mcp_server_enabled", "mcp_server_allow", "mcp_tool_enabled", "mcp_tool_allow",
    "mcp_tool_plan", "mcp_tool_ptc", "mcp_tool_rubric",
    "subagent_enabled", "tracing",
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
ReloadCallback = Callable[[], Awaitable[bool]]
CloseCallback = Callable[[], None]
INHERIT_VALUE = "__mira_default__"
SETTINGS_TAB_PAGES = {
    "general": "settings-body",
    "models": "settings-models-body",
    "custom": "settings-custom-body",
    "mcp": "settings-mcp-body",
    "access": "settings-access-body",
}
ACCESS_POLICIES = ("plan", "ptc", "rubric")
ACCESS_POLICY_LABELS = {"plan": "Plan", "ptc": "PTC", "rubric": "Rubric"}
ACCESS_POLICY_LABEL_WIDTH = 52
ACCESS_POLICY_GUIDANCE = {
    "plan": "Only enabled tools are shown. Read-only access can be changed; action tools are blocked in Plan mode.",
    "ptc": "Only enabled tools are shown. Choose which tools can be called programmatically through eval.",
    "rubric": "Only enabled tools are shown. Choose which tools the Rubric verifier can use; some tools are blocked by policy.",
}


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
        reload_runtime: ReloadCallback | None = None,
        close_panel: CloseCallback | None = None,
        mcp_manager: Any = None,
        model_registry: Any = None,
        tracing_registry: TracingRegistry | None = None,
        subagent_metadata: list[dict[str, str]] | None = None,
        initial_tab: str = "general",
        **kwargs: Any,
    ) -> None:
        super().__init__(id="settings-overlay", **kwargs)
        self.settings = settings
        self.tool_metadata = tool_metadata or []
        self.apply_change = apply_change
        self.reload_runtime = reload_runtime
        self.close_panel = close_panel
        self.mcp_manager = mcp_manager
        self.model_registry = model_registry
        self.tracing_registry = tracing_registry or TracingRegistry()
        self.subagent_metadata = list(subagent_metadata or [])
        self.initial_tab = initial_tab if initial_tab in SETTINGS_TAB_PAGES else "general"
        self._model_controls_ready = False
        self._button_cells: dict[str, ToggleCell] = {}

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
                yield Button(
                    "Access",
                    id="settings-tab-access",
                    classes=f"settings-tab{' active' if self.initial_tab == 'access' else ''}",
                )
            with ContentSwitcher(
                initial=SETTINGS_TAB_PAGES[self.initial_tab],
                id="settings-switcher",
            ):
                with VerticalScroll(
                    id="settings-body",
                    classes="settings-body settings-general-body",
                ):
                    yield Static("System Settings", classes="settings-section system")
                    yield SettingsHeaderRow("", show_always=False)
                    with Horizontal(classes="settings-row settings-policy-row"):
                        yield Static("Git Protection", classes="settings-label")
                        yield self._toggle_button(ToggleCell("git", "git_protection"), git_protection_enabled(self.settings))
                    with Horizontal(classes="settings-row settings-policy-row settings-tracing-row"):
                        yield Static("Tracing", classes="settings-label")
                        yield self._toggle_button(
                            ToggleCell("tracing", "tracing"),
                            tracing_enabled(self.settings),
                        )
                        yield Button(
                            "Config",
                            id="settings-tracing-config",
                            classes="settings-config-button",
                            disabled=not tracing_enabled(self.settings),
                        )
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
                    yield SettingsHeaderRow("")
                    for tool_name in INBUILT_TOOLS:
                        enabled = tool_enabled(self.settings, tool_name)
                        with Horizontal(classes="settings-row settings-policy-row settings-inbuilt-tool-row"):
                            yield Static(tool_name, classes="settings-label")
                            yield self._toggle_button(
                                ToggleCell("enabled", tool_name),
                                enabled,
                            )
                            yield self._toggle_button(
                                ToggleCell(
                                    "always_allow",
                                    tool_name,
                                    locked=tool_name in READ_ONLY_BUILTIN_TOOLS,
                                ),
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

                    with ContentSwitcher(
                        initial=f"settings-execute-env-page-{execute_env_mode}",
                        id="settings-execute-env-switcher",
                    ):
                        with Vertical(
                            id="settings-execute-env-page-system",
                            classes="settings-execute-env-page",
                        ):
                            pass
                        for mode, (key, label, placeholder) in EXECUTE_ENV_FIELDS.items():
                            with Vertical(
                                id=f"settings-execute-env-page-{mode}",
                                classes="settings-execute-env-page",
                            ):
                                with Horizontal(
                                    id=f"settings-execute-env-{key}-row",
                                    classes="settings-row settings-wide-row",
                                ):
                                    yield Static(label, classes="settings-label")
                                    yield Input(
                                        value=str(execute_env.get(key) or ""),
                                        placeholder=f"<{placeholder}>",
                                        id=f"settings-execute-env-{key}",
                                        classes="settings-input",
                                    )

                    with Horizontal(
                        id="settings-execute-env-allow-row",
                        classes="settings-row settings-wide-row",
                    ):
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

                with VerticalScroll(id="settings-mcp-body", classes="settings-body"):
                    yield Static(
                        "MCP access changes apply immediately; config or environment edits require Reload Runtime.",
                        id="settings-mcp-guidance",
                        classes="settings-mcp-guidance",
                    )
                    states = list(getattr(self.mcp_manager, "servers", {}).values())
                    if not states:
                        yield Static(
                            "No MCP servers configured",
                            classes="settings-empty settings-mcp-empty",
                        )
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
                with Vertical(id="settings-access-body", classes="settings-body"):
                    with ContentSwitcher(initial="settings-access-landing", id="settings-access-switcher"):
                        with VerticalScroll(id="settings-access-landing"):
                            yield Static(
                                "Tool Access",
                                classes="settings-access-heading settings-access-landing-heading",
                            )
                            yield Static(
                                "Review and configure tool access for each policy.",
                                classes="settings-access-description",
                            )
                            for policy in ACCESS_POLICIES:
                                yield Button(
                                    self._access_policy_label(policy),
                                    id=f"settings-access-policy-{policy}",
                                    classes="settings-access-policy",
                                )
                        with VerticalScroll(id="settings-access-detail"):
                            pass

            with Horizontal(id="settings-footer"):
                yield Static("", id="settings-status", classes="settings-status")
                yield Button(
                    "Reload runtime",
                    id="settings-reload-runtime",
                    classes="settings-config-action settings-reload-button settings-footer-reload",
                )

    def on_mount(self) -> None:
        """Focus the first editable toggle when the panel appears."""
        self.call_after_refresh(self._finish_mount)

    def _finish_mount(self) -> None:
        self._model_controls_ready = True
        switcher = self.query_one("#settings-switcher", ContentSwitcher)
        if switcher.current == SETTINGS_TAB_PAGES["access"]:
            self.screen.set_focus(None)
            return
        if not any(widget.has_focus for widget in self.query("Button, Input, Select")):
            self._focus_first_toggle()

    @on(Button.Pressed, ".settings-tab")
    def press_tab(self, event: Button.Pressed) -> None:
        event.stop()
        selected = (event.button.id or "").removeprefix("settings-tab-")
        page_id = SETTINGS_TAB_PAGES.get(selected)
        if page_id is None:
            return
        self.query_one("#settings-switcher", ContentSwitcher).current = page_id
        for name in SETTINGS_TAB_PAGES:
            self.query_one(f"#settings-tab-{name}", Button).set_class(name == selected, "active")
        if selected == "access":
            self._show_access_landing()
        else:
            self.call_after_refresh(self._focus_first_toggle)

    @on(Button.Pressed, ".settings-access-policy")
    async def press_access_policy(self, event: Button.Pressed) -> None:
        """Recycle the shared detail body for the selected access policy."""
        event.stop()
        policy = (event.button.id or "").removeprefix("settings-access-policy-")
        if policy not in ACCESS_POLICIES:
            return
        detail = self.query_one("#settings-access-detail", VerticalScroll)
        await detail.remove_children()
        await detail.mount(*self._access_widgets(policy))
        self.query_one("#settings-access-switcher", ContentSwitcher).current = "settings-access-detail"
        self.call_after_refresh(detail.query_one("#settings-access-back", Button).focus)

    @on(Button.Pressed, "#settings-tracing-config")
    async def press_tracing_config(self, event: Button.Pressed) -> None:
        """Open tracing details inside the existing Settings switcher."""
        event.stop()
        switcher = self.query_one("#settings-switcher", ContentSwitcher)
        existing = list(self.query("#settings-tracing-detail"))
        if existing:
            await existing[0].remove()
        detail = VerticalScroll(*self._tracing_widgets(), id="settings-tracing-detail", classes="settings-body")
        await switcher.mount(detail)
        switcher.current = "settings-tracing-detail"
        self.call_after_refresh(detail.query_one("#settings-tracing-profile", Select).focus)

    @on(Button.Pressed, "#settings-tracing-back")
    async def press_tracing_back(self, event: Button.Pressed) -> None:
        """Return to General settings after profile changes save in place."""
        event.stop()
        await self._close_tracing_detail()

    @on(Select.Changed, "#settings-tracing-profile")
    async def change_tracing_profile(self, event: Select.Changed) -> None:
        """Persist the selected registry profile and refresh its effective preview."""
        event.stop()
        selected = str(event.value) if event.value is not Select.NULL else ""
        current = str(tracing_settings(self.settings)["profile"])
        if not selected or selected == current:
            return
        if selected not in self.tracing_registry.profiles:
            self._set_status(f"tracing profile '{selected}' is unavailable")
            self._restore_select_value(event.select, current)
            return
        updated = set_tracing_profile(self.settings, selected)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            self._refresh_tracing_preview()
        else:
            self._restore_select_value(event.select, current)

    @on(Select.Changed, "#settings-middleware-spans")
    async def change_middleware_spans(self, event: Select.Changed) -> None:
        """Persist middleware-span visibility and rebuild constructed graphs."""
        event.stop()
        selected = str(event.value) if event.value is not Select.NULL else ""
        current = str(tracing_settings(self.settings)["middleware_spans"])
        if not selected or selected == current:
            return
        updated = set_middleware_span_mode(self.settings, selected)
        ok, message = await self.apply_change(updated)
        self._set_status(message)
        if ok:
            self.settings = updated
            self._refresh_tracing_preview()
        else:
            self._restore_select_value(event.select, current)

    @on(Button.Pressed, "#settings-access-back")
    def press_access_back(self, event: Button.Pressed) -> None:
        """Return from the shared policy detail to the Access landing page."""
        event.stop()
        self._show_access_landing()

    async def on_key(self, event: Key) -> None:
        """Handle direct yes/no and close shortcuts."""
        key = event.key.lower()
        if key == "escape" and self._tracing_detail_visible():
            event.stop()
            await self._close_tracing_detail()
            return
        if key == "escape" and self._access_detail_visible():
            event.stop()
            self._show_access_landing()
            return
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
            self.query_one("#settings-execute-env-switcher", ContentSwitcher).current = (
                f"settings-execute-env-page-{event.value}"
            )
            allow_row = self.query_one("#settings-execute-env-allow-row", Horizontal)
            self.call_after_refresh(
                allow_row.scroll_visible,
                animate=False,
                immediate=True,
            )
        else:
            self._restore_select_value(event.select, current)

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
            saved = execute_env_settings(self.settings)
            if input_id == "settings-execute-env-allow":
                event.input.value = ", ".join(saved.get("allow") or [])
            else:
                key = input_id.removeprefix("settings-execute-env-")
                event.input.value = str(saved.get(key) or "")

    @on(Button.Pressed, "#settings-close")
    def press_close(self, event: Button.Pressed) -> None:
        """Close the settings panel from the visible close button."""
        event.stop()
        self._close()

    @on(Button.Pressed, "#settings-reload-runtime")
    async def press_reload_runtime(self, event: Button.Pressed) -> None:
        """Run the shared full-runtime reload from the Settings footer."""
        event.stop()
        await self._reload_from_settings("runtime reloaded")

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
        elif cell.kind == "tracing":
            updated = set_tracing_enabled(self.settings, value)
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
        elif cell.kind == "rubric_access":
            updated = set_tool_rubric_access(self.settings, cell.name, value)
        elif cell.kind == "mcp_server_enabled":
            if self.mcp_manager is None or not await self.mcp_manager.set_server_enabled(cell.server, value):
                self._set_status("could not update MCP server")
                return
            from config.settings import load_settings

            self.settings = load_settings(self.mcp_manager.workspace)
            self._set_status("MCP server updated; agents rebuilt; no reload required")
            self._refresh_buttons()
            self._refresh_access_counts()
            return
        elif cell.kind == "mcp_server_allow":
            if self.mcp_manager is None or not await self.mcp_manager.set_server_always_allow(cell.server, value):
                self._set_status("could not update MCP server approval")
                return
            from config.settings import load_settings

            self.settings = load_settings(self.mcp_manager.workspace)
            self._set_status("MCP approval updated; no reload required")
            self._refresh_buttons()
            return
        elif cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc", "mcp_tool_rubric"}:
            key = {
                "mcp_tool_enabled": "enabled",
                "mcp_tool_allow": "always_allow",
                "mcp_tool_plan": "plan_access",
                "mcp_tool_ptc": "ptc",
                "mcp_tool_rubric": "rubric",
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
        self._refresh_tracing_control()
        self._refresh_access_counts()
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
            matches = list(self.query(f"#{button_id}"))
            if not matches:
                continue
            button = matches[0]
            locked = self._cell_locked(cell)
            value = selected_value(self.settings, cell)
            button.label = button_label(cell, value)
            button.disabled = locked
            button.set_classes(toggle_classes(value, locked=locked))

    def _cell_locked(self, cell: ToggleCell) -> bool:
        if cell.locked:
            return True
        if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc", "mcp_tool_rubric"}:
            if not mcp_server_enabled(self.settings, cell.server) or not self._mcp_tool_available(cell):
                return True
            if cell.kind != "mcp_tool_enabled" and not mcp_tool_policy(self.settings, cell.server, cell.name).enabled:
                return True
            if cell.kind == "mcp_tool_ptc" and not tool_enabled(self.settings, "eval"):
                return True
            return False
        if cell.kind in {"always_allow", "plan_access", "ptc", "rubric_access"}:
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

    async def _reload_from_settings(self, success_message: str) -> None:
        """Run the shared reload callback and keep Settings feedback truthful."""
        if self.reload_runtime is None:
            self._set_status("runtime reload unavailable")
            return
        buttons = list(self.query("#settings-reload-runtime"))
        for button in buttons:
            button.disabled = True
        try:
            reloaded = await self.reload_runtime()
        except Exception:
            reloaded = False
        finally:
            for button in buttons:
                if button.is_mounted:
                    button.disabled = False
        self._set_status(success_message if reloaded else "runtime reload failed")

    def refresh_tracing_registry(self, registry: TracingRegistry) -> None:
        """Refresh tracing choices after the shared runtime reload."""
        self.tracing_registry = registry
        selects = list(self.query("#settings-tracing-profile"))
        if not selects:
            return
        select = selects[0]
        options = self._tracing_profile_options()
        if not options:
            self._refresh_tracing_preview()
            return
        selected = str(tracing_settings(self.settings)["profile"])
        value = selected if selected in self.tracing_registry.profiles else options[0][1]
        with select.prevent(Select.Changed):
            select.set_options(options)
            select.value = value
        select.query_one("SelectCurrent").update(self._tracing_profile_label(str(value)))
        self._refresh_tracing_preview()

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

    def _refresh_rubric_control(self) -> None:
        """Refresh dependent rubric controls after the parent toggle changes."""
        control = self.query_one("#settings-rubric-max-iterations", Input)
        control.disabled = not rubric_enabled(self.settings)
        control.value = str(rubric_max_iterations(self.settings))

    def _refresh_tracing_control(self) -> None:
        """Keep Config available only while tracing is enabled."""
        controls = list(self.query("#settings-tracing-config"))
        if controls:
            controls[0].disabled = not tracing_enabled(self.settings)

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

    def _access_policy_label(self, policy: str) -> str:
        count = sum(
            value for _title, rows in self._access_groups(policy) for _name, _cell, value in rows
        )
        suffix = f"{count} allowed      >"
        label = ACCESS_POLICY_LABELS[policy]
        return f"{label}{suffix.rjust(ACCESS_POLICY_LABEL_WIDTH - len(label))}"

    def _access_detail_visible(self) -> bool:
        switcher = self.query_one("#settings-access-switcher", ContentSwitcher)
        return switcher.current == "settings-access-detail"

    def _tracing_detail_visible(self) -> bool:
        return self.query_one("#settings-switcher", ContentSwitcher).current == "settings-tracing-detail"

    async def _close_tracing_detail(self) -> None:
        switcher = self.query_one("#settings-switcher", ContentSwitcher)
        switcher.current = "settings-body"
        matches = list(self.query("#settings-tracing-detail"))
        if matches:
            await matches[0].remove()
        self.call_after_refresh(self.query_one("#settings-tracing-config", Button).focus)

    def _tracing_widgets(self) -> list[Any]:
        values = tracing_settings(self.settings)
        profile = str(values["profile"])
        options = self._tracing_profile_options()
        available = profile in self.tracing_registry.profiles
        return [
            Button(
                "< Back",
                id="settings-tracing-back",
                classes="settings-access-back",
                compact=True,
            ),
            Static(
                "Tracing Configuration",
                classes="settings-access-heading settings-access-detail-heading settings-tracing-heading",
            ),
            Static("Profile", classes="settings-tracing-label"),
            Select(
                options,
                value=profile if available else Select.NULL,
                prompt="Select profile",
                allow_blank=not available,
                disabled=not options,
                id="settings-tracing-profile",
                classes="settings-tracing-profile",
            ),
            Static("Middleware spans", classes="settings-tracing-label"),
            Select(
                [("Hidden", "hidden"), ("Full", "full")],
                value=str(values["middleware_spans"]),
                allow_blank=False,
                id="settings-middleware-spans",
                classes="settings-tracing-profile",
            ),
            Static(
                "Hidden removes LangChain middleware tracing noise.\n"
                "Full preserves the complete framework trace.",
                classes="settings-tracing-help",
            ),
            Static("settings.yml preview", classes="settings-tracing-label"),
            Static(
                self._tracing_preview_text(),
                id="settings-tracing-preview",
                classes="settings-tracing-preview",
                markup=False,
            ),
        ]

    def _refresh_tracing_preview(self) -> None:
        previews = list(self.query("#settings-tracing-preview"))
        if previews:
            previews[0].update(self._tracing_preview_text())

    def _tracing_preview_text(self) -> str:
        values = tracing_settings(self.settings)
        selected = str(values["profile"])
        profile = self.tracing_registry.profile(selected)
        if profile is None:
            return f"Preview unavailable: tracing profile '{selected}' was not found"
        return tracing_yaml_fragment(
            enabled=bool(values["enabled"]),
            profile=selected,
            middleware_spans=str(values["middleware_spans"]),
            endpoint=profile.endpoint,
            headers=profile.headers,
            span_attributes=profile.span_attributes,
        )

    def _tracing_profile_options(self) -> list[tuple[str, str]]:
        return [
            (self._tracing_profile_label(name), name)
            for name in self.tracing_registry.profiles
        ]

    @staticmethod
    def _tracing_profile_label(name: str) -> str:
        return name[:1].upper() + name[1:]

    def _show_access_landing(self) -> None:
        self.query_one("#settings-access-switcher", ContentSwitcher).current = "settings-access-landing"
        self.screen.set_focus(None)

    def _refresh_access_counts(self) -> None:
        for policy in ACCESS_POLICIES:
            buttons = list(self.query(f"#settings-access-policy-{policy}"))
            if buttons:
                buttons[0].label = self._access_policy_label(policy)

    def _access_groups(self, policy: str) -> list[tuple[str, list[tuple[str, ToggleCell, bool]]]]:
        """Discover enabled rows once for both Access counts and the shared body."""
        local_kind = {"plan": "plan_access", "ptc": "ptc", "rubric": "rubric_access"}[policy]
        mcp_kind = {"plan": "mcp_tool_plan", "ptc": "mcp_tool_ptc", "rubric": "mcp_tool_rubric"}[policy]
        groups: list[tuple[str, list[tuple[str, ToggleCell, bool]]]] = []

        def build_rows(
            names: list[str],
            kind: ToggleKind = local_kind,
            server: str = "",
        ) -> list[tuple[str, ToggleCell, bool]]:
            cells = [ToggleCell(kind, name, server=server) for name in names]
            return [(cell.name, cell, selected_value(self.settings, cell)) for cell in cells]

        builtin_rows = []
        for name in INBUILT_TOOLS:
            if not tool_enabled(self.settings, name):
                continue
            locked = fixed_builtin_access_value(policy, name) is not None
            cell = ToggleCell(local_kind, name, locked=locked)
            builtin_rows.append((name, cell, selected_value(self.settings, cell)))
        if builtin_rows:
            groups.append(("Inbuilt Tools", builtin_rows))

        custom_rows = build_rows(
            [name for name in custom_tool_names(self.tool_metadata) if tool_enabled(self.settings, name)]
        )
        if custom_rows:
            groups.append(("Custom Tools", custom_rows))
        for state in getattr(self.mcp_manager, "servers", {}).values():
            if not mcp_server_enabled(self.settings, state.name):
                continue
            access_rows = []
            for item in state.tool_metadata:
                original = item.get("original_name", "")
                tool_policy = mcp_tool_policy(self.settings, state.name, original)
                if tool_policy.enabled:
                    access_rows.extend(build_rows([original], mcp_kind, state.name))
            if access_rows:
                groups.append((f"MCP · {state.name} [{state.transport.upper()}]", access_rows))
        return groups

    def _access_widgets(self, policy: str) -> list[Any]:
        label = ACCESS_POLICY_LABELS[policy]
        widgets: list[Any] = [
            Button(
                "< Back",
                id="settings-access-back",
                classes="settings-access-back",
                compact=True,
            ),
            Static(
                f"{label} Access",
                classes="settings-access-heading settings-access-detail-heading",
            ),
            Static(
                ACCESS_POLICY_GUIDANCE[policy],
                classes="settings-access-guidance",
            ),
        ]
        groups = self._access_groups(policy)
        if not groups:
            widgets.append(
                Static(f"No enabled tools available for {label}.", classes="settings-empty")
            )
            return widgets
        for index, (title, rows) in enumerate(groups):
            if title == "Inbuilt Tools":
                title_classes = "settings-section inbuilt settings-access-section"
            elif title == "Custom Tools":
                title_classes = "settings-section custom settings-access-section"
            else:
                title_classes = "settings-mcp-server-title settings-access-section"
            if index:
                title_classes += " subsequent"
            widgets.append(Static(title, classes=title_classes, markup=False))
            widgets.append(
                SettingsHeaderRow(
                    "",
                    show_always=False,
                    enabled_label="Access",
                    row_class="settings-access-header",
                )
            )
            for name, cell, value in rows:
                fixed = fixed_builtin_access_value(policy, name) if title == "Inbuilt Tools" else None
                value_widget: Button | Static
                if fixed is None:
                    value_widget = self._toggle_button(cell, value)
                else:
                    fixed_class = "blocked" if fixed == "blocked" else "inapplicable"
                    value_widget = Static(
                        fixed,
                        id=f"settings-access-fixed-{policy}-{safe_id(name)}",
                        classes=f"settings-mode settings-access-fixed-value {fixed_class}",
                    )
                widgets.append(
                    Horizontal(
                        Static(name, classes="settings-label"),
                        value_widget,
                        classes="settings-row settings-policy-row settings-access-tool-row",
                    )
                )
        return widgets

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
    if cell.kind == "tracing":
        return tracing_enabled(settings)
    if cell.kind == "subagent_enabled":
        return subagent_enabled(settings, cell.name)
    if cell.kind == "enabled":
        return tool_enabled(settings, cell.name)
    if cell.kind == "plan_access":
        return tool_plan_access(settings, cell.name)
    if cell.kind == "ptc":
        return tool_ptc(settings, cell.name)
    if cell.kind == "rubric_access":
        return tool_rubric_access(settings, cell.name)
    if cell.kind == "mcp_server_enabled":
        return mcp_server_enabled(settings, cell.server)
    if cell.kind == "mcp_server_allow":
        return mcp_server_always_allow(settings, cell.server)
    if cell.kind in {"mcp_tool_enabled", "mcp_tool_allow", "mcp_tool_plan", "mcp_tool_ptc", "mcp_tool_rubric"}:
        policy = mcp_tool_policy(settings, cell.server, cell.name)
        return {
            "mcp_tool_enabled": policy.enabled,
            "mcp_tool_allow": policy.always_allow,
            "mcp_tool_plan": policy.plan_access,
            "mcp_tool_ptc": policy.ptc,
            "mcp_tool_rubric": policy.rubric,
        }[cell.kind]
    return tool_always_allow(settings, cell.name)


def button_id_for(cell: ToggleCell) -> str:
    """Return a stable Textual id for a settings button."""
    identity = f"{cell.server}-{cell.name}" if cell.server else cell.name
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", identity)
    return f"settings-toggle-{cell.kind}-{safe_name}"


def fixed_builtin_access_value(policy: str, tool_name: str) -> str | None:
    """Return the non-interactive value for a fixed built-in policy row."""
    if policy == "plan" and tool_name in PLAN_DISABLED_TOOLS:
        return "blocked"
    if policy == "ptc" and tool_name in PTC_INAPPLICABLE_TOOLS:
        return "-"
    if (
        policy == "rubric"
        and tool_name in INBUILT_DANGEROUS_TOOLS
        and tool_name != EXECUTE_TOOL
    ):
        return "blocked"
    return None


def button_label(cell: ToggleCell, value: bool) -> str:
    """Return display text for a settings toggle."""
    if cell.kind == "always_allow" and cell.name in READ_ONLY_BUILTIN_TOOLS and not cell.server:
        return "-"
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
        if key == "escape" and self.panel._access_detail_visible():
            event.stop()
            self.panel._show_access_landing()
            return
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
        show_model: bool = False,
        enabled_label: str = "enable",
        row_class: str = "",
    ) -> None:
        classes = "settings-row settings-header-row"
        if row_class:
            classes += f" {row_class}"
        super().__init__(classes=classes)
        self.name_label = name_label
        self.show_always = show_always
        self.show_model = show_model
        self.enabled_label = enabled_label

    def compose(self) -> ComposeResult:
        yield Static(self.name_label, classes="settings-column-label name")
        yield Static(self.enabled_label, classes="settings-column-label enabled")
        if self.show_always:
            yield Static("always allow", classes="settings-column-label always")
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
