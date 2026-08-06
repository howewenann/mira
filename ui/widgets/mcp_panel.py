"""Expandable MCP server status and lifecycle controls."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ui.spinners import SPINNER_FRAMES


class MCPPanelScreen(ModalScreen[None]):
    """Render every parsed server directly from the manager registry."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, manager: Any) -> None:
        super().__init__()
        self.manager = manager
        self.expanded: set[str] = set()
        self._spinner = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-dialog"):
            with Horizontal(id="mcp-title-row"):
                yield Static("MCP SERVERS", id="mcp-title")
                yield Button("Close (c)", id="mcp-close")
            with VerticalScroll(id="mcp-scroll"):
                for index, state in enumerate(self.manager.servers.values()):
                    yield from self._server_widgets(state)
                    if index + 1 < len(self.manager.servers):
                        yield Static("-" * 64, classes="mcp-divider")

    def _server_widgets(self, state: Any) -> ComposeResult:
        expanded = state.name in self.expanded
        marker = "-" if expanded else "+"
        status = state.status
        if state.transient:
            status = f"{SPINNER_FRAMES[self._spinner % len(SPINNER_FRAMES)]} {status}"
        yield Button(
            f"{marker} {state.name} · {state.transport} · {status} · {capability_counts(state)}",
            id=f"mcp-header-{safe_id(state.name)}",
            classes=f"mcp-server-header {status_class(state.status)}",
            name=state.name,
        )
        with Horizontal(classes="mcp-controls"):
            for action in controls_for(state.status):
                button = Button(
                    action,
                    id=f"mcp-{action.lower()}-{safe_id(state.name)}",
                    classes="mcp-control",
                    name=state.name,
                )
                button.disabled = state.transient
                yield button
        if expanded:
            yield Static(server_details(state), classes="mcp-details", markup=False)

    def on_mount(self) -> None:
        self.set_interval(0.12, self._tick)
        headers = list(self.query(".mcp-server-header"))
        if headers:
            headers[0].focus()

    @on(Button.Pressed, ".mcp-server-header")
    def toggle_server(self, event: Button.Pressed) -> None:
        event.stop()
        name = event.button.name or ""
        if name in self.expanded:
            self.expanded.remove(name)
        else:
            self.expanded.add(name)
            self.run_worker(self._discover_and_refresh(name), name=f"mcp-discover-{safe_id(name)}", exclusive=False)
        self.run_worker(self.refresh_from_manager(), name="mcp-panel-refresh", exclusive=False)

    @on(Button.Pressed, ".mcp-control")
    def control_server(self, event: Button.Pressed) -> None:
        event.stop()
        action = str(event.button.label).lower()
        name = event.button.name or ""
        self.run_worker(self._apply_control(name, action), name=f"mcp-{action}-{safe_id(name)}", exclusive=False)

    @on(Button.Pressed, "#mcp-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    async def _apply_control(self, name: str, action: str) -> None:
        if action == "enable":
            await self.manager.set_server_enabled(name, True)
        elif action == "disable":
            await self.manager.set_server_enabled(name, False)
        elif action == "restart":
            await self.manager.restart_server(name)
        await self.refresh_from_manager()

    async def _discover_and_refresh(self, name: str) -> None:
        await self.manager.discover_prompts(name)
        await self.manager.discover_resources(name)
        await self.refresh_from_manager()

    async def refresh_from_manager(self) -> None:
        if self.is_mounted:
            await self.recompose()

    def _tick(self) -> None:
        if not any(state.transient for state in self.manager.servers.values()):
            return
        self._spinner += 1
        self.run_worker(self.refresh_from_manager(), name="mcp-panel-spinner", exclusive=True)

    def action_close(self) -> None:
        self.dismiss()


def capability_counts(state: Any) -> str:
    tools = len(state.tools)
    prompts = "?" if state.prompts is None else str(len(state.prompts))
    resources = "?" if state.resources is None else str(len(state.resources))
    return f"{tools} tools · {prompts} prompts · {resources} resources"


def controls_for(status: str) -> tuple[str, ...]:
    if status == "Disabled":
        return ("Enable",)
    return ("Disable", "Restart")


def status_class(status: str) -> str:
    return {
        "Available": "available",
        "Partially available": "warning",
        "Approval required": "warning",
        "Failed": "failed",
        "Disabled": "disabled",
    }.get(status, "transient")


def server_details(state: Any) -> str:
    lines = ["Tools"]
    lines.extend(f"  {item.get('original_name') or item.get('name')}" for item in state.tool_metadata)
    if not state.tool_metadata:
        lines.append("  No tools")
    lines.append("Prompts")
    lines.extend(f"  {item.command}    {item.description}" for item in (state.prompts or []))
    if state.prompts == []:
        lines.append("  No prompts")
    elif state.prompts is None:
        lines.append("  Not discovered")
    lines.append("Fixed resources")
    lines.extend(f"  {item.uri}    {item.description or item.name}" for item in (state.resources or []))
    if state.resources == []:
        lines.append("  No resources")
    elif state.resources is None:
        lines.append("  Not discovered")
    errors = [state.error, state.prompt_error, state.resource_error]
    if any(errors):
        lines.extend(("Errors", *(f"  {error}" for error in errors if error)))
    return "\n".join(lines)


def safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "-" for character in value)


__all__ = ["MCPPanelScreen", "capability_counts", "controls_for", "status_class"]
