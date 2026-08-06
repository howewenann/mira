"""Expandable MCP server status and lifecycle controls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ui.spinners import SPINNER_FRAMES


class MCPPanelScreen(ModalScreen[None]):
    """Render every configured server as a structured status card."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, manager: Any, reload_runtime: Callable[[], Awaitable[None]]) -> None:
        super().__init__()
        self.manager = manager
        self.reload_runtime = reload_runtime
        self.expanded: set[str] = set()
        self.reloading = False
        self._spinner = 0
        self._refresh_lock = asyncio.Lock()
        self._presentation_signature = self._server_presentation_signature()

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-dialog"):
            with Horizontal(id="mcp-title-row"):
                yield Static("MCP SERVERS", id="mcp-title")
                title_close = Button("x", id="mcp-title-close", classes="mcp-close panel-close")
                title_close.disabled = self.reloading
                yield title_close
            with VerticalScroll(id="mcp-scroll"):
                for state in self.manager.servers.values():
                    yield from self._server_widgets(state)
            with Horizontal(id="mcp-actions"):
                reload_button = Button("Reload Runtime", id="mcp-reload", variant="primary")
                reload_button.disabled = self.reloading
                yield reload_button
                close_button = Button("Close", id="mcp-close", classes="mcp-close")
                close_button.disabled = self.reloading
                yield close_button

    def _server_widgets(self, state: Any) -> ComposeResult:
        expanded = state.name in self.expanded
        marker = "-" if expanded else "+"
        status = status_badge(state.status, spinner=self._spinner)
        with Vertical(
            id=f"mcp-card-{safe_id(state.name)}",
            classes=f"mcp-server-card {status_class(state.status)}",
        ):
            with Horizontal(classes="mcp-server-top"):
                yield Button(
                    Text(f"{marker} {state.name} [{state.transport.upper()}]"),
                    id=f"mcp-header-{safe_id(state.name)}",
                    classes=f"mcp-server-header {status_class(state.status)}",
                    name=state.name,
                )
                yield Static(
                    status,
                    classes=f"mcp-status-badge {status_class(state.status)}",
                    markup=False,
                )
            yield Static(capability_summary(state), classes="mcp-counts", markup=False)
            with Horizontal(classes="mcp-controls"):
                persisted_login = bool(
                    state.transport == "http"
                    and getattr(self.manager, "has_persisted_login", lambda _name: False)(state.name)
                )
                for action in controls_for(state.status, persisted_login=persisted_login):
                    button = Button(
                        action,
                        id=f"mcp-{safe_id(action.lower())}-{safe_id(state.name)}",
                        classes="mcp-control",
                        name=state.name,
                    )
                    button.disabled = state.transient
                    yield button
            if expanded:
                with Vertical(classes="mcp-details"):
                    for title, content, error in server_detail_sections(state):
                        yield Static(
                            f"{title}\n{content}",
                            classes="mcp-error-block" if error else "mcp-detail-section",
                            markup=False,
                        )

    def on_mount(self) -> None:
        self.set_interval(0.12, self._tick)
        self.call_after_refresh(self._focus_first_header)

    @on(Button.Pressed, ".mcp-server-header")
    def toggle_server(self, event: Button.Pressed) -> None:
        event.stop()
        name = event.button.name or ""
        if name in self.expanded:
            self.expanded.remove(name)
        else:
            self.expanded.add(name)
        self.run_worker(
            self.refresh_from_manager(preferred_focus_id=event.button.id),
            name="mcp-panel-refresh",
            exclusive=False,
        )

    @on(Button.Pressed, ".mcp-control")
    def control_server(self, event: Button.Pressed) -> None:
        event.stop()
        action = str(event.button.label).lower()
        name = event.button.name or ""
        self.run_worker(self._apply_control(name, action), name=f"mcp-{action}-{safe_id(name)}", exclusive=False)

    @on(Button.Pressed, "#mcp-reload")
    def reload_pressed(self, event: Button.Pressed) -> None:
        """Run the same full runtime reload used by /reload."""
        event.stop()
        if self.reloading:
            return
        self.reloading = True
        self._sync_reload_controls()
        self.run_worker(self._reload(), name="mcp-runtime-reload", exclusive=False)

    @on(Button.Pressed, ".mcp-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    async def _reload(self) -> None:
        """Reload the full runtime and redraw server state without closing the panel."""
        try:
            await self.reload_runtime()
        finally:
            self.reloading = False
            if self.is_mounted:
                await self.refresh_from_manager(preferred_focus_id="mcp-reload")

    def _sync_reload_controls(self) -> None:
        """Prevent reload or dismissal while runtime replacement is in progress."""
        for selector in ("#mcp-reload", "#mcp-close", "#mcp-title-close"):
            try:
                self.query_one(selector, Button).disabled = self.reloading
            except NoMatches:
                pass

    async def _apply_control(self, name: str, action: str) -> None:
        if action == "enable":
            await self.manager.set_server_enabled(name, True)
        elif action == "disable":
            await self.manager.set_server_enabled(name, False)
        elif action == "restart":
            await self.manager.restart_server(name)
        elif action == "login":
            await self.manager.login_server(name)
        elif action == "forget login":
            await self.manager.forget_server_login(name)
        await self.refresh_from_manager()

    async def refresh_from_manager(self, preferred_focus_id: str | None = None) -> None:
        async with self._refresh_lock:
            if not self.is_mounted:
                return
            self._presentation_signature = self._server_presentation_signature()
            focused_id = preferred_focus_id or (self.focused.id if self.focused is not None else None)
            try:
                scroll_y = self.query_one("#mcp-scroll", VerticalScroll).scroll_y
            except NoMatches:
                return
            await self.recompose()
            if not self.is_mounted:
                return
            restored = asyncio.get_running_loop().create_future()

            def restore_presentation() -> None:
                try:
                    restored_focus = False
                    if focused_id:
                        try:
                            self.query_one(f"#{focused_id}").focus(scroll_visible=False)
                            restored_focus = True
                        except NoMatches:
                            pass
                    if not restored_focus:
                        self._focus_first_header()
                    scroll = self.query_one("#mcp-scroll", VerticalScroll)
                    scroll.scroll_to(
                        y=scroll_y,
                        animate=False,
                        force=True,
                        immediate=True,
                    )
                except NoMatches:
                    pass
                finally:
                    if not restored.done():
                        restored.set_result(None)

            if self.call_after_refresh(restore_presentation):
                await restored

    def _focus_first_header(self) -> None:
        headers = list(self.query(".mcp-server-header"))
        if headers:
            headers[0].focus(scroll_visible=False)

    def _tick(self) -> None:
        presentation_signature = self._server_presentation_signature()
        if presentation_signature != self._presentation_signature:
            self._presentation_signature = presentation_signature
            self.run_worker(
                self.refresh_from_manager(),
                name="mcp-panel-state-refresh",
                exclusive=False,
            )
            return
        if not any(state.transient for state in self.manager.servers.values()):
            return
        self._spinner += 1
        for state in self.manager.servers.values():
            if not state.transient:
                continue
            server_id = safe_id(state.name)
            state_class = status_class(state.status)
            try:
                card = self.query_one(f"#mcp-card-{server_id}")
                header = self.query_one(f"#mcp-header-{server_id}", Button)
                badge = card.query_one(".mcp-status-badge", Static)
            except NoMatches:
                continue
            card.set_classes(f"mcp-server-card {state_class}")
            header.set_classes(f"mcp-server-header {state_class}")
            badge.set_classes(f"mcp-status-badge {state_class}")
            badge.update(status_badge(state.status, spinner=self._spinner))
            for control in card.query(".mcp-control"):
                control.disabled = True

    def _server_presentation_signature(self) -> tuple[Any, ...]:
        """Return the manager state that can change an already-open card."""
        return tuple(
            (
                state.name,
                state.transport,
                state.status,
                bool(state.transient),
                capability_counts(state),
                tuple(server_detail_sections(state)),
            )
            for state in self.manager.servers.values()
        )

    def action_close(self) -> None:
        if not self.reloading:
            self.dismiss()


def capability_counts(state: Any) -> tuple[tuple[str, str], ...]:
    prompts = "-" if state.prompts is None else str(len(state.prompts))
    resources = "-" if state.resources is None else str(len(state.resources))
    return (
        ("Tools", str(len(state.tools))),
        ("Prompts", prompts),
        ("Resources", resources),
    )


def capability_metric(label: str, count: str) -> Text:
    label_colour = {
        "Tools": "#78d5cf",
        "Prompts": "#8fb9e8",
        "Resources": "#c7a0e8",
    }[label]
    metric = Text()
    metric.append(label, style=label_colour)
    metric.append("  ")
    metric.append(count, style="bold #eef7f8")
    return metric


def capability_summary(state: Any) -> Text:
    summary = Text()
    for index, (label, count) in enumerate(capability_counts(state)):
        if index:
            summary.append("   ·   ", style="#6f8389")
        summary.append_text(capability_metric(label, count))
    return summary


def controls_for(status: str, *, persisted_login: bool = False) -> tuple[str, ...]:
    if status == "Disabled":
        return ("Enable", "Forget login") if persisted_login else ("Enable",)
    if status in {"Login required", "Authenticating"}:
        controls = ("Login", "Disable")
    else:
        controls = ("Restart", "Disable")
    return (*controls, "Forget login") if persisted_login else controls


def status_class(status: str) -> str:
    return {
        "Available": "available",
        "Partially available": "warning",
        "Approval required": "warning",
        "Login required": "warning",
        "Failed": "failed",
        "Disabled": "disabled",
    }.get(status, "transient")


def status_badge(status: str, *, spinner: int = 0) -> str:
    if status in {"Authenticating", "Starting", "Restarting", "Stopping"}:
        return f"{SPINNER_FRAMES[spinner % len(SPINNER_FRAMES)]} {status}"
    return "Partial" if status == "Partially available" else status


def mcp_summary_symbol(states: Iterable[Any], *, spinner: int = 0) -> str:
    statuses = [state.status for state in states]
    if any(status == "Failed" for status in statuses):
        return "x"
    if any(status in {"Partially available", "Approval required", "Login required"} for status in statuses):
        return "!"
    if any(status in {"Authenticating", "Starting", "Restarting", "Stopping"} for status in statuses):
        return SPINNER_FRAMES[spinner % len(SPINNER_FRAMES)]
    if statuses and all(status == "Disabled" for status in statuses):
        return "–"
    if any(status == "Disabled" for status in statuses):
        return "!"
    return "✓"


def server_detail_sections(state: Any) -> list[tuple[str, str, bool]]:
    tools = [f"  {item.get('original_name') or item.get('name')}" for item in state.tool_metadata]
    prompts = [f"  {item.command}    {item.description}" for item in (state.prompts or [])]
    resources = [f"  {item.uri}    {item.description or item.name}" for item in (state.resources or [])]
    sections = [
        ("Tools", "\n".join(tools) if tools else "  No tools", False),
        ("Prompts", _discovery_content(prompts, state.prompts, "prompts"), False),
        ("Fixed resources", _discovery_content(resources, state.resources, "resources"), False),
    ]
    errors = [error for error in (state.error, state.prompt_error, state.resource_error) if error]
    if errors:
        sections.append(("Needs attention", "\n".join(f"  {error}" for error in errors), True))
    return sections


def _discovery_content(lines: list[str], values: list[Any] | None, noun: str) -> str:
    if lines:
        return "\n".join(lines)
    if values is None:
        return "  Not loaded"
    return f"  No {noun}"


def safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "-" for character in value)


__all__ = [
    "MCPPanelScreen",
    "capability_counts",
    "controls_for",
    "mcp_summary_symbol",
    "status_badge",
    "status_class",
]
