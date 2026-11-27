"""Supported MCP operations with detached capability projections."""

from copy import deepcopy
from typing import Any

from core.interface.catalog import CommandInfo, MCPServerSnapshot, ResourceInfo


def prompt_info(spec: Any) -> CommandInfo:
    return CommandInfo(spec.command, spec.usage, spec.description, source=spec.source,
                       server=spec.server, arguments=deepcopy(spec.arguments))


class MiraMCP:
    """Manage workspace MCP servers without exposing clients or executable tools."""

    def __init__(self, application: Any) -> None:
        self._application = application

    @property
    def servers(self) -> dict[str, MCPServerSnapshot]:
        manager = self._application.mcp_manager
        if manager is None:
            return {}
        result = {}
        for state in manager.servers.values():
            skills = None if getattr(state, "skills", None) is None else tuple(ResourceInfo(
                skill.name, skill.description,
                tuple([skill.manifest.uri, *(item.uri for item in (skill.manifest.resources if isinstance(skill.manifest.resources, list) else ()))]),
            ) for skill in state.skills)
            result[state.name] = MCPServerSnapshot(
                state.name, state.transport, state.status, state.status in {"Available", "Partially available"}, state.status in {"Starting", "Restarting", "Stopping"},
                state.error, tuple(deepcopy(state.tool_metadata)),
                None if state.prompts is None else tuple(prompt_info(spec) for spec in state.prompts),
                None if state.resources is None else tuple(deepcopy(state.resources)), skills,
                state.prompt_error, state.resource_error, getattr(state, "skill_error", ""), getattr(state, "stderr_tail", ""),
            )
        return result

    @property
    def resources(self) -> tuple[Any, ...]:
        manager = self._application.mcp_manager
        return tuple(deepcopy(list(manager.resource_registry.values()))) if manager is not None else ()

    def settings(self) -> dict[str, Any]:
        return self._application.settings()

    @property
    def show_status(self) -> bool:
        return bool(self.configured_count)

    def commands(self) -> tuple[CommandInfo, ...]:
        return self._application.commands("prompt")

    @property
    def usable_count(self) -> int:
        manager = self._application.mcp_manager
        return sum(state.status in {"Available", "Partially available"}
                   for state in manager.servers.values()) if manager is not None else 0

    @property
    def configured_count(self) -> int:
        manager = self._application.mcp_manager
        return len(manager.servers) if manager is not None else 0

    def attachments_from_text(self, text: str) -> list[dict[str, str]]:
        manager = self._application.mcp_manager
        return deepcopy(manager.attachments_from_text(text)) if manager is not None else []

    def resource_errors(self) -> list[str]:
        manager = self._application.mcp_manager
        return list(manager.resource_errors()) if manager is not None else []

    async def discover_resources(self, name: str | None = None) -> None:
        if self._application.mcp_manager is not None:
            await self._application.mcp_manager.discover_resources(name)

    async def discover_prompts(self, name: str | None = None) -> None:
        if self._application.mcp_manager is not None:
            await self._application.mcp_manager.discover_prompts(name)

    async def set_server_enabled(self, name: str, enabled: bool) -> bool:
        with self._application._changing_runtime():
            return await self._application.mcp_manager.set_server_enabled(name, enabled)

    async def set_server_always_allow(self, name: str, allowed: bool) -> bool:
        with self._application._changing_runtime():
            return await self._application.mcp_manager.set_server_always_allow(name, allowed)

    async def restart_server(self, name: str) -> None:
        with self._application._changing_runtime():
            await self._application.mcp_manager.restart_server(name)
