"""Single-owner MCP lifecycle and capability registries."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.resources import get_mcp_resource
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langgraph.prebuilt.tool_node import ToolRuntime

from agent.mcp.configuration import (
    MCPConfiguration,
    adapter_connection,
    approval_preview,
    load_mcp_configuration,
)
from agent.mcp.auth import (
    TOKEN_ROOT,
    MiraOAuthProvider,
    OAuthLoginRequired,
    forget_persisted_login,
    has_authorization_header,
    has_persisted_login,
    is_known_oauth_failure,
    is_oauth_login_required,
    sanitized_error,
)
from agent.mcp.client import MiraMCPClient
from agent.mcp.models import MCPResource, MCPServerState, PromptArgument, PromptSpec
from agent.mcp.prompts import PromptRegistry
from agent.mcp.runtime import MCPServerRuntime
from config.settings import (
    load_settings,
    mcp_server_always_allow,
    mcp_server_approved_fingerprint,
    mcp_server_enabled,
    mcp_tool_policy,
    save_settings,
    set_mcp_server_always_allow,
    set_mcp_server_approved_fingerprint,
    set_mcp_server_enabled,
)
from runtime.diagnostics import get_diagnostics_logger

ApprovalHandler = Callable[[MCPServerState, str], Awaitable[str]]
ChangeHandler = Callable[[], Awaitable[None]]
_ATTACHMENT_PATTERN = re.compile(r'(?<![\w@])@(?:"([^"\r\n]+)"|([^\s@]+))')
_MAX_PAGES = 1000


class MCPManager:
    """Own every configured server runtime and all MCP-facing registries."""

    def __init__(self, workspace: Path, *, token_root: Path | None = None) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.token_root = (token_root or TOKEN_ROOT).expanduser()
        self.configuration: MCPConfiguration = load_mcp_configuration(self.workspace)
        self.servers: dict[str, MCPServerState] = self.configuration.servers
        self.prompt_registry = PromptRegistry(self.workspace)
        self.resource_registry: dict[str, MCPResource] = {}
        self._tool_owners: dict[str, str] = {}
        self._ambiguous_tools: set[str] = set()
        self._session_approvals: set[tuple[str, str]] = set()
        self._prompt_locks: dict[str, asyncio.Lock] = {}
        self._resource_locks: dict[str, asyncio.Lock] = {}
        self._approval_handler: ApprovalHandler | None = None
        self._change_handler: ChangeHandler | None = None
        self._operation_lock = asyncio.Lock()
        self._oauth_providers: dict[str, MiraOAuthProvider] = {}
        self._runtimes: dict[str, MCPServerRuntime] = {}
        self.client = self._new_client()
        self.read_tool = self._build_read_tool()

    @property
    def config_issue(self) -> Any | None:
        return self.configuration.issue

    @property
    def configured_count(self) -> int:
        return len(self.servers) if self.configuration.valid else 0

    @property
    def usable_count(self) -> int:
        return sum(state.usable for state in self.servers.values())

    @property
    def show_status(self) -> bool:
        return self.configuration.valid and bool(self.servers)

    def set_change_handler(self, handler: ChangeHandler | None) -> None:
        self._change_handler = handler

    async def initialize(self, approval_handler: ApprovalHandler | None = None) -> None:
        """Start every independently valid, enabled, and approved server once."""
        self._approval_handler = approval_handler or self._approval_handler
        settings = load_settings(self.workspace)
        for state in self.servers.values():
            if state.status == "Failed" and not state.config.get("transport"):
                continue
            if state.error and state.status == "Failed":
                continue
            if not mcp_server_enabled(settings, state.name):
                state.status = "Disabled"
                continue
            await self._start_server(state, restarting=False)

    async def reload(self) -> None:
        """Cleanly replace configuration, runtimes, and capability caches."""
        await _complete_lifecycle(self._reload(), name="mcp-reload")

    async def _reload(self) -> None:
        async with self._operation_lock:
            await self._shutdown_unlocked()
            self.configuration = load_mcp_configuration(self.workspace)
            self.servers = self.configuration.servers
            self.prompt_registry.reload_local()
            self.prompt_registry.mcp = {}
            self.resource_registry = {}
            self._tool_owners = {}
            self._ambiguous_tools = set()
            self._prompt_locks = {}
            self._resource_locks = {}
            self._oauth_providers = {}
            self._runtimes = {}
            self.client = self._new_client()
            await self.initialize(self._approval_handler)

    async def shutdown(self) -> None:
        await _complete_lifecycle(self._shutdown(), name="mcp-shutdown")

    async def _shutdown(self) -> None:
        async with self._operation_lock:
            await self._shutdown_unlocked()

    async def _shutdown_unlocked(self) -> None:
        for state in list(self.servers.values()):
            try:
                await self._stop_server(state, final_status="Disabled")
            except BaseException:
                get_diagnostics_logger().exception("MCP cleanup failed for %s", state.name)
        for runtime in list(self._runtimes.values()):
            try:
                await runtime.shutdown()
            except BaseException:
                get_diagnostics_logger().exception("MCP runtime shutdown failed for %s", runtime.name)
        self._runtimes = {}

    async def set_server_enabled(self, server_name: str, enabled: bool) -> bool:
        """Persist and apply a server enable transition through the sole pathway."""
        return await _complete_lifecycle(
            self._set_server_enabled(server_name, enabled),
            name=f"mcp-enable-{server_name}",
        )

    async def _set_server_enabled(self, server_name: str, enabled: bool) -> bool:
        state = self._server(server_name)
        settings = load_settings(self.workspace)
        updated = set_mcp_server_enabled(settings, server_name, enabled)
        if not save_settings(self.workspace, updated):
            return False
        async with self._operation_lock:
            if enabled:
                await self._start_server(state, restarting=False)
            else:
                await self._stop_server(state, final_status="Disabled")
        await self._notify_changed()
        return True

    async def restart_server(self, server_name: str) -> bool:
        """Restart one server without modifying its persisted enable value."""
        return await _complete_lifecycle(
            self._restart_server(server_name),
            name=f"mcp-restart-{server_name}",
        )

    async def _restart_server(self, server_name: str) -> bool:
        state = self._server(server_name)
        if not mcp_server_enabled(load_settings(self.workspace), server_name):
            state.status = "Disabled"
            await self._notify_changed()
            return False
        request_approval_again = state.status == "Approval required"
        async with self._operation_lock:
            await self._stop_server(state, final_status="Restarting")
            await self._start_server(state, restarting=True, force_approval=request_approval_again)
        await self._notify_changed()
        return state.usable

    def has_persisted_login(self, server_name: str) -> bool:
        state = self._server(server_name)
        return state.transport == "http" and has_persisted_login(state, token_root=self.token_root)

    async def login_server(self, server_name: str) -> bool:
        """Run browser OAuth only for this explicit interactive panel action."""
        return await _complete_lifecycle(
            self._login_server(server_name),
            name=f"mcp-login-{server_name}",
        )

    async def _login_server(self, server_name: str) -> bool:
        state = self._server(server_name)
        if state.transport != "http" or state.status != "Login required":
            return False
        async with self._operation_lock:
            state.status = "Authenticating"
            state.error = ""
            provider = MiraOAuthProvider(state, interactive=True, token_root=self.token_root)
            self._set_server_auth(state, provider)
            try:
                await self._start_server(state, restarting=False)
            except asyncio.CancelledError:
                state.status = "Login required"
                state.error = "Browser authorization was cancelled."
                raise
            if not state.usable:
                state.status = "Login required"
                if not state.error:
                    state.error = "Browser authorization did not complete."
        await self._notify_changed()
        return state.usable

    async def forget_server_login(self, server_name: str) -> bool:
        """Stop one server and delete only its user-level OAuth state."""
        return await _complete_lifecycle(
            self._forget_server_login(server_name),
            name=f"mcp-forget-login-{server_name}",
        )

    async def _forget_server_login(self, server_name: str) -> bool:
        state = self._server(server_name)
        if state.transport != "http":
            return False
        async with self._operation_lock:
            enabled = mcp_server_enabled(load_settings(self.workspace), server_name)
            await self._stop_server(state, final_status="Disabled")
            await forget_persisted_login(state, token_root=self.token_root)
            self._oauth_providers.pop(state.name, None)
            self._set_server_auth(state, None)
            if enabled:
                state.status = "Login required"
                state.error = "Login required."
        await self._notify_changed()
        return True

    async def set_server_always_allow(self, server_name: str, value: bool) -> bool:
        """Persist launch/use approval for the server's current fingerprint."""
        state = self._server(server_name)
        settings = set_mcp_server_always_allow(load_settings(self.workspace), server_name, value)
        settings = set_mcp_server_approved_fingerprint(
            settings,
            server_name,
            state.fingerprint if value else "",
        )
        if not save_settings(self.workspace, settings):
            return False
        await self._notify_changed()
        return True

    async def _start_server(
        self,
        state: MCPServerState,
        *,
        restarting: bool,
        force_approval: bool = False,
    ) -> None:
        runtime = self._runtime_for(state)
        if runtime.open:
            return
        state.status = "Restarting" if restarting else "Starting"
        state.error = ""
        settings = load_settings(self.workspace)
        if not await self._approved(state, settings, force=force_approval):
            state.status = "Approval required"
            return
        if (
            not has_authorization_header(state)
            and self.has_persisted_login(state.name)
            and state.name not in self._oauth_providers
        ):
            self._set_server_auth(
                state,
                MiraOAuthProvider(state, interactive=False, token_root=self.token_root),
            )
        try:
            state.session = await runtime.start()
            await self._discover_tools(state)
            await self._discover_server_prompts(state, during_start=True)
            await self._discover_server_resources(state, during_start=True)
            state.status = "Partially available" if state.error else "Available"
        except BaseException as error:
            await self._stop_runtime(runtime, state.name)
            state.session = None
            known_oauth = state.name in self._oauth_providers and is_known_oauth_failure(error, state)
            login_required = isinstance(error, OAuthLoginRequired) or known_oauth
            if not login_required:
                login_required = await is_oauth_login_required(error, state)
            state.status = "Login required" if login_required else "Failed"
            state.error = "Login required." if login_required else _concise_error(error)
            logger = get_diagnostics_logger()
            if login_required:
                logger.warning("MCP server %s requires login", state.name)
            else:
                logger.error("MCP server %s failed to start: %s", state.name, state.error)

    async def _approved(self, state: MCPServerState, settings: dict[str, Any], *, force: bool) -> bool:
        key = (state.name, state.fingerprint)
        persisted = mcp_server_approved_fingerprint(settings, state.name)
        if not force and mcp_server_always_allow(settings, state.name) and persisted == state.fingerprint:
            return True
        if not force and key in self._session_approvals:
            return True
        if self._approval_handler is None:
            return False
        answer = await self._approval_handler(state, approval_preview(state))
        if answer == "always_allow":
            updated = set_mcp_server_always_allow(settings, state.name, True)
            updated = set_mcp_server_approved_fingerprint(updated, state.name, state.fingerprint)
            if save_settings(self.workspace, updated):
                self._session_approvals.add(key)
                return True
            return False
        if answer == "allow":
            self._session_approvals.add(key)
            return True
        return False

    async def _stop_server(self, state: MCPServerState, *, final_status: str) -> None:
        runtime = self._runtimes.get(state.name)
        if state.status != "Disabled" or (runtime is not None and runtime.open):
            state.status = "Stopping" if final_status != "Restarting" else "Restarting"
        self._remove_server_capabilities(state)
        state.session = None
        if runtime is not None and runtime.open:
            await self._stop_runtime(runtime, state.name)
        state.status = final_status  # type: ignore[assignment]

    async def _discover_tools(self, state: MCPServerState) -> None:
        state.tools = []
        state.tool_metadata = []
        if not _advertises_capability(state.session, "tools"):
            return
        try:
            descriptors = await _list_all(state.session, "list_tools", "tools")
        except BaseException as error:
            state.error = f"tools: {_concise_error(error)}"
            return
        for descriptor in descriptors:
            original = str(getattr(descriptor, "name", "") or "")
            generated = f"mcp__{state.name}__{original}"
            if generated in self._ambiguous_tools:
                state.error = _join_error(state.error, f"ambiguous tool name skipped: {generated}")
                continue
            owner = self._tool_owners.get(generated)
            if owner is not None:
                previous = self.servers[owner]
                previous.tools = [tool for tool in previous.tools if getattr(tool, "name", "") != generated]
                previous.tool_metadata = [item for item in previous.tool_metadata if item.get("name") != generated]
                previous.error = _join_error(previous.error, f"ambiguous tool name skipped: {generated}")
                if previous.usable:
                    previous.status = "Partially available"
                state.error = _join_error(state.error, f"ambiguous tool name skipped: {generated}")
                self._tool_owners.pop(generated, None)
                self._ambiguous_tools.add(generated)
                continue
            try:
                converted = convert_mcp_tool_to_langchain_tool(
                    state.session,
                    descriptor,
                    server_name=state.name,
                    callbacks=self.client.callbacks,
                )
                converted.name = generated
                metadata = dict(getattr(converted, "metadata", None) or {})
                metadata["mira_mcp"] = {"server": state.name, "tool": original}
                converted.metadata = metadata
                converted = self._observe_tool_auth(state, converted)
            except BaseException as error:
                state.error = _join_error(state.error, f"tool {original}: {_concise_error(error)}")
                continue
            self._tool_owners[generated] = state.name
            state.tools.append(converted)
            state.tool_metadata.append(
                {
                    "name": generated,
                    "original_name": original,
                    "server": state.name,
                    "source": "mcp",
                    "path": state.name,
                    "replaces": "",
                    "runtime": "MCP",
                    "environment": state.transport,
                    "description": str(getattr(converted, "description", "") or ""),
                }
            )

    async def discover_prompts(self, server_name: str | None = None) -> None:
        states = self._discovery_states(server_name)
        await asyncio.gather(*(self._discover_server_prompts(state) for state in states))

    async def _discover_server_prompts(self, state: MCPServerState, *, during_start: bool = False) -> None:
        lock = self._prompt_locks.setdefault(state.name, asyncio.Lock())
        async with lock:
            if state.prompts is not None or state.session is None:
                return
            if not _advertises_capability(state.session, "prompts"):
                state.prompts = []
                state.prompt_error = ""
                self.prompt_registry.replace_server(state.name, [])
                return
            try:
                session = state.session
                descriptors = await _list_all(session, "list_prompts", "prompts")
                if state.session is not session:
                    return
                specs: list[PromptSpec] = []
                for descriptor in descriptors:
                    original = str(getattr(descriptor, "name", "") or "")
                    arguments = tuple(
                        PromptArgument(str(argument.name), bool(getattr(argument, "required", False)))
                        for argument in (getattr(descriptor, "arguments", None) or [])
                    )

                    async def resolve(
                        values: dict[str, str],
                        *,
                        _state: MCPServerState = state,
                        _name: str = original,
                    ) -> list[BaseMessage]:
                        try:
                            return list(await load_mcp_prompt(_state.session, _name, arguments=values or None))
                        except BaseException as error:
                            if await self._handle_later_auth_failure(_state, error):
                                raise RuntimeError("MCP login required.") from error
                            raise

                    specs.append(
                        PromptSpec(
                            command=f"/mcp__{state.name}__{original}",
                            description=str(getattr(descriptor, "description", "") or "MCP prompt"),
                            arguments=arguments,
                            source="mcp",
                            resolver=resolve,
                            server=state.name,
                        )
                    )
                state.prompts = specs
                state.prompt_error = ""
                self.prompt_registry.replace_server(state.name, specs)
            except BaseException as error:
                if during_start and state.name in self._oauth_providers and is_known_oauth_failure(error, state):
                    raise
                if await self._handle_later_auth_failure(state, error):
                    return
                state.prompts = []
                state.prompt_error = _concise_error(error)
                self._mark_partial(state, f"prompts: {state.prompt_error}")

    async def discover_resources(self, server_name: str | None = None) -> None:
        states = self._discovery_states(server_name)
        await asyncio.gather(*(self._discover_server_resources(state) for state in states))

    async def _discover_server_resources(self, state: MCPServerState, *, during_start: bool = False) -> None:
        lock = self._resource_locks.setdefault(state.name, asyncio.Lock())
        async with lock:
            if state.resources is not None or state.session is None:
                return
            if not _advertises_capability(state.session, "resources"):
                state.resources = []
                state.resource_error = ""
                return
            try:
                session = state.session
                descriptors = await _list_all(session, "list_resources", "resources")
                if state.session is not session:
                    return
                resources = []
                for descriptor in descriptors:
                    uri = str(getattr(descriptor, "uri", "") or "")
                    token = f"mcp__{state.name}__{uri}"
                    resource = MCPResource(
                        token=token,
                        server=state.name,
                        uri=uri,
                        name=str(getattr(descriptor, "title", None) or getattr(descriptor, "name", "") or ""),
                        description=str(getattr(descriptor, "description", "") or ""),
                        mime_type=str(getattr(descriptor, "mimeType", "") or ""),
                    )
                    resources.append(resource)
                    self.resource_registry[token] = resource
                state.resources = resources
                state.resource_error = ""
            except BaseException as error:
                if during_start and state.name in self._oauth_providers and is_known_oauth_failure(error, state):
                    raise
                if await self._handle_later_auth_failure(state, error):
                    return
                state.resources = []
                state.resource_error = _concise_error(error)
                self._mark_partial(state, f"resources: {state.resource_error}")

    def tools_for_mode(self, settings: dict[str, Any] | None, *, planning: bool) -> tuple[list[Any], list[dict[str, str]]]:
        tools: list[Any] = [self.read_tool]
        metadata: list[dict[str, str]] = []
        for state in self.servers.values():
            if not state.usable:
                continue
            for item, tool_value in zip(state.tool_metadata, state.tools, strict=True):
                policy = mcp_tool_policy(settings, state.name, item["original_name"])
                if not policy.enabled or (planning and not policy.plan_access):
                    continue
                tools.append(tool_value)
                metadata.append(item)
        return tools, metadata

    def attachments_from_text(self, text: str) -> list[dict[str, str]]:
        attachments: list[dict[str, str]] = []
        seen: set[str] = set()
        for quoted, plain in _ATTACHMENT_PATTERN.findall(text):
            token = quoted or plain.rstrip(".,;:!?)]}")
            resource = self.resource_registry.get(token)
            if resource is not None and token not in seen:
                attachments.append(resource.attachment())
                seen.add(token)
        return attachments

    def resource_errors(self) -> list[str]:
        return [f"{state.name}: {state.resource_error}" for state in self.servers.values() if state.resource_error]

    def _build_read_tool(self) -> Any:
        manager = self

        @tool("read_mcp_resource")
        async def read_mcp_resource(server: str, uri: str, runtime: ToolRuntime) -> str:
            """Read one exact text MCP resource explicitly attached by the user."""
            allowed = _attached_pairs(runtime.state.get("messages", []) if isinstance(runtime.state, dict) else [])
            if (server, uri) not in allowed:
                raise ToolException("MCP resource was not attached by the user.")
            state = manager.servers.get(server)
            if state is None or state.session is None or not state.usable:
                raise ToolException("MCP server is not available.")
            try:
                blobs = await get_mcp_resource(state.session, uri)
            except BaseException as error:
                if await manager._handle_later_auth_failure(state, error):
                    raise ToolException("MCP login required.") from error
                raise ToolException(f"MCP resource read failed: {_concise_error(error)}") from error
            parts: list[str] = []
            for blob in blobs:
                data = getattr(blob, "data", None)
                if not isinstance(data, str):
                    raise ToolException("MCP resource content is binary or unsupported.")
                parts.append(data)
            return "\n\n".join(parts)

        read_mcp_resource.metadata = {"mira_trusted": True}
        return read_mcp_resource

    def _remove_server_capabilities(self, state: MCPServerState) -> None:
        for item in state.tool_metadata:
            self._tool_owners.pop(item.get("name", ""), None)
        state.tools = []
        self.prompt_registry.remove_server(state.name)
        for token in [token for token, resource in self.resource_registry.items() if resource.server == state.name]:
            self.resource_registry.pop(token, None)
        state.prompts = None
        state.resources = None
        state.prompt_error = ""
        state.resource_error = ""

    def _discovery_states(self, server_name: str | None) -> list[MCPServerState]:
        if server_name is not None:
            state = self.servers.get(server_name)
            return [state] if state is not None and state.usable else []
        return [state for state in self.servers.values() if state.usable]

    def _mark_partial(self, state: MCPServerState, error: str) -> None:
        state.error = _join_error(state.error, error)
        if state.usable:
            state.status = "Partially available"

    def _observe_tool_auth(self, state: MCPServerState, converted: Any) -> Any:
        if not isinstance(converted, StructuredTool) or converted.coroutine is None:
            return converted
        original = converted.coroutine
        manager = self

        async def observed(**arguments: Any) -> Any:
            try:
                return await original(**arguments)
            except BaseException as error:
                if await manager._handle_later_auth_failure(state, error):
                    raise ToolException("MCP login required.") from error
                raise

        return converted.model_copy(update={"coroutine": observed})

    async def _handle_later_auth_failure(self, state: MCPServerState, error: BaseException) -> bool:
        if state.name not in self._oauth_providers or not is_known_oauth_failure(error, state):
            return False
        async with self._operation_lock:
            if state.status == "Login required":
                return True
            self._remove_server_capabilities(state)
            state.session = None
            state.status = "Login required"
            state.error = "Login required. Stored credentials could not be refreshed."
            runtime = self._runtimes.get(state.name)
            if runtime is not None and runtime.open:
                await self._stop_runtime(runtime, state.name)
        await self._notify_changed()
        return True

    def _set_server_auth(self, state: MCPServerState, provider: MiraOAuthProvider | None) -> None:
        if provider is None:
            self._oauth_providers.pop(state.name, None)
        else:
            self._oauth_providers[state.name] = provider
        connections = getattr(self.client, "connections", None)
        if not isinstance(connections, dict) or state.name not in connections:
            return
        connection = adapter_connection(state)
        if provider is not None:
            connection["auth"] = provider
        connections[state.name] = connection

    def _runtime_for(self, state: MCPServerState) -> MCPServerRuntime:
        runtime = self._runtimes.get(state.name)
        if runtime is None:
            runtime = MCPServerRuntime(state.name, lambda: self.client.session(state.name))
            self._runtimes[state.name] = runtime
        return runtime

    async def _stop_runtime(self, runtime: MCPServerRuntime, server_name: str) -> None:
        try:
            await runtime.stop()
        except BaseException:
            get_diagnostics_logger().exception("MCP session cleanup failed for %s", server_name)

    def _new_client(self) -> MiraMCPClient:
        connections: dict[str, dict[str, Any]] = {}
        for name, state in self.servers.items():
            if state.config.get("transport") not in {"stdio", "http"}:
                continue
            connection = adapter_connection(state)
            if (
                state.transport == "http"
                and not has_authorization_header(state)
                and has_persisted_login(state, token_root=self.token_root)
            ):
                provider = MiraOAuthProvider(state, interactive=False, token_root=self.token_root)
                self._oauth_providers[name] = provider
                connection["auth"] = provider
            connections[name] = connection

        async def log_message(params: Any, context: Any) -> None:
            data = getattr(params, "data", "")
            level = str(getattr(params, "level", "info") or "info").lower()
            logger = get_diagnostics_logger()
            method = getattr(logger, level, logger.info)
            method("MCP %s: %s", context.server_name, data)

        return MiraMCPClient(connections, callbacks=Callbacks(on_logging_message=log_message))

    def _server(self, name: str) -> MCPServerState:
        if name not in self.servers:
            raise KeyError(f"unknown MCP server: {name}")
        return self.servers[name]

    async def _notify_changed(self) -> None:
        if self._change_handler is not None:
            await self._change_handler()


async def _list_all(session: Any, method_name: str, field: str) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        result = await getattr(session, method_name)(cursor=cursor)
        items.extend(getattr(result, field, None) or [])
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            return items
    raise RuntimeError(f"{method_name} exceeded {_MAX_PAGES} pages")


def _attached_pairs(messages: list[Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for message in messages:
        metadata = getattr(message, "additional_kwargs", None)
        attachments = metadata.get("mira_mcp_attachments", []) if isinstance(metadata, dict) else []
        for item in attachments:
            if isinstance(item, dict) and item.get("kind") == "mcp_resource":
                pairs.add((str(item.get("server") or ""), str(item.get("uri") or "")))
    return pairs


def _advertises_capability(session: Any, name: str) -> bool:
    """Treat missing SDK metadata as unknown, but honor explicit capability absence."""
    getter = getattr(session, "get_server_capabilities", None)
    if not callable(getter):
        return True
    capabilities = getter()
    return capabilities is None or getattr(capabilities, name, None) is not None


def _join_error(current: str, addition: str) -> str:
    return "; ".join(value for value in (current, addition) if value)


def _concise_error(error: BaseException) -> str:
    return sanitized_error(error)


async def _complete_lifecycle(awaitable: Awaitable[Any], *, name: str) -> Any:
    """Let a lifecycle transition finish even if its requesting UI worker exits."""
    task = asyncio.create_task(awaitable, name=name)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


__all__ = ["MCPManager"]
