"""Single-owner MCP lifecycle and capability registries."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt.tool_node import ToolRuntime

from agent.mcp.configuration import (
    MCPConfiguration,
    approval_preview,
    load_mcp_configuration,
)
from agent.mcp.errors import sanitized_error
from agent.mcp.integration import MiraMCPIntegration
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
from core.diagnostics.logging import get_diagnostics_logger

ApprovalHandler = Callable[[MCPServerState, str], Awaitable[str]]
ChangeHandler = Callable[[], Awaitable[None]]
ActivityHandler = Callable[[str, dict[str, Any]], None]
_ATTACHMENT_PATTERN = re.compile(r'(?<![\w@])@(?:"([^"\r\n]+)"|([^\s@]+))')
_MAX_STARTUP_ERROR = 800


@dataclass(slots=True)
class _LifecycleBatch:
    batch_id: str
    kind: str
    started_at: float
    server_names: tuple[str, ...]
    enabled: set[str]
    active_server: str = ""
    completed: set[str] = field(default_factory=set)
    finished: bool = False


class MCPManager:
    """Own every configured server runtime and all MCP-facing registries."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
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
        self._activity_handler: ActivityHandler | None = None
        self._activity_batch: _LifecycleBatch | None = None
        self._batch_sequence = 0
        self._operation_lock = asyncio.Lock()
        self._runtimes: dict[str, MCPServerRuntime] = {}
        self.read_tool = self._build_read_tool()

    @property
    def issues(self) -> list[Any]:
        return list(self.configuration.issues)

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

    def set_activity_handler(self, handler: ActivityHandler | None) -> None:
        """Install the runtime-only projection for MCP lifecycle batches."""
        self._activity_handler = handler

    async def initialize(self, approval_handler: ApprovalHandler | None = None) -> None:
        """Start every independently valid, enabled, and approved server once."""
        self._approval_handler = approval_handler or self._approval_handler
        settings = load_settings(self.workspace)
        batch = self._begin_batch("startup", settings)
        try:
            await self._initialize_servers(settings, batch)
        except BaseException:
            self._finish_batch(batch, failed=True)
            raise
        self._finish_batch(batch)

    async def _initialize_servers(
        self,
        settings: dict[str, Any],
        batch: _LifecycleBatch,
    ) -> None:
        """Run the existing sequential startup loop within one UI batch."""
        for state in self.servers.values():
            if state.status == "Failed" and not state.config.get("transport"):
                batch.completed.add(state.name)
                self._emit_batch("initializing", batch)
                continue
            if state.error and state.status == "Failed":
                batch.completed.add(state.name)
                self._emit_batch("initializing", batch)
                continue
            if not mcp_server_enabled(settings, state.name):
                state.status = "Disabled"
                batch.completed.add(state.name)
                self._emit_batch("initializing", batch)
                continue
            batch.active_server = state.name
            self._emit_batch("initializing", batch)
            await self._start_server(state, restarting=False)
            batch.completed.add(state.name)
            batch.active_server = ""
            self._emit_batch("initializing", batch)

    async def reload(self) -> None:
        """Cleanly replace configuration, runtimes, and capability caches."""
        await _complete_lifecycle(self._reload(), name="mcp-reload")

    async def _reload(self) -> None:
        async with self._operation_lock:
            settings = load_settings(self.workspace)
            batch = self._begin_batch("reload", settings)
            try:
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
                self._runtimes = {}
                settings = load_settings(self.workspace)
                self._reset_batch_servers(batch, settings)
                await self._initialize_servers(settings, batch)
            except BaseException:
                self._finish_batch(batch, failed=True)
                raise
            self._finish_batch(batch)

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
        batch = self._begin_batch("restart", load_settings(self.workspace), (server_name,))
        async with self._operation_lock:
            try:
                batch.active_server = server_name
                self._emit_batch("initializing", batch)
                await self._stop_server(state, final_status="Restarting")
                await self._start_server(state, restarting=True, force_approval=request_approval_again)
                batch.completed.add(server_name)
                batch.active_server = ""
            except BaseException:
                self._finish_batch(batch, failed=True)
                raise
            self._finish_batch(batch)
        await self._notify_changed()
        return state.usable

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
        state.latest_stderr_line = ""
        state.stderr_tail = ""
        state.startup_started_at = monotonic()
        self._emit_active_batch()
        settings = load_settings(self.workspace)
        if not await self._approved(state, settings, force=force_approval):
            state.status = "Approval required"
            return
        try:
            state.session = await runtime.start()
            await self._discover_tools(state)
            await self._discover_server_prompts(state)
            await self._discover_server_resources(state)
            state.status = "Partially available" if state.error else "Available"
            self._emit_active_batch()
        except BaseException as error:
            await self._stop_runtime(runtime, state.name)
            state.session = None
            state.status = "Failed"
            state.error = _startup_error(error, state.stderr_tail)
            self._emit_active_batch()
            get_diagnostics_logger().error(
                "MCP server %s failed to start: %s",
                state.name,
                state.error,
            )

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
            descriptors = await state.session.list_tool_descriptors()
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
                converted = await state.session.as_tool(descriptor)
                converted.name = generated
                metadata = dict(getattr(converted, "metadata", None) or {})
                metadata["mira_mcp"] = {"server": state.name, "tool": original}
                converted.metadata = metadata
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

    async def _discover_server_prompts(self, state: MCPServerState) -> None:
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
                descriptors = await session.list_prompts()
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
                        return list(
                            await _state.session.get_prompt_messages(
                                _name,
                                values or None,
                            )
                        )

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
                state.prompts = []
                state.prompt_error = _concise_error(error)
                self._mark_partial(state, f"prompts: {state.prompt_error}")

    async def discover_resources(self, server_name: str | None = None) -> None:
        states = self._discovery_states(server_name)
        await asyncio.gather(*(self._discover_server_resources(state) for state in states))

    async def _discover_server_resources(self, state: MCPServerState) -> None:
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
                descriptors = await session.list_resources()
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
                        mime_type=str(
                            getattr(descriptor, "mime_type", None)
                            or getattr(descriptor, "mimeType", "")
                            or ""
                        ),
                    )
                    resources.append(resource)
                    self.resource_registry[token] = resource
                state.resources = resources
                state.resource_error = ""
            except BaseException as error:
                state.resources = []
                state.resource_error = _concise_error(error)
                self._mark_partial(state, f"resources: {state.resource_error}")

    def tools_for_mode(self, settings: dict[str, Any] | None, *, planning: bool) -> tuple[list[Any], list[dict[str, str]]]:
        tools: list[Any] = [self.read_tool]
        metadata: list[dict[str, str]] = []
        for state in self.servers.values():
            if not state.usable or not mcp_server_enabled(settings, state.name):
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
                contents = await state.session.read_resource(uri)
            except BaseException as error:
                raise ToolException(f"MCP resource read failed: {_concise_error(error)}") from error
            parts: list[str] = []
            for content in contents:
                data = getattr(content, "text", None)
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

    def _runtime_for(self, state: MCPServerState) -> MCPServerRuntime:
        runtime = self._runtimes.get(state.name)
        if runtime is None:
            runtime = MCPServerRuntime(state.name, lambda: self._new_integration(state))
            self._runtimes[state.name] = runtime
        return runtime

    async def _stop_runtime(self, runtime: MCPServerRuntime, server_name: str) -> None:
        try:
            await runtime.stop()
        except BaseException:
            get_diagnostics_logger().exception("MCP session cleanup failed for %s", server_name)

    def _new_integration(self, state: MCPServerState) -> MiraMCPIntegration:
        return MiraMCPIntegration(state, self._stderr_changed)

    def _server(self, name: str) -> MCPServerState:
        if name not in self.servers:
            raise KeyError(f"unknown MCP server: {name}")
        return self.servers[name]

    async def _notify_changed(self) -> None:
        if self._change_handler is not None:
            await self._change_handler()

    def _stderr_changed(self, state: MCPServerState) -> None:
        batch = self._activity_batch
        if batch is not None and not batch.finished and batch.active_server == state.name:
            self._emit_batch("initializing", batch)

    def _begin_batch(
        self,
        kind: str,
        settings: dict[str, Any],
        server_names: tuple[str, ...] | None = None,
    ) -> _LifecycleBatch:
        self._batch_sequence += 1
        names = server_names or tuple(self.servers)
        batch = _LifecycleBatch(
            batch_id=f"mcp-{self._batch_sequence}",
            kind=kind,
            started_at=monotonic(),
            server_names=tuple(names),
            enabled={
                name
                for name in names
                if name in self.servers and mcp_server_enabled(settings, name)
            },
        )
        self._activity_batch = batch
        self._emit_batch("initializing", batch)
        return batch

    def _reset_batch_servers(self, batch: _LifecycleBatch, settings: dict[str, Any]) -> None:
        batch.server_names = tuple(self.servers)
        batch.enabled = {
            name for name in batch.server_names if mcp_server_enabled(settings, name)
        }
        batch.active_server = ""
        batch.completed.clear()
        self._emit_batch("initializing", batch)

    def _finish_batch(self, batch: _LifecycleBatch, *, failed: bool = False) -> None:
        if batch.finished:
            return
        batch.active_server = ""
        batch.finished = True
        self._emit_batch("error" if failed else "initialized", batch)
        if self._activity_batch is batch:
            self._activity_batch = None

    def _emit_active_batch(self) -> None:
        batch = self._activity_batch
        if batch is not None and not batch.finished:
            self._emit_batch("initializing", batch)

    def _emit_batch(self, phase: str, batch: _LifecycleBatch) -> None:
        if self._activity_handler is None:
            return
        try:
            self._activity_handler(phase, self._batch_snapshot(batch))
        except Exception:
            get_diagnostics_logger().exception("MCP activity projection failed")

    def _batch_snapshot(self, batch: _LifecycleBatch) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for name in batch.server_names:
            state = self.servers.get(name)
            enabled = name in batch.enabled
            if not enabled:
                stage = "disabled"
            elif name == batch.active_server:
                stage = "active"
            elif state is not None and state.status == "Failed" and state.error:
                stage = "complete"
            elif name in batch.completed or batch.finished:
                stage = "complete"
            else:
                stage = "waiting"
            rows.append(
                {
                    "name": name,
                    "stage": stage,
                    "status": state.status if state is not None else "Disabled",
                    "error": state.error if state is not None else "",
                    "latest_stderr_line": state.latest_stderr_line if state is not None else "",
                    "started_at": state.startup_started_at if state is not None else None,
                    "tool_count": len(state.tools) if state is not None else 0,
                    "prompt_count": len(state.prompts or ()) if state is not None else 0,
                    "resource_count": len(state.resources or ()) if state is not None else 0,
                }
            )
        ready = sum(
            row["stage"] == "complete" and row["status"] == "Available" for row in rows
        )
        partial = sum(
            row["stage"] == "complete" and row["status"] == "Partially available"
            for row in rows
        )
        failed = sum(
            row["stage"] == "complete" and row["status"] == "Failed" for row in rows
        )
        approval = sum(
            row["stage"] == "complete" and row["status"] == "Approval required"
            for row in rows
        )
        disabled = sum(row["stage"] == "disabled" for row in rows)
        waiting = sum(row["stage"] == "waiting" for row in rows)
        tools = sum(len(state.tools) for state in self.servers.values())
        prompts = sum(len(state.prompts or ()) for state in self.servers.values())
        resources = sum(len(state.resources or ()) for state in self.servers.values())
        attempted = len(rows) - disabled
        successful = bool(batch.finished and attempted and ready == attempted)
        return {
            "batch_id": batch.batch_id,
            "kind": batch.kind,
            "started_at": batch.started_at,
            "finished": batch.finished,
            "successful": successful,
            "active_server": batch.active_server,
            "servers": tuple(rows),
            "counts": {
                "ready": ready,
                "partial": partial,
                "failed": failed,
                "approval": approval,
                "disabled": disabled,
                "waiting": waiting,
                "servers": attempted,
                "usable": self.usable_count,
                "configured": self.configured_count,
                "tools": tools,
                "prompts": prompts,
                "resources": resources,
            },
        }


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


def _startup_error(error: BaseException, stderr_tail: str) -> str:
    transport = sanitized_error(error)
    if not _vague_transport_error(transport):
        return transport
    stderr_lines = [line.strip() for line in stderr_tail.splitlines() if line.strip()]
    if not stderr_lines:
        return transport
    cause = stderr_lines[-1]
    if cause.casefold() in transport.casefold() or transport.casefold() in cause.casefold():
        return cause[:_MAX_STARTUP_ERROR]
    return f"{cause} (transport: {transport})"[:_MAX_STARTUP_ERROR]


def _vague_transport_error(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return normalized in {
        "mcperror: connection closed",
        "runtimeerror: client failed to connect: connection closed",
        "connectionerror: connection closed",
    } or normalized.endswith(": connection closed")


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
