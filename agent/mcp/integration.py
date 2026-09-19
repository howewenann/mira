"""Thin per-server integration with LangChain's first-party MCP stack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

with suppress_langchain_beta_warning():
    from langchain.mcp import MCPAdapter, as_langchain_tool

from agent.mcp.auth import create_http_oauth
from agent.mcp.models import MCPServerState
from agent.mcp.stderr import MCPStderrCapture, MCPStderrSnapshot
from core.diagnostics.logging import get_diagnostics_logger

_MAX_PAGES = 1000


class MiraMCPIntegration:
    """Connect one configured server through ``MCPAdapter`` and FastMCP."""

    def __init__(
        self,
        state: MCPServerState,
        stderr_handler: Callable[[MCPServerState], None] | None = None,
    ) -> None:
        config = state.connection_config
        self._state = state
        self._stderr_handler = stderr_handler
        self._stderr_capture: MCPStderrCapture | None = None
        if state.transport == "stdio":
            self._stderr_capture = MCPStderrCapture(
                self._stderr_updated,
                secret_values=_configured_secret_values(state),
            )
            transport = StdioTransport(
                command=str(config["command"]),
                args=list(config.get("args") or []),
                env=dict(config.get("env") or {}),
                keep_alive=False,
                # FastMCP passes this TextIO sink directly to the MCP SDK's
                # subprocess stderr handle. The capture drains it continuously.
                log_file=self._stderr_capture.sink,
            )
        else:
            transport = StreamableHttpTransport(
                url=str(config["url"]),
                headers=dict(config.get("headers") or {}),
            )

        async def log_message(message: Any) -> None:
            data = getattr(message, "data", "")
            level = str(getattr(message, "level", "info") or "info").lower()
            logger = get_diagnostics_logger()
            method = getattr(logger, level, logger.info)
            method("MCP %s: %s", state.name, data)

        # Constructing MCPAdapter around this client arms native LangGraph
        # interrupt elicitation while retaining MIRA's transport lifecycle.
        self._transport = transport
        self._client_options: dict[str, Any] = {
            "name": state.name,
            "log_handler": log_message,
        }
        self.adapter: MCPAdapter | None = None
        if state.transport == "stdio":
            self.adapter = MCPAdapter(Client(transport, **self._client_options))

    @property
    def client(self) -> Client[Any]:
        if self.adapter is None:
            raise RuntimeError("HTTP MCP integration has not started")
        client = self.adapter.client
        if not isinstance(client, Client):  # pragma: no cover - one-server invariant
            raise TypeError("MIRA MCP integration requires one FastMCP client")
        return client

    async def __aenter__(self) -> MiraMCPIntegration:
        if self.adapter is None:
            url = str(self._state.connection_config["url"])
            oauth = await create_http_oauth(url)
            self.adapter = MCPAdapter(
                Client(self._transport, auth=oauth, **self._client_options)
            )
        if self._stderr_capture is not None:
            self._stderr_capture.start()
        try:
            await self.adapter.__aenter__()
        except BaseException:
            if self._stderr_capture is not None:
                await self._stderr_capture.aclose()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.adapter is None:  # pragma: no cover - enter constructs HTTP adapter
            return
        try:
            await self.adapter.__aexit__(*args)
        finally:
            if self._stderr_capture is not None:
                await self._stderr_capture.aclose()

    def __del__(self) -> None:
        capture = getattr(self, "_stderr_capture", None)
        if capture is not None and not capture.started:
            capture.close_unstarted()

    def _stderr_updated(self, snapshot: MCPStderrSnapshot) -> None:
        self._state.latest_stderr_line = snapshot.latest_line
        self._state.stderr_tail = snapshot.tail
        if self._stderr_handler is not None:
            self._stderr_handler(self._state)

    def get_server_capabilities(self) -> Any:
        return self.client.server_capabilities

    async def list_tool_descriptors(self) -> list[Any]:
        return list(await self.client.list_tools(max_pages=_MAX_PAGES))

    async def as_tool(self, descriptor: Any) -> Any:
        return await as_langchain_tool(descriptor, self.client)

    async def list_prompts(self) -> list[Any]:
        return list(await self.client.list_prompts(max_pages=_MAX_PAGES))

    async def get_prompt_messages(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> list[BaseMessage]:
        response = await self.client.get_prompt(name, arguments)
        messages: list[BaseMessage] = []
        for message in response.messages:
            content = message.content
            if getattr(content, "type", "") != "text":
                raise ValueError(
                    f"Unsupported MCP prompt content type: {getattr(content, 'type', type(content).__name__)}"
                )
            text = str(getattr(content, "text", ""))
            if message.role == "user":
                messages.append(HumanMessage(content=text))
            elif message.role == "assistant":
                messages.append(AIMessage(content=text))
            else:
                raise ValueError(f"Unsupported MCP prompt role: {message.role}")
        return messages

    async def list_resources(self) -> list[Any]:
        return list(await self.client.list_resources(max_pages=_MAX_PAGES))

    async def read_resource(self, uri: str) -> list[Any]:
        return list(await self.client.read_resource(uri))


def _configured_secret_values(state: MCPServerState) -> set[str]:
    """Return resolved values MIRA already treats as private configuration."""
    values = {
        str(value)
        for value in (state.connection_config.get("env") or {}).values()
        if value
    }

    def collect(source: Any, resolved: Any) -> None:
        if isinstance(source, str) and isinstance(resolved, str):
            if "${" in source and source != resolved:
                values.add(resolved)
            return
        if isinstance(source, list) and isinstance(resolved, list):
            for source_item, resolved_item in zip(source, resolved):
                collect(source_item, resolved_item)
            return
        if isinstance(source, dict) and isinstance(resolved, dict):
            for key, source_item in source.items():
                if key in resolved:
                    collect(source_item, resolved[key])

    collect(state.config, state.connection_config)
    return values


__all__ = ["MiraMCPIntegration"]
