"""Thin per-server integration with LangChain's first-party MCP stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

with suppress_langchain_beta_warning():
    from langchain.mcp import MCPAdapter, as_langchain_tool

from agent.mcp.auth import create_http_oauth
from agent.mcp.models import MCPServerState
from core.diagnostics.logging import get_diagnostics_logger

_MAX_PAGES = 1000


class MiraMCPIntegration:
    """Connect one configured server through ``MCPAdapter`` and FastMCP."""

    def __init__(self, state: MCPServerState) -> None:
        config = state.connection_config
        self._state = state
        if state.transport == "stdio":
            transport = StdioTransport(
                command=str(config["command"]),
                args=list(config.get("args") or []),
                env=dict(config.get("env") or {}),
                keep_alive=False,
                # FastMCP's public transport option is the modern equivalent of
                # the former MCP SDK ``stdio_client(..., errlog=None)`` override.
                log_file=Path(os.devnull),
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
        await self.adapter.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.adapter is None:  # pragma: no cover - enter constructs HTTP adapter
            return
        await self.adapter.__aexit__(*args)

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


__all__ = ["MiraMCPIntegration"]
