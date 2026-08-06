"""MCP client transport behavior owned by MIRA."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_mcp_adapters.callbacks import CallbackContext
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import _expand_env_vars
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MiraMCPClient(MultiServerMCPClient):
    """Keep stdio server errors off Textual's redirected stderr."""

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        connection = self.connections.get(server_name)
        if connection is None or connection.get("transport") != "stdio":
            async with super().session(server_name, auto_initialize=auto_initialize) as session:
                yield session
            return

        callbacks = self.callbacks.to_mcp_format(context=CallbackContext(server_name=server_name))
        session_kwargs = {}
        if callbacks.logging_callback is not None:
            session_kwargs["logging_callback"] = callbacks.logging_callback
        if callbacks.elicitation_callback is not None:
            session_kwargs["elicitation_callback"] = callbacks.elicitation_callback

        env = connection.get("env")
        resolved_env = {key: _expand_env_vars(value) for key, value in env.items()} if env is not None else None
        server = StdioServerParameters(
            command=connection["command"],
            args=connection["args"],
            env=resolved_env,
        )
        async with (
            stdio_client(server, errlog=None) as (read, write),
            ClientSession(read, write, **session_kwargs) as session,
        ):
            if auto_initialize:
                await session.initialize()
            yield session


__all__ = ["MiraMCPClient"]
