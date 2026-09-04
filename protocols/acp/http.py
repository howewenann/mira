"""Experimental stock-SDK ACP Streamable HTTP bootstrap."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from protocols.acp.listen import validate_listen
from protocols.acp.server import MiraAgent


class MiraAgentFactory:
    """Create isolated agents per ACP connection and own shutdown cleanup."""

    def __init__(self) -> None:
        self._agents: set[MiraAgent] = set()

    def __call__(self, connection: Any) -> MiraAgent:
        del connection
        agent = MiraAgent()
        self._agents.add(agent)
        return agent

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    async def shutdown(self) -> None:
        agents = tuple(self._agents)
        self._agents.clear()
        await asyncio.gather(
            *(agent.shutdown() for agent in agents),
            return_exceptions=True,
        )


async def serve_http(
    listen: str,
    *,
    shutdown_trigger: Callable[[], Awaitable[None]] | None = None,
    agent_factory: MiraAgentFactory | None = None,
) -> None:
    """Serve one stock ``MiraAgent`` per HTTP connection through Hypercorn."""
    listen = validate_listen(listen)
    from acp.http.asgi import create_asgi_app
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    factory = agent_factory or MiraAgentFactory()
    app = create_asgi_app(factory)
    config = Config()
    config.bind = [listen]
    config.alpn_protocols = ["h2", "http/1.1"]
    try:
        await serve(app, config, shutdown_trigger=shutdown_trigger)
    finally:
        await factory.shutdown()


def run_http_server(listen: str) -> None:
    asyncio.run(serve_http(listen))


__all__ = ["MiraAgentFactory", "run_http_server", "serve_http"]
