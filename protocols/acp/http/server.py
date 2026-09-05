"""Experimental stock-SDK ACP Streamable HTTP bootstrap."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from protocols.acp.http.listen import validate_listen
from protocols.acp.http.splash import print_http_splash
from protocols.acp.shared.agent import MiraAgent


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

    factory = agent_factory or MiraAgentFactory()
    app = create_asgi_app(factory)
    config = _server_config(listen)
    server_task = asyncio.create_task(
        serve(app, config, shutdown_trigger=shutdown_trigger)
    )
    try:
        await _wait_for_listener(listen, server_task)
        print_http_splash(listen)
        await server_task
    finally:
        if not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        await factory.shutdown()


async def _wait_for_listener(listen: str, server_task: asyncio.Task[None]) -> None:
    """Wait for the public TCP listener without relying on Hypercorn internals."""
    parsed = urlsplit(f"//{listen}")
    host = parsed.hostname
    port = parsed.port
    assert host is not None and port is not None

    while not server_task.done():
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.02)
            continue

        writer.close()
        await writer.wait_closed()

        # Give an immediate bind failure (for example, an occupied port) time
        # to win before reporting that this specific Hypercorn task is ready.
        await asyncio.sleep(0.05)
        if not server_task.done():
            return

    await server_task
    raise RuntimeError("ACP HTTP server stopped before its listener became ready")


def _server_config(listen: str) -> Any:
    """Build Hypercorn configuration through its public configuration API."""
    from hypercorn.config import Config

    config = Config()
    config.bind = [listen]
    config.alpn_protocols = ["h2", "http/1.1"]
    # The MIRA ready panel reports the endpoint. Keep Hypercorn warnings and
    # errors while suppressing only its redundant normal startup INFO line.
    config.loglevel = "WARNING"
    return config


def run_http_server(listen: str) -> None:
    asyncio.run(serve_http(listen))


__all__ = ["MiraAgentFactory", "run_http_server", "serve_http"]
