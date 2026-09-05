"""Stock ACP stdio bootstrap."""

from __future__ import annotations

import asyncio
import sys

from acp import run_agent

from protocols.acp.shared.agent import MiraAgent


async def serve() -> None:
    server = MiraAgent()
    if sys.platform == "win32":
        # Finish optional native initialization before ACP starts its stdin reader.
        try:
            import numpy  # noqa: F401
        except ImportError:
            pass
    try:
        await run_agent(server)
    finally:
        await server.shutdown()


def run_server() -> None:
    asyncio.run(serve())


__all__ = ["run_server", "serve"]
