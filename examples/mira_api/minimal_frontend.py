"""The smallest understandable frontend for MIRA's direct Python API."""

import asyncio
from typing import Any

from mira import MiraApplication
from mira.api import FrontendEvent, FrontendRequest, MessageEvent


class MinimalFrontend:
    """Callbacks through which MIRA communicates with this host application."""

    def emit(self, event: FrontendEvent) -> None:
        # MIRA streams ordered frontend events through this synchronous callback.
        # This minimal host displays only assistant text and ignores richer events.
        if isinstance(event, MessageEvent) and event.phase == "content":
            print(event.text, end="", flush=True)

    async def request(self, request: FrontendRequest) -> Any:
        # Approvals, AskUser, and other interactions pause MIRA here. A minimal
        # frontend has no trusted interaction UI, so it fails safely instead of
        # inventing an answer or approving a tool.
        raise RuntimeError(f"Interaction not supported: {type(request).__name__}")


async def main() -> None:
    frontend = MinimalFrontend()

    # A MiraApplication owns runtime resources shared by all of its sessions,
    # including agents, checkpoints, and configured MCP connections.
    application = await MiraApplication.start(workspace=".", frontend=frontend)
    try:
        # A MiraSession is one durable conversation inside the application.
        session = await application.open_session()

        # prompt() drives one agent turn. Output returns through frontend.emit(),
        # while any blocking interaction would arrive through frontend.request().
        await session.prompt("Reply only with PONG")
    finally:
        # Always release application-owned sessions, MCP servers, and resources.
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
