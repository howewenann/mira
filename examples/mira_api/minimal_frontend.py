"""Minimal frontend for MIRA's direct Python API."""

import asyncio
from typing import Any

from mira import MiraApplication
from mira.api import FrontendEvent, FrontendRequest, MessageEvent


class MinimalFrontend:
    def emit(self, event: FrontendEvent) -> None:
        if isinstance(event, MessageEvent) and event.phase == "content":
            print(event.text, end="", flush=True)

    async def request(self, request: FrontendRequest) -> Any:
        raise RuntimeError(f"Interaction not supported: {type(request).__name__}")


async def main() -> None:
    app = await MiraApplication.start(workspace=".", frontend=MinimalFrontend())
    try:
        session = await app.open_session()
        await session.prompt("Reply only with PONG")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
