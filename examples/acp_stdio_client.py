"""Spawn MIRA over ACP stdio and send one prompt."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client


class ExampleClient(Client):
    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any
    ) -> None:
        del session_id, kwargs
        text = getattr(getattr(update, "content", None), "text", None)
        if text:
            print(text, end="", flush=True)

    async def request_permission(self, **kwargs: Any) -> Any:
        del kwargs
        return {"outcome": {"outcome": "cancelled"}}


async def main() -> None:
    async with spawn_agent_process(
        ExampleClient(), sys.executable, "-m", "cli.main", "--acp"
    ) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(Path.cwd()))
        await connection.prompt(
            session.session_id,
            [text_block("Reply only with PONG")],
        )


if __name__ == "__main__":
    asyncio.run(main())
