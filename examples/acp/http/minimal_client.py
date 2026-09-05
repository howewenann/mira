"""Connect to MIRA's experimental ACP HTTP endpoint and send one prompt."""

import asyncio
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.http import create_http_stream
from acp.interfaces import Client


class MinimalClient(Client):
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
    transport = create_http_stream("http://127.0.0.1:8765/acp")
    connection = connect_to_agent(MinimalClient(), transport)
    try:
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(Path.cwd()))
        await connection.prompt(
            session.session_id,
            [text_block("Reply only with PONG")],
        )
    finally:
        await connection.close()
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
