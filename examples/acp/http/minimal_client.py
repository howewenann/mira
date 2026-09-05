"""Minimal external client for MIRA's experimental ACP HTTP endpoint."""

import asyncio
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.http import create_http_stream
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallUpdate,
)


class MinimalClient(Client):
    """The two callbacks an ACP agent uses to communicate with its client."""

    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any
    ) -> None:
        # MIRA streams assistant text, thoughts, and tool updates through this
        # callback. The minimal client displays text content only.
        del session_id, kwargs
        if isinstance(update, AgentMessageChunk) and isinstance(
            update.content, TextContentBlock
        ):
            print(update.content.text, end="", flush=True)

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        # MIRA may pause for human approval. This minimal example has no
        # interaction UI, so it always cancels. See full_client.py for choices.
        del session_id, tool_call, options, kwargs
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


async def main() -> None:
    client = MinimalClient()

    # HTTP differs from stdio only in connection setup: MIRA is already running,
    # so the client creates a transport for its public ACP endpoint.
    transport = create_http_stream("http://127.0.0.1:8765/acp")
    connection = connect_to_agent(client, transport)
    try:
        # The ACP session and prompt lifecycle matches the stdio example.
        await connection.initialize(PROTOCOL_VERSION)
        created_session = await connection.new_session(str(Path.cwd()))

        # Responses arrive asynchronously through client.session_update().
        await connection.prompt(
            created_session.session_id,
            [text_block("Reply only with PONG")],
        )
    finally:
        # The connection and its underlying HTTP transport own separate resources.
        await connection.close()
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
