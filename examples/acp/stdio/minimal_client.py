"""Minimal external ACP client that spawns MIRA over stdio."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
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

    # The client owns the child process and ACP stdio streams for this context.
    # stdout belongs to ACP; MIRA keeps protocol-unrelated output off that stream.
    async with spawn_agent_process(
        client, sys.executable, "-m", "cli.main", "--acp"
    ) as (connection, _process):
        # ACP connections must be initialized before session methods are used.
        await connection.initialize(PROTOCOL_VERSION)

        # session/new creates one MIRA conversation rooted in this workspace.
        created_session = await connection.new_session(str(Path.cwd()))

        # Responses arrive asynchronously through client.session_update().
        await connection.prompt(
            created_session.session_id,
            [text_block("Reply only with PONG")],
        )

    # Leaving the context closes the ACP streams and the MIRA child process.


if __name__ == "__main__":
    asyncio.run(main())
