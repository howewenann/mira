"""Interactive reference client for MIRA's ACP Streamable HTTP transport."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.http import create_http_stream
from acp.schema import ClientCapabilities

# These transport entrypoints are intentionally runnable as ordinary files.
# Add their shared examples directory so both can reuse one small console layer.
ACP_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACP_EXAMPLES_DIR))

from client_common import (  # noqa: E402
    ReferenceClient as FullClient,
    ReferenceConsole,
    UpdateRenderer,
)


def parse_args() -> argparse.Namespace:
    """Accept either a scripted prompt or no prompt for interactive mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Send one prompt and exit.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/acp",
        help="Running MIRA ACP endpoint.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--follow-up", help="Send a second scripted prompt.")
    arguments = parser.parse_args()
    if arguments.follow_up and arguments.prompt is None:
        parser.error("--follow-up requires a prompt")
    return arguments


def response_mode(response: Any) -> str:
    """Read the current mode advertised by a stock ACP session response."""
    modes = getattr(response, "modes", None)
    return str(getattr(modes, "current_mode_id", "act") or "act")


async def main() -> None:
    arguments = parse_args()
    workspace = arguments.workspace.expanduser().resolve()

    # The client callback and reference console are identical to the stdio
    # example. Only the transport bootstrap and ownership rules differ here.
    renderer = UpdateRenderer(output=sys.stdout)
    client = FullClient(renderer)

    # HTTP connects to an already-running MIRA process instead of spawning one.
    transport = create_http_stream(arguments.url)
    connection = connect_to_agent(client, transport)
    try:
        # After transport setup, ACP uses the same initialize/session/prompt
        # lifecycle as stdio.
        await connection.initialize(PROTOCOL_VERSION, ClientCapabilities())
        session_response = await connection.new_session(str(workspace))
        session_id = session_response.session_id

        console = ReferenceConsole(
            connection=connection,
            client=client,
            workspace=workspace,
            session_id=session_id,
            mode=response_mode(session_response),
            transport_name="HTTP",
            allow_load=False,
            output=sys.stdout,
        )

        # ACP 0.12.1 cannot continue a loaded session on a new HTTP connection,
        # so this console focuses on new sessions and live multi-turn prompts.
        await console.run(
            initial_prompt=arguments.prompt,
            follow_up=arguments.follow_up,
        )
    finally:
        # The SDK connection and its HTTP transport own separate resources.
        await connection.close()
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
