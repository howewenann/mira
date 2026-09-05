"""Interactive reference client for MIRA's ACP stdio transport."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process
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
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session", help="Load and continue a saved MIRA session.")
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

    # The client receives all ACP updates and permission requests. Rendering and
    # the local REPL are shared with the HTTP example so transport comparisons
    # use the same presentation behavior.
    renderer = UpdateRenderer(output=sys.stdout)
    client = FullClient(renderer)

    # stdio clients own the MIRA child process. The context manager starts
    # `python -m cli.main --acp` and closes its protocol streams on exit.
    async with spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "cli.main",
        "--acp",
    ) as (connection, _process):
        # Every ACP connection must negotiate the protocol before session calls.
        await connection.initialize(PROTOCOL_VERSION, ClientCapabilities())

        if arguments.session:
            # stdio supports true durable load-and-continue through session/load.
            session_response = await connection.load_session(
                str(workspace),
                arguments.session,
            )
            session_id = arguments.session
        else:
            # session/new creates a durable MIRA conversation in this workspace.
            session_response = await connection.new_session(str(workspace))
            session_id = session_response.session_id

        console = ReferenceConsole(
            connection=connection,
            client=client,
            workspace=workspace,
            session_id=session_id,
            mode=response_mode(session_response),
            transport_name="stdio",
            allow_load=True,
            output=sys.stdout,
        )

        # With a prompt this is a one-shot client. Without one it enters the
        # reference REPL and reuses the same ACP session for every normal prompt.
        await console.run(
            initial_prompt=arguments.prompt,
            follow_up=arguments.follow_up,
        )


if __name__ == "__main__":
    asyncio.run(main())
