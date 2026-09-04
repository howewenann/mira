"""Minimal safe client for MIRA's stdio and experimental HTTP ACP transports."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, spawn_agent_process, text_block
from acp.interfaces import Client
from acp.schema import ClientCapabilities


class ExampleClient(Client):
    """Print streamed text and deny permission requests by default."""

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        del session_id, kwargs
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if text:
            print(text, end="", flush=True)

    async def request_permission(self, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call")
        title = getattr(tool_call, "title", "Permission required")
        print(f"\nDenied: {title}", file=sys.stderr)
        return {"outcome": {"outcome": "cancelled"}}


async def run_prompts(connection: Any, args: argparse.Namespace) -> None:
    await connection.initialize(PROTOCOL_VERSION, ClientCapabilities())
    if args.session:
        await connection.load_session(str(args.workspace), args.session)
        session_id = args.session
    else:
        created = await connection.new_session(str(args.workspace))
        session_id = created.session_id
    print(f"Session: {session_id}", file=sys.stderr)
    await connection.prompt(session_id, [text_block(args.prompt)])
    if args.follow_up:
        await connection.prompt(session_id, [text_block(args.follow_up)])


async def run_stdio(args: argparse.Namespace) -> None:
    client = ExampleClient()
    async with spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "cli.main",
        "--acp",
    ) as (connection, _process):
        await run_prompts(connection, args)


async def run_http(args: argparse.Namespace) -> None:
    from acp.http import create_http_stream

    client = ExampleClient()
    transport = create_http_stream(args.http)
    connection = connect_to_agent(client, transport)
    try:
        await run_prompts(connection, args)
    finally:
        await connection.close()
        await transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--stdio", action="store_true", help="Spawn mira --acp.")
    transport.add_argument("--http", metavar="URL", help="Connect to a running ACP URL.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session", help="Load a saved session (stdio only in ACP 0.12.1).")
    parser.add_argument("--follow-up", help="Send a second prompt on the live connection.")
    parser.add_argument("prompt")
    args = parser.parse_args()
    if args.http and args.session:
        parser.error(
            "agent-client-protocol 0.12.1 cannot continue a loaded session over HTTP"
        )
    return args


async def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    if args.http:
        await run_http(args)
    else:
        await run_stdio(args)


if __name__ == "__main__":
    asyncio.run(main())
