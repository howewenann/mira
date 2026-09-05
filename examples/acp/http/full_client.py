"""Full safe client for MIRA's experimental ACP HTTP transport."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.http import create_http_stream
from acp.interfaces import Client
from acp.schema import ClientCapabilities


class FullClient(Client):
    """Print updates and ask before selecting a permission option."""

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
        else:
            print(f"\n[{type(update).__name__}] {update}", file=sys.stderr)

    async def request_permission(self, **kwargs: Any) -> Any:
        tool_call = kwargs.get("tool_call")
        options = kwargs.get("options") or []
        title = getattr(tool_call, "title", "Permission required")
        print(f"\n{title}", file=sys.stderr)
        for index, option in enumerate(options, start=1):
            print(f"  {index}. {option.name} ({option.kind})", file=sys.stderr)
        answer = (await asyncio.to_thread(input, "Select an option [deny]: ")).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            option = options[int(answer) - 1]
            return {
                "outcome": {
                    "outcome": "selected",
                    "optionId": option.option_id,
                }
            }
        return {"outcome": {"outcome": "cancelled"}}


async def run(args: argparse.Namespace) -> None:
    client = FullClient()
    transport = create_http_stream(args.url)
    connection = connect_to_agent(client, transport)
    try:
        await connection.initialize(PROTOCOL_VERSION, ClientCapabilities())
        created = await connection.new_session(str(args.workspace))
        session_id = created.session_id
        print(f"Session: {session_id}", file=sys.stderr)
        await connection.prompt(session_id, [text_block(args.prompt)])
        if args.follow_up:
            await connection.prompt(session_id, [text_block(args.follow_up)])
    finally:
        await connection.close()
        await transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/acp",
        help="Running MIRA ACP endpoint.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--follow-up", help="Send a second prompt on the live connection.")
    parser.add_argument("prompt")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
