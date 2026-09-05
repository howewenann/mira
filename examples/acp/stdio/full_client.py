"""Full safe client for MIRA's ACP stdio transport."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
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


async def run(args: argparse.Namespace) -> None:
    client = FullClient()
    async with spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "cli.main",
        "--acp",
    ) as (connection, _process):
        await run_prompts(connection, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session", help="Load a saved MIRA session.")
    parser.add_argument("--follow-up", help="Send a second prompt on the live connection.")
    parser.add_argument("prompt")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
