"""Use ACP stdio for an interactive, inspectable conversation with MIRA."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    ContentToolCallContent,
    DeniedOutcome,
    FileEditToolCallContent,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
)


class MiraClient(Client):
    """Receive the callbacks that MIRA sends from the other end of ACP."""

    def __init__(self) -> None:
        self.raw_updates = False
        self._open_message: tuple[str, str | None] | None = None

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        """Display one streamed agent-to-client session update."""
        del session_id, kwargs

        if self.raw_updates:
            print("\nRAW ACP UPDATE")
            print(update.model_dump_json(by_alias=True, exclude_none=True, indent=2))

        # Direct branches make the public ACP update types easy to discover.
        if isinstance(update, AgentMessageChunk):
            self._display_message("MIRA", update.message_id, update.content)
        elif isinstance(update, AgentThoughtChunk):
            self._display_message("THOUGHT", update.message_id, update.content)
        elif isinstance(update, UserMessageChunk):
            self._display_message("USER [replay]", update.message_id, update.content)
        elif isinstance(update, ToolCallStart):
            self._display_tool_start(update)
        elif isinstance(update, ToolCallProgress):
            self._display_tool_progress(update)
        elif isinstance(update, AgentPlanUpdate):
            self.finish_message()
            print("\nACP PLAN / write_todos")
            symbols = {"pending": " ", "in_progress": ">", "completed": "x"}
            for entry in update.entries:
                print(f"[{symbols.get(entry.status, ' ')}] {entry.content}")
        else:
            self.finish_message()
            print(f"\nACP UPDATE [{type(update).__name__}]")
            print(update.model_dump_json(by_alias=True, exclude_none=True, indent=2))

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Let the user select one of the exact choices supplied by ACP."""
        del session_id, kwargs
        self.finish_message()

        print("\nPERMISSION REQUIRED")
        print(tool_call.title or "MIRA needs your approval.")
        for number, option in enumerate(options, start=1):
            print(f"{number}. {option.name} ({option.kind})")

        try:
            answer = await asyncio.to_thread(input, "Choice [cancel]: ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        answer = answer.strip()

        if not answer.isdigit() or not 1 <= int(answer) <= len(options):
            print("Permission cancelled.")
            return RequestPermissionResponse(
                outcome=DeniedOutcome(outcome="cancelled")
            )

        selected_option = options[int(answer) - 1]
        if selected_option.name in {"Reply in chat", "Revise in chat"}:
            print("The interrupted turn ended. Enter your reply as the next message.")
        return RequestPermissionResponse(
            outcome=AllowedOutcome(
                outcome="selected",
                optionId=selected_option.option_id,
            )
        )

    def finish_message(self) -> None:
        """Finish streamed text before displaying another kind of update."""
        if self._open_message is not None:
            print()
            self._open_message = None

    def _display_message(
        self,
        heading: str,
        message_id: str | None,
        content: Any,
    ) -> None:
        message = (heading, message_id)
        if message != self._open_message:
            self.finish_message()
            print(f"\n{heading}")
            self._open_message = message

        if isinstance(content, TextContentBlock):
            print(content.text, end="", flush=True)
        else:
            print(content.model_dump_json(by_alias=True, exclude_none=True, indent=2))

    def _display_tool_start(self, update: ToolCallStart) -> None:
        self.finish_message()
        print(f"\nTOOL [{update.kind or 'other'}]")
        print(f"id: {update.tool_call_id}")
        print(f"title: {update.title}")
        print(f"status: {update.status or 'pending'}")
        if update.raw_input is not None:
            print("input:")
            print(json.dumps(update.raw_input, indent=2, default=str))
        self._display_tool_content(update.content)

    def _display_tool_progress(self, update: ToolCallProgress) -> None:
        self.finish_message()
        if update.status in {"completed", "failed"}:
            heading = "TOOL RESULT"
        else:
            heading = "TOOL UPDATE"
        print(f"\n{heading}")
        print(f"id: {update.tool_call_id}")
        if update.title:
            print(f"title: {update.title}")
        if update.status:
            print(f"status: {update.status}")
        self._display_tool_content(update.content)
        if update.raw_output is not None:
            print("raw output:")
            print(json.dumps(update.raw_output, indent=2, default=str))

    def _display_tool_content(self, content: list[Any] | None) -> None:
        for item in content or []:
            if isinstance(item, ContentToolCallContent):
                if isinstance(item.content, TextContentBlock):
                    print(f"content: {item.content.text}")
                else:
                    print(item.content.model_dump_json(by_alias=True, indent=2))
            elif isinstance(item, FileEditToolCallContent):
                old_text = item.old_text if item.old_text is not None else "<new file>"
                print(f"diff: {item.path}")
                print(f"--- before\n{old_text}\n+++ after\n{item.new_text}")
            else:
                print(item.model_dump_json(by_alias=True, exclude_none=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Send one prompt and exit.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session", help="Load and continue a saved MIRA session.")
    parser.add_argument("--follow-up", help="Send a second scripted prompt.")
    arguments = parser.parse_args()
    if arguments.follow_up and arguments.prompt is None:
        parser.error("--follow-up requires a prompt")
    return arguments


def show_commands() -> None:
    print(
        "Local commands:\n"
        "  /help\n"
        "  /new\n"
        "  /session\n"
        "  /load <session-id>\n"
        "  /mode act|plan\n"
        "  /raw on|off\n"
        "  /cancel-after <seconds> <prompt>\n"
        "  /quit"
    )


async def prompt_loop(
    connection: Any,
    client: MiraClient,
    workspace: Path,
    session_id: str,
    mode: str,
) -> None:
    """Send messages and ACP session commands until the user quits."""
    print("Connected to MIRA")
    print("Transport: stdio")
    print(f"Workspace: {workspace}")
    print(f"Session: {session_id}")
    print(f"Mode: {mode.upper()}")
    print("\nType /help for commands.")

    while True:
        try:
            line = await asyncio.to_thread(input, "\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nDisconnecting.")
            return

        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=2)
        command = parts[0].lower()

        try:
            if command == "/quit":
                return
            if command == "/help":
                show_commands()
            elif command == "/session":
                print(f"Session: {session_id}")
                print(f"Mode: {mode.upper()}")
            elif command == "/new":
                session_response = await connection.new_session(str(workspace))
                session_id = session_response.session_id
                mode = session_response.modes.current_mode_id
                print(f"New session: {session_id}")
            elif command == "/load":
                if len(parts) != 2:
                    print("Usage: /load <session-id>")
                    continue
                session_id = parts[1]
                session_response = await connection.load_session(
                    str(workspace), session_id
                )
                mode = session_response.modes.current_mode_id
                client.finish_message()
                print(f"Loaded session: {session_id}")
            elif command == "/mode":
                if len(parts) != 2 or parts[1].lower() not in {"act", "plan"}:
                    print("Usage: /mode act|plan")
                    continue
                mode = parts[1].lower()
                await connection.set_session_mode(session_id, mode)
                print(f"Mode: {mode.upper()}")
            elif command == "/raw":
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    print("Usage: /raw on|off")
                    continue
                client.raw_updates = parts[1].lower() == "on"
                print(f"Raw ACP updates: {parts[1].lower()}")
            elif command == "/cancel-after":
                if len(parts) != 3:
                    print("Usage: /cancel-after <seconds> <prompt>")
                    continue
                try:
                    delay = float(parts[1])
                except ValueError:
                    print("Delay must be a number of seconds.")
                    continue
                if delay < 0:
                    print("Delay must not be negative.")
                    continue
                prompt_task = asyncio.create_task(
                    connection.prompt(session_id, [text_block(parts[2])])
                )
                await asyncio.sleep(delay)
                if not prompt_task.done():
                    await connection.cancel(session_id)
                await prompt_task
                client.finish_message()
            elif command.startswith("/"):
                print(f"Unknown command: {command}. Type /help.")
            else:
                await connection.prompt(session_id, [text_block(line)])
                client.finish_message()
        except Exception as error:
            print(f"ERROR: {error}")


async def main() -> None:
    arguments = parse_args()
    workspace = arguments.workspace.expanduser().resolve()
    client = MiraClient()

    # A stdio ACP client owns the MIRA process. This context manager starts
    # `python -m cli.main --acp` and connects to its stdin/stdout streams.
    async with spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "cli.main",
        "--acp",
    ) as (connection, _process):
        # ACP must be initialized before the client creates or loads a session.
        await connection.initialize(PROTOCOL_VERSION, ClientCapabilities())

        if arguments.session:
            session_response = await connection.load_session(
                str(workspace), arguments.session
            )
            session_id = arguments.session
        else:
            session_response = await connection.new_session(str(workspace))
            session_id = session_response.session_id
        mode = session_response.modes.current_mode_id

        # A positional prompt keeps one-shot scripting useful. With no prompt,
        # the loop below sends multiple messages through this same ACP session.
        if arguments.prompt is not None:
            await connection.prompt(session_id, [text_block(arguments.prompt)])
            client.finish_message()
            if arguments.follow_up is not None:
                await connection.prompt(session_id, [text_block(arguments.follow_up)])
                client.finish_message()
        else:
            await prompt_loop(connection, client, workspace, session_id, mode)

    # Leaving the context closes the streams and the MIRA child process.


if __name__ == "__main__":
    asyncio.run(main())
