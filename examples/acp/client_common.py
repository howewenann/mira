"""Shared presentation and REPL behavior for MIRA's full ACP clients.

This module deliberately knows only the public Agent Client Protocol. It does
not import MIRA, so the reference clients see exactly what any external ACP
frontend can see.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from acp import text_block
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    PermissionOption,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
)

InputReader = Callable[[str], str]


def structured_text(value: Any) -> str:
    """Return a readable JSON representation of an ACP object or native value."""
    return json.dumps(
        _json_compatible(value),
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _json_compatible(value: Any) -> Any:
    """Recursively convert public ACP models nested inside native containers."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
        return _json_compatible(dumped)
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


class UpdateRenderer:
    """Render stock ACP session updates as a small reference transcript."""

    def __init__(self, *, output: TextIO) -> None:
        self.output = output
        self.raw_enabled = False
        self._active_message: tuple[str, str | None] | None = None

    def render(self, update: Any) -> None:
        """Render one update without assuming that MIRA sent a known subtype."""
        if self.raw_enabled:
            self._render_raw("ACP UPDATE", update)

        if isinstance(update, AgentMessageChunk):
            self._render_message_chunk("MIRA", update)
        elif isinstance(update, AgentThoughtChunk):
            self._render_message_chunk("THOUGHT", update)
        elif isinstance(update, UserMessageChunk):
            self._render_message_chunk("USER [replay]", update)
        elif isinstance(update, ToolCallStart):
            self._render_tool_start(update)
        elif isinstance(update, ToolCallProgress):
            self._render_tool_progress(update)
        elif isinstance(update, AgentPlanUpdate):
            self._render_plan(update)
        else:
            self._section(f"ACP UPDATE [{type(update).__name__}]")
            print(structured_text(update), file=self.output)

    def finish_stream(self) -> None:
        """End an in-progress message stream before the next console block."""
        if self._active_message is not None:
            print(file=self.output)
            self._active_message = None

    def render_permission_raw(
        self,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> None:
        """Show the exact public ACP permission objects when raw mode is on."""
        if not self.raw_enabled:
            return
        self._render_raw(
            "ACP PERMISSION",
            {"tool_call": tool_call, "options": options},
        )

    def _render_message_chunk(self, heading: str, update: Any) -> None:
        message_key = (heading, getattr(update, "message_id", None))
        if message_key != self._active_message:
            self.finish_stream()
            print(f"\n{heading}", file=self.output)
            self._active_message = message_key

        content = update.content
        text = getattr(content, "text", None)
        if text is not None:
            print(text, end="", file=self.output, flush=True)
        else:
            print(structured_text(content), file=self.output)

    def _render_tool_start(self, update: ToolCallStart) -> None:
        self._section(f"TOOL [{update.kind or 'other'}]")
        print(f"id: {update.tool_call_id}", file=self.output)
        print(f"title: {update.title}", file=self.output)
        print(f"status: {update.status or 'pending'}", file=self.output)
        if update.raw_input is not None:
            print("\ninput:", file=self.output)
            print(structured_text(update.raw_input), file=self.output)
        self._render_tool_content(update.content)

    def _render_tool_progress(self, update: ToolCallProgress) -> None:
        terminal_statuses = {"completed", "failed"}
        heading = (
            "TOOL RESULT" if update.status in terminal_statuses else "TOOL UPDATE"
        )
        self._section(heading)
        print(f"id: {update.tool_call_id}", file=self.output)
        if update.title:
            print(f"title: {update.title}", file=self.output)
        if update.status:
            print(f"status: {update.status}", file=self.output)
        self._render_tool_content(update.content)
        if update.raw_output is not None:
            print("\nraw output:", file=self.output)
            print(structured_text(update.raw_output), file=self.output)

    def _render_tool_content(self, content: list[Any] | None) -> None:
        for item in content or []:
            if isinstance(item, ContentToolCallContent):
                print("\ncontent:", file=self.output)
                text = getattr(item.content, "text", None)
                rendered = text if text is not None else structured_text(item.content)
                print(rendered, file=self.output)
            elif isinstance(item, FileEditToolCallContent):
                print(f"\ndiff: {item.path}", file=self.output)
                print("--- before", file=self.output)
                old_text = item.old_text if item.old_text is not None else "<new file>"
                print(old_text, file=self.output)
                print("+++ after", file=self.output)
                print(item.new_text, file=self.output)
            else:
                print("\ncontent:", file=self.output)
                print(structured_text(item), file=self.output)

    def _render_plan(self, update: AgentPlanUpdate) -> None:
        symbols = {"pending": " ", "in_progress": ">", "completed": "x"}
        self._section("ACP PLAN / write_todos")
        for entry in update.entries:
            print(f"[{symbols.get(entry.status, ' ')}] {entry.content}", file=self.output)

    def _render_raw(self, label: str, value: Any) -> None:
        self._section(f"RAW {label} [{type(value).__name__}]")
        print(structured_text(value), file=self.output)

    def _section(self, heading: str) -> None:
        self.finish_stream()
        print(f"\n{heading}", file=self.output)


class ReferenceClient(Client):
    """Receive ACP callbacks and resolve permission requests conservatively."""

    def __init__(
        self,
        renderer: UpdateRenderer,
        *,
        input_reader: InputReader = input,
    ) -> None:
        self.renderer = renderer
        self.input_reader = input_reader

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        """Receive streamed output from MIRA through the ACP connection."""
        del session_id, kwargs
        self.renderer.render(update)

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> Any:
        """Present exactly the choices supplied by ACP and return their option ID."""
        del session_id, kwargs
        self.renderer.finish_stream()
        self.renderer.render_permission_raw(tool_call, options)

        print("\nPERMISSION REQUIRED", file=self.renderer.output)
        print(tool_call.title or "MIRA needs your approval.", file=self.renderer.output)
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option.name} ({option.kind})", file=self.renderer.output)

        try:
            answer = await asyncio.to_thread(self.input_reader, "Choice [cancel]: ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        answer = answer.strip()

        if not answer.isdigit() or not 1 <= int(answer) <= len(options):
            print("Permission cancelled.", file=self.renderer.output)
            return {"outcome": {"outcome": "cancelled"}}

        selected_option = options[int(answer) - 1]
        if selected_option.name in {"Reply in chat", "Revise in chat"}:
            print(
                "The interrupted turn ended. Enter your response as the next message.",
                file=self.renderer.output,
            )
        return {
            "outcome": {
                "outcome": "selected",
                "optionId": selected_option.option_id,
            }
        }


class ReferenceConsole:
    """Small transport-independent REPL around a public ACP connection."""

    def __init__(
        self,
        *,
        connection: Any,
        client: ReferenceClient,
        workspace: Path,
        session_id: str,
        mode: str,
        transport_name: str,
        allow_load: bool,
        output: TextIO,
        input_reader: InputReader = input,
    ) -> None:
        self.connection = connection
        self.client = client
        self.workspace = workspace
        self.session_id = session_id
        self.mode = mode
        self.transport_name = transport_name
        self.allow_load = allow_load
        self.output = output
        self.input_reader = input_reader

    def show_welcome(self) -> None:
        """Describe the live connection using only client-known ACP state."""
        print("Connected to MIRA", file=self.output)
        print(f"Transport: {self.transport_name}", file=self.output)
        print(f"Workspace: {self.workspace}", file=self.output)
        print(f"Session: {self.session_id}", file=self.output)
        print(f"Mode: {self.mode.upper()}", file=self.output)
        print("\nType /help for commands.", file=self.output)

    async def run(
        self,
        *,
        initial_prompt: str | None,
        follow_up: str | None,
    ) -> None:
        """Run scripted prompts or enter the interactive reference console."""
        self.show_welcome()
        if initial_prompt is not None:
            await self.send_prompt(initial_prompt)
            if follow_up is not None:
                await self.send_prompt(follow_up)
            return

        while True:
            try:
                line = await asyncio.to_thread(self.input_reader, "\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\nDisconnecting.", file=self.output)
                return

            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    keep_running = await self.handle_command(line)
                else:
                    keep_running = await self.send_prompt(line)
            except Exception as exc:
                print(f"ERROR: {exc}", file=self.output)
                continue
            if keep_running is False:
                return

    async def send_prompt(self, prompt: str) -> bool:
        """Send a normal user message to the current ACP session."""
        response = await self.connection.prompt(
            self.session_id,
            [text_block(prompt)],
        )
        self.client.renderer.finish_stream()
        if getattr(response, "stop_reason", None) == "cancelled":
            print("[turn cancelled]", file=self.output)
        return True

    async def handle_command(self, line: str) -> bool:
        """Handle one local console command without sending it to the model."""
        parts = line.split(maxsplit=2)
        command = parts[0].lower()

        if command == "/help":
            self._show_help()
        elif command == "/new":
            created = await self.connection.new_session(str(self.workspace))
            self.session_id = created.session_id
            self.mode = _response_mode(created, default="act")
            print(f"New session: {self.session_id}", file=self.output)
        elif command == "/session":
            print(f"Session: {self.session_id}", file=self.output)
            print(f"Mode: {self.mode.upper()}", file=self.output)
        elif command == "/mode":
            await self._set_mode(parts)
        elif command == "/raw":
            self._set_raw(parts)
        elif command == "/load":
            await self._load(parts)
        elif command == "/cancel-after":
            await self._cancel_after(parts)
        elif command == "/quit":
            return False
        else:
            print(f"Unknown command: {command}. Type /help.", file=self.output)
        return True

    async def _set_mode(self, parts: list[str]) -> None:
        if len(parts) != 2 or parts[1].lower() not in {"act", "plan"}:
            print("Usage: /mode act|plan", file=self.output)
            return
        mode = parts[1].lower()
        await self.connection.set_session_mode(self.session_id, mode)
        self.mode = mode
        print(f"Mode: {mode.upper()}", file=self.output)

    def _set_raw(self, parts: list[str]) -> None:
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            print("Usage: /raw on|off", file=self.output)
            return
        self.client.renderer.raw_enabled = parts[1].lower() == "on"
        state = "on" if self.client.renderer.raw_enabled else "off"
        print(f"Raw ACP inspection: {state}", file=self.output)

    async def _load(self, parts: list[str]) -> None:
        if not self.allow_load:
            print(
                "HTTP session/load is replay-only in ACP 0.12.1 and cannot be continued; "
                "this console leaves it disabled.",
                file=self.output,
            )
            return
        if len(parts) != 2:
            print("Usage: /load <session-id>", file=self.output)
            return

        session_id = parts[1]
        loaded = await self.connection.load_session(str(self.workspace), session_id)
        self.session_id = session_id
        self.mode = _response_mode(loaded, default="act")
        self.client.renderer.finish_stream()
        print(f"Loaded session: {self.session_id}", file=self.output)
        print(f"Mode: {self.mode.upper()}", file=self.output)

    async def _cancel_after(self, parts: list[str]) -> None:
        if len(parts) != 3:
            print("Usage: /cancel-after <seconds> <prompt>", file=self.output)
            return
        try:
            delay = float(parts[1])
        except ValueError:
            print("Delay must be a number of seconds.", file=self.output)
            return
        if delay < 0:
            print("Delay must not be negative.", file=self.output)
            return

        prompt_task = asyncio.create_task(self.send_prompt(parts[2]))
        await asyncio.sleep(delay)
        await self.connection.cancel(self.session_id)
        await prompt_task

    def _show_help(self) -> None:
        commands = [
            "/help",
            "/new",
            "/session",
            "/mode act",
            "/mode plan",
            "/raw on",
            "/raw off",
            "/cancel-after <seconds> <prompt>",
        ]
        if self.allow_load:
            commands.append("/load <session-id>")
        commands.append("/quit")
        print("Local commands:\n  " + "\n  ".join(commands), file=self.output)


def _response_mode(response: Any, *, default: str) -> str:
    modes = getattr(response, "modes", None)
    return str(getattr(modes, "current_mode_id", default) or default)


__all__ = [
    "ReferenceClient",
    "ReferenceConsole",
    "UpdateRenderer",
    "structured_text",
]
