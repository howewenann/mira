"""Focused behavior tests for the self-contained full ACP examples."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from acp import (
    plan_entry,
    start_tool_call,
    text_block,
    tool_diff_content,
    update_plan,
    update_tool_call,
)
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    CurrentModeUpdate,
    PermissionOption,
    ToolCallUpdate,
    UserMessageChunk,
)

from examples.acp.http.full_client import MiraClient as HttpMiraClient
from examples.acp.stdio.full_client import MiraClient as StdioMiraClient


CLIENT_TYPES = (StdioMiraClient, HttpMiraClient)


class FullClientRenderingTests(unittest.IsolatedAsyncioTestCase):
    async def render_with_both_clients(self, *updates: object) -> list[str]:
        outputs: list[str] = []
        for client_type in CLIENT_TYPES:
            output = io.StringIO()
            client = client_type()
            with redirect_stdout(output):
                for update in updates:
                    await client.session_update("session-1", update)
                client.finish_message()
            outputs.append(output.getvalue())
        return outputs

    async def test_renders_messages_thoughts_replay_tools_and_todos(self) -> None:
        updates = (
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=text_block("answer"),
                message_id="assistant-1",
            ),
            AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=text_block("reasoning"),
                message_id="thought-1",
            ),
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("earlier prompt"),
                message_id="user-1",
            ),
            start_tool_call(
                "read-1",
                "Read README.md",
                kind="read",
                status="pending",
                raw_input={"path": "README.md"},
            ),
            update_tool_call("read-1", status="completed", raw_output="contents"),
            update_tool_call("exec-1", status="failed", raw_output="boom"),
            start_tool_call(
                "edit-1",
                "Edit a.py",
                kind="edit",
                content=[tool_diff_content("a.py", "new text", "old text")],
            ),
            update_plan(
                [
                    plan_entry("Inspect", status="in_progress"),
                    plan_entry("Verify", status="completed"),
                ]
            ),
        )

        for output in await self.render_with_both_clients(*updates):
            self.assertIn("MIRA\nanswer", output)
            self.assertIn("THOUGHT\nreasoning", output)
            self.assertIn("USER [replay]\nearlier prompt", output)
            self.assertIn("TOOL [read]", output)
            self.assertIn("status: completed", output)
            self.assertIn("status: failed", output)
            self.assertIn("diff: a.py", output)
            self.assertIn("new text", output)
            self.assertIn("ACP PLAN / write_todos", output)

    async def test_unknown_updates_and_optional_raw_output_do_not_crash(self) -> None:
        update = CurrentModeUpdate(
            session_update="current_mode_update",
            current_mode_id="plan",
        )
        for client_type in CLIENT_TYPES:
            output = io.StringIO()
            client = client_type()
            client.raw_updates = True
            with redirect_stdout(output):
                await client.session_update("session-1", update)
            self.assertIn("RAW ACP UPDATE", output.getvalue())
            self.assertIn("ACP UPDATE [CurrentModeUpdate]", output.getvalue())

    async def test_permission_selection_returns_exact_option_id(self) -> None:
        for client_type in CLIENT_TYPES:
            client = client_type()
            with redirect_stdout(io.StringIO()):
                with patch("builtins.input", return_value="2"):
                    response = await client.request_permission(
                        session_id="session-1",
                        tool_call=ToolCallUpdate(
                            tool_call_id="call-1",
                            title="Edit file",
                        ),
                        options=[
                            PermissionOption(
                                option_id="approve",
                                name="Approve",
                                kind="allow_once",
                            ),
                            PermissionOption(
                                option_id="reject-exact",
                                name="Reject",
                                kind="reject_once",
                            ),
                        ],
                    )
            self.assertEqual(
                response.model_dump(by_alias=True, exclude_none=True),
                {"outcome": {"outcome": "selected", "optionId": "reject-exact"}},
            )

    async def test_reply_in_chat_returns_only_the_public_option_id(self) -> None:
        for client_type in CLIENT_TYPES:
            output = io.StringIO()
            client = client_type()
            with redirect_stdout(output):
                with patch("builtins.input", return_value="1"):
                    response = await client.request_permission(
                        session_id="session-1",
                        tool_call=ToolCallUpdate(
                            tool_call_id="ask-1",
                            title="Choose an answer",
                        ),
                        options=[
                            PermissionOption(
                                option_id="reply-option",
                                name="Reply in chat",
                                kind="reject_once",
                            )
                        ],
                    )

            self.assertEqual(response.outcome.option_id, "reply-option")
            self.assertIn("Enter your reply as the next message", output.getvalue())

    async def test_invalid_permission_selection_safely_cancels(self) -> None:
        for client_type in CLIENT_TYPES:
            client = client_type()
            with redirect_stdout(io.StringIO()):
                with patch("builtins.input", return_value="invalid"):
                    response = await client.request_permission(
                        session_id="session-1",
                        tool_call=ToolCallUpdate(
                            tool_call_id="call-1",
                            title="Execute",
                        ),
                        options=[
                            PermissionOption(
                                option_id="approve",
                                name="Approve",
                                kind="allow_once",
                            )
                        ],
                    )
            self.assertEqual(
                response.model_dump(by_alias=True, exclude_none=True),
                {"outcome": {"outcome": "cancelled"}},
            )


if __name__ == "__main__":
    unittest.main()
