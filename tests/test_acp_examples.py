"""Focused tests for the transport-independent ACP reference console."""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace

from acp import (
    plan_entry,
    start_tool_call,
    text_block,
    tool_content,
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

from examples.acp.client_common import (
    ReferenceClient,
    ReferenceConsole,
    UpdateRenderer,
)


class FakeConnection:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, list[object]]] = []
        self.new_sessions = 0
        self.loaded_sessions: list[tuple[str, str]] = []
        self.mode_changes: list[tuple[str, str]] = []
        self.cancellations: list[str] = []

    async def prompt(self, session_id: str, prompt: list[object]) -> object:
        self.prompts.append((session_id, prompt))
        return SimpleNamespace(stop_reason="end_turn")

    async def new_session(self, workspace: str) -> object:
        del workspace
        self.new_sessions += 1
        return SimpleNamespace(
            session_id=f"new-{self.new_sessions}",
            modes=SimpleNamespace(current_mode_id="act"),
        )

    async def load_session(self, workspace: str, session_id: str) -> object:
        self.loaded_sessions.append((workspace, session_id))
        return SimpleNamespace(modes=SimpleNamespace(current_mode_id="plan"))

    async def set_session_mode(self, session_id: str, mode_id: str) -> object:
        self.mode_changes.append((session_id, mode_id))
        return object()

    async def cancel(self, session_id: str) -> None:
        self.cancellations.append(session_id)


class UpdateRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.renderer = UpdateRenderer(output=self.output)

    def rendered(self) -> str:
        self.renderer.finish_stream()
        return self.output.getvalue()

    def test_renders_agent_thought_and_replayed_user_messages(self) -> None:
        self.renderer.render(
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=text_block("answer"),
                message_id="assistant-1",
            )
        )
        self.renderer.render(
            AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=text_block("reasoning"),
                message_id="thought-1",
            )
        )
        self.renderer.render(
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("earlier prompt"),
                message_id="user-1",
            )
        )

        output = self.rendered()
        self.assertIn("MIRA\nanswer", output)
        self.assertIn("THOUGHT\nreasoning", output)
        self.assertIn("USER [replay]\nearlier prompt", output)

    def test_renders_tool_start_completion_failure_and_diff(self) -> None:
        self.renderer.render(
            start_tool_call(
                "read-1",
                "Read `README.md`",
                kind="read",
                status="pending",
                raw_input={"path": "README.md"},
            )
        )
        self.renderer.render(
            update_tool_call(
                "read-1",
                status="completed",
                content=[tool_content(text_block("contents"))],
                raw_output="contents",
            )
        )
        self.renderer.render(
            update_tool_call(
                "exec-1",
                status="failed",
                raw_output={"error": "boom"},
            )
        )
        self.renderer.render(
            start_tool_call(
                "edit-1",
                "Edit `a.py`",
                kind="edit",
                status="pending",
                content=[tool_diff_content("a.py", "new text", "old text")],
            )
        )

        output = self.rendered()
        self.assertIn("TOOL [read]", output)
        self.assertIn("id: read-1", output)
        self.assertIn("TOOL RESULT", output)
        self.assertIn("status: completed", output)
        self.assertIn("status: failed", output)
        self.assertIn("diff: a.py", output)
        self.assertIn("new text", output)

    def test_renders_agent_plan_unknown_updates_and_raw_objects(self) -> None:
        self.renderer.raw_enabled = True
        self.renderer.render(
            update_plan(
                [
                    plan_entry("Inspect", status="in_progress"),
                    plan_entry("Verify", status="completed"),
                ]
            )
        )
        self.renderer.render(
            CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id="plan",
            )
        )

        output = self.rendered()
        self.assertIn("RAW ACP UPDATE [AgentPlanUpdate]", output)
        self.assertIn('"sessionUpdate": "plan"', output)
        self.assertIn("ACP PLAN / write_todos", output)
        self.assertIn("[>] Inspect", output)
        self.assertIn("[x] Verify", output)
        self.assertIn("ACP UPDATE [CurrentModeUpdate]", output)


class ReferenceClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_permission_mode_shows_public_option_fields(self) -> None:
        output = io.StringIO()
        renderer = UpdateRenderer(output=output)
        renderer.raw_enabled = True
        client = ReferenceClient(renderer, input_reader=lambda _prompt: "")

        await client.request_permission(
            session_id="session-1",
            tool_call=ToolCallUpdate(tool_call_id="call-1", title="Execute"),
            options=[
                PermissionOption(
                    option_id="reject-exact",
                    name="Reject",
                    kind="reject_once",
                )
            ],
        )

        rendered = output.getvalue()
        self.assertIn("RAW ACP PERMISSION", rendered)
        self.assertIn('"optionId": "reject-exact"', rendered)

    async def test_permission_selection_returns_the_exact_option_id(self) -> None:
        output = io.StringIO()
        client = ReferenceClient(
            UpdateRenderer(output=output),
            input_reader=lambda _prompt: "2",
        )
        response = await client.request_permission(
            session_id="session-1",
            tool_call=ToolCallUpdate(tool_call_id="call-1", title="Edit File"),
            options=[
                PermissionOption(
                    option_id="approve",
                    name="Approve",
                    kind="allow_once",
                ),
                PermissionOption(
                    option_id="reject",
                    name="Reject",
                    kind="reject_once",
                ),
            ],
        )

        self.assertEqual(
            response,
            {"outcome": {"outcome": "selected", "optionId": "reject"}},
        )
        self.assertIn("PERMISSION REQUIRED", output.getvalue())

    async def test_reply_in_chat_returns_only_its_option_id(self) -> None:
        output = io.StringIO()
        client = ReferenceClient(
            UpdateRenderer(output=output),
            input_reader=lambda _prompt: "1",
        )

        response = await client.request_permission(
            session_id="session-1",
            tool_call=ToolCallUpdate(tool_call_id="ask-1", title="Choose an answer"),
            options=[
                PermissionOption(
                    option_id="reply-option",
                    name="Reply in chat",
                    kind="reject_once",
                )
            ],
        )

        self.assertEqual(
            response,
            {"outcome": {"outcome": "selected", "optionId": "reply-option"}},
        )
        self.assertIn("Enter your response as the next message", output.getvalue())

    async def test_invalid_permission_selection_safely_cancels(self) -> None:
        client = ReferenceClient(
            UpdateRenderer(output=io.StringIO()),
            input_reader=lambda _prompt: "not a choice",
        )
        response = await client.request_permission(
            session_id="session-1",
            tool_call=ToolCallUpdate(tool_call_id="call-1", title="Execute"),
            options=[
                PermissionOption(option_id="approve", name="Approve", kind="allow_once")
            ],
        )

        self.assertEqual(response, {"outcome": {"outcome": "cancelled"}})


class ReferenceConsoleTests(unittest.IsolatedAsyncioTestCase):
    def make_console(
        self,
        *,
        allow_load: bool = True,
    ) -> tuple[ReferenceConsole, FakeConnection, io.StringIO]:
        connection = FakeConnection()
        output = io.StringIO()
        renderer = UpdateRenderer(output=output)
        client = ReferenceClient(renderer, input_reader=lambda _prompt: "")
        console = ReferenceConsole(
            connection=connection,
            client=client,
            workspace=Path("workspace").resolve(),
            session_id="session-1",
            mode="act",
            transport_name="stdio" if allow_load else "HTTP",
            allow_load=allow_load,
            output=output,
            input_reader=lambda _prompt: "",
        )
        return console, connection, output

    async def test_reuses_one_session_across_prompts(self) -> None:
        console, connection, _output = self.make_console()

        await console.send_prompt("first")
        await console.send_prompt("second")

        self.assertEqual(
            [session_id for session_id, _prompt in connection.prompts],
            ["session-1", "session-1"],
        )

    async def test_mode_and_stdio_load_commands_use_real_acp_methods(self) -> None:
        console, connection, _output = self.make_console()

        await console.handle_command("/mode plan")
        await console.handle_command("/load saved-session")

        self.assertEqual(connection.mode_changes, [("session-1", "plan")])
        self.assertEqual(connection.loaded_sessions[0][1], "saved-session")
        self.assertEqual(console.session_id, "saved-session")
        self.assertEqual(console.mode, "plan")

    async def test_http_load_command_reports_replay_limitation_safely(self) -> None:
        console, connection, output = self.make_console(allow_load=False)

        await console.handle_command("/load saved-session")

        self.assertEqual(connection.loaded_sessions, [])
        self.assertIn("HTTP session/load is replay-only", output.getvalue())

    async def test_cancel_after_uses_the_current_session(self) -> None:
        console, connection, _output = self.make_console()

        await console.handle_command("/cancel-after 0 keep this prompt intact")

        self.assertEqual(connection.prompts[0][0], "session-1")
        self.assertEqual(connection.cancellations, ["session-1"])


if __name__ == "__main__":
    unittest.main()
