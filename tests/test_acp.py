"""Product tests for MIRA's direct stock-SDK ACP adapter."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from acp import (
    PROTOCOL_VERSION,
    resource_link_block,
    schema,
    spawn_agent_process,
    text_block,
)
from acp.exceptions import RequestError
from typer.testing import CliRunner

from cli.main import app as cli_app
from mira.api import (
    ApprovalRequest,
    ArtifactDisplayRequest,
    ArtifactReviewRequest,
    AskUserRequest,
    ConfirmationRequest,
    MCPApprovalRequest,
    MessageEvent,
    ToolEvent,
)
from protocols.acp import server as acp_server
from protocols.acp.frontend import InteractionCancelled, ReplyInChat
from protocols.acp.mapping import artifact_text
from protocols.acp.server import MiraAgent


class FakeConnection:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.permissions: list[dict[str, object]] = []
        self.permission_choices: list[str | None] = []
        self.activity: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        del kwargs
        await asyncio.sleep(0)
        self.updates.append((session_id, update))
        self.activity.append(("update", update))

    async def request_permission(self, **kwargs: object) -> object:
        self.permissions.append(dict(kwargs))
        self.activity.append(("permission", dict(kwargs)))
        choice = self.permission_choices.pop(0) if self.permission_choices else "approve"
        if choice is None:
            return SimpleNamespace(outcome=SimpleNamespace(outcome="cancelled"))
        return SimpleNamespace(
            outcome=SimpleNamespace(outcome="selected", option_id=choice)
        )


class FakeSession:
    def __init__(
        self,
        application: "FakeApplication | SimpleNamespace",
        session_id: str,
        transcript: tuple[dict, ...] = (),
    ) -> None:
        self.application = application
        self.id = session_id
        self.mode = "action"
        self.prompts: list[str] = []
        self.answers: list[str] = []
        self.cancelled = 0
        self.closed = 0
        self.transcript = transcript
        self.current_goal = None
        self.current_plan = None

    async def prompt(self, text: str) -> object:
        self.prompts.append(text)
        if text == "ask":
            answer = await self.application.frontend.request(
                AskUserRequest(
                    {
                        "question": "Which approach?",
                        "options": ["A", "B", "Tell MIRA what to do differently"],
                        "open_option": "Tell MIRA what to do differently",
                    }
                )
            )
            self.answers.append(answer)
        else:
            self.application.frontend.emit(
                MessageEvent(
                    session_id=self.id,
                    message_id="reply-1",
                    phase="content",
                    text="reply",
                )
            )
        return SimpleNamespace(final_text="reply")

    async def set_mode(self, mode: str) -> None:
        self.mode = "planning" if mode == "plan" else "action"

    async def cancel(self) -> None:
        self.cancelled += 1

    async def close(self, *, persist: bool = True) -> None:
        del persist
        self.closed += 1

    def snapshot(self) -> object:
        return SimpleNamespace(
            mode=self.mode,
            transcript=self.transcript,
            current_goal=self.current_goal,
            current_plan=self.current_plan,
        )


class FakeApplication:
    def __init__(
        self,
        workspace: Path,
        frontend: object,
        *,
        existing: set[str] | None = None,
        transcripts: dict[str, tuple[dict, ...]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.frontend = frontend
        self.existing = set(existing or ())
        self.transcripts = transcripts or {}
        self.sessions: dict[str, FakeSession] = {}
        self.shutdowns = 0

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.existing

    async def open_session(
        self,
        session_id: str | None = None,
        resume: bool = False,
    ) -> FakeSession:
        del resume
        assert session_id is not None
        session = FakeSession(self, session_id, self.transcripts.get(session_id, ()))
        self.sessions[session_id] = session
        self.existing.add(session_id)
        return session

    async def shutdown(self) -> None:
        self.shutdowns += 1
        for session in self.sessions.values():
            await session.close()


class ACPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.connection = FakeConnection()
        self.server = MiraAgent()
        self.server.on_connect(self.connection)
        await self.server.initialize(1, schema.ClientCapabilities())

    async def asyncTearDown(self) -> None:
        await self.server.frontend.shutdown()

    async def test_implements_stock_agent_without_foreign_server_inheritance(self) -> None:
        response = await self.server.initialize(
            PROTOCOL_VERSION,
            schema.ClientCapabilities(),
        )
        self.assertEqual(MiraAgent.__bases__, (object,))
        self.assertTrue(response.agent_capabilities.load_session)
        self.assertFalse(response.agent_capabilities.prompt_capabilities.image)
        self.assertFalse(response.agent_capabilities.prompt_capabilities.audio)
        self.assertEqual(response.agent_info.name, "mira")

    async def test_initialize_reports_the_supported_protocol_version(self) -> None:
        response = await self.server.initialize(999, schema.ClientCapabilities())

        self.assertEqual(response.protocol_version, PROTOCOL_VERSION)

    async def test_normalized_workspace_reuses_one_application_concurrently(self) -> None:
        applications: list[FakeApplication] = []

        async def start(*, workspace: Path, frontend: object) -> FakeApplication:
            await asyncio.sleep(0)
            application = FakeApplication(workspace, frontend)
            applications.append(application)
            return application

        with tempfile.TemporaryDirectory() as directory, patch(
            "protocols.acp.server.MiraApplication.start",
            new=AsyncMock(side_effect=start),
        ):
            first = await self.server.new_session(directory)
            second = await self.server.new_session(str(Path(directory) / "."))
            await asyncio.gather(
                self.server.prompt(first.session_id, [text_block("one")]),
                self.server.prompt(second.session_id, [text_block("two")]),
            )

        self.assertEqual(len(applications), 1)
        self.assertEqual(applications[0].sessions[first.session_id].prompts, ["one"])
        self.assertEqual(applications[0].sessions[second.session_id].prompts, ["two"])

    async def test_different_workspaces_stay_isolated(self) -> None:
        applications: list[FakeApplication] = []

        async def start(*, workspace: Path, frontend: object) -> FakeApplication:
            application = FakeApplication(workspace, frontend)
            applications.append(application)
            return application

        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
            patch(
                "protocols.acp.server.MiraApplication.start",
                new=AsyncMock(side_effect=start),
            ),
        ):
            first = await self.server.new_session(first_dir)
            second = await self.server.new_session(second_dir)
            await self.server.prompt(first.session_id, [text_block("one")])
            await self.server.prompt(second.session_id, [text_block("two")])
        self.assertEqual(len(applications), 2)

    async def test_prompt_and_mode_switch_use_public_mira_session_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            application = FakeApplication(Path(directory).resolve(), self.server.frontend)
            with patch(
                "protocols.acp.server.MiraApplication.start",
                new=AsyncMock(return_value=application),
            ):
                response = await self.server.new_session(directory)
                config_response = await self.server.set_config_option(
                    "mode",
                    response.session_id,
                    "plan",
                )
                session = application.sessions[response.session_id]
                await self.server.prompt(response.session_id, [text_block("plan this")])

        self.assertEqual(session.mode, "planning")
        self.assertEqual(session.prompts, ["plan this"])
        self.assertEqual(self.server._session_modes[response.session_id], "plan")
        self.assertEqual(config_response.config_options[0].current_value, "plan")

    async def test_load_replays_authoritative_transcript_without_formal_acp_plan(self) -> None:
        transcript = (
            {"id": 1, "type": "user", "text": "hello"},
            {"id": 2, "type": "reasoning", "text": "think"},
            {"id": 3, "type": "assistant", "text": "world"},
            {
                "id": 4,
                "type": "plan",
                "plan": {
                    "title": "Formal",
                    "objective": "Keep semantics",
                    "key_changes": ["One"],
                },
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            application = FakeApplication(
                Path(directory).resolve(),
                self.server.frontend,
                existing={"saved"},
                transcripts={"saved": transcript},
            )
            with patch(
                "protocols.acp.server.MiraApplication.start",
                new=AsyncMock(return_value=application),
            ):
                await self.server.load_session(directory, "saved")

        updates = [update for _, update in self.connection.updates]
        self.assertTrue(any(isinstance(update, schema.UserMessageChunk) for update in updates))
        self.assertTrue(any(isinstance(update, schema.AgentThoughtChunk) for update in updates))
        self.assertFalse(any(isinstance(update, schema.AgentPlanUpdate) for update in updates))
        self.assertIn("MIRA Plan", "\n".join(str(update) for update in updates))

    async def test_load_unknown_is_resource_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            application = FakeApplication(Path(directory).resolve(), self.server.frontend)
            with patch(
                "protocols.acp.server.MiraApplication.start",
                new=AsyncMock(return_value=application),
            ):
                with self.assertRaises(RequestError):
                    await self.server.load_session(directory, "unknown")
        self.assertNotIn("unknown", application.sessions)

    async def test_rejects_rich_prompts_and_client_owned_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RequestError):
                await self.server.new_session(
                    directory,
                    additional_directories=["elsewhere"],
                )
            with self.assertRaises(RequestError):
                await self.server.new_session(directory, mcp_servers=[object()])
        with self.assertRaises(RequestError):
            await self.server.prompt("missing", [])
        with self.assertRaises(RequestError) as raised:
            await self.server.prompt(
                "missing",
                [resource_link_block("file:///not-text", "not text")],
            )
        self.assertEqual(raised.exception.code, -32602)
        self.assertIn("text content only", str(raised.exception.data))

    async def test_reply_in_chat_ends_turn_without_becoming_model_input(self) -> None:
        application = SimpleNamespace(frontend=self.server.frontend)
        session = FakeSession(application, "chat")
        self.server._mira_sessions["chat"] = session
        self.connection.permission_choices = ["reply-in-chat"]

        response = await self.server.prompt("chat", [text_block("ask")])
        next_response = await self.server.prompt("chat", [text_block("free text answer")])

        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(next_response.stop_reason, "end_turn")
        self.assertEqual(session.prompts, ["ask", "free text answer"])
        self.assertEqual(session.answers, [])

    async def test_cancel_targets_only_requested_session_and_shutdown_cleans_apps(self) -> None:
        first = FakeSession(SimpleNamespace(frontend=self.server.frontend), "first")
        second = FakeSession(SimpleNamespace(frontend=self.server.frontend), "second")
        self.server._mira_sessions.update(first=first, second=second)
        await asyncio.gather(self.server.cancel("first"), self.server.cancel("missing"))
        self.assertEqual(first.cancelled, 1)
        self.assertEqual(second.cancelled, 0)

        first_app = FakeApplication(Path("one"), self.server.frontend)
        second_app = FakeApplication(Path("two"), self.server.frontend)
        self.server._applications = {"one": first_app, "two": second_app}
        await self.server.shutdown()
        self.assertEqual((first_app.shutdowns, second_app.shutdowns), (1, 1))


class ACPFrontendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.connection = FakeConnection()
        self.server = MiraAgent()
        self.server.on_connect(self.connection)
        self.frontend = self.server.frontend

    async def asyncTearDown(self) -> None:
        await self.frontend.shutdown()

    async def test_ordered_messages_reasoning_tools_results_failures_and_diffs(self) -> None:
        session_id = "ordered"
        self.frontend.emit(MessageEvent(session_id=session_id, phase="content", text="A"))
        self.frontend.emit(MessageEvent(session_id=session_id, phase="reasoning", text="B"))
        self.frontend.emit(
            ToolEvent(
                session_id=session_id,
                phase="start",
                name="edit_file",
                tool_call_id="edit-1",
                arguments={"file_path": "a.py", "old_string": "a", "new_string": "b"},
            )
        )
        self.frontend.emit(
            ToolEvent(
                session_id=session_id,
                phase="completed_result",
                name="edit_file",
                tool_call_id="edit-1",
                result="done",
            )
        )
        self.frontend.emit(
            ToolEvent(
                session_id=session_id,
                phase="start",
                name="execute",
                tool_call_id="exec-1",
                arguments={"command": "pytest"},
            )
        )
        self.frontend.emit(
            ToolEvent(
                session_id=session_id,
                phase="completed_error",
                name="execute",
                tool_call_id="exec-1",
                result="failed",
            )
        )
        await self.frontend.flush(session_id)

        updates = [update for _, update in self.connection.updates]
        self.assertEqual(
            [update.session_update for update in updates],
            [
                "agent_message_chunk",
                "agent_thought_chunk",
                "tool_call",
                "tool_call_update",
                "tool_call",
                "tool_call_update",
            ],
        )
        self.assertEqual(updates[2].kind, "edit")
        self.assertEqual(updates[2].content[0].type, "diff")
        self.assertEqual(updates[3].status, "completed")
        self.assertEqual(updates[5].status, "failed")
        self.assertIn("**Command:**", updates[5].content[0].content.text)

    async def test_tool_titles_cover_stock_and_generic_mira_tools(self) -> None:
        cases = [
            ("read_file", {"file_path": "a.py"}, "read"),
            ("write_file", {"file_path": "a.py", "content": "x"}, "edit"),
            ("ls", {"path": "."}, "search"),
            ("glob", {"pattern": "*.py"}, "search"),
            ("grep", {"pattern": "needle"}, "search"),
            ("custom_mcp_tool", {"value": 1}, "other"),
        ]
        for index, (name, arguments, _kind) in enumerate(cases):
            self.frontend.emit(
                ToolEvent(
                    session_id="tools",
                    phase="start",
                    name=name,
                    tool_call_id=f"call-{index}",
                    arguments=arguments,
                )
            )
        await self.frontend.flush("tools")
        updates = [update for _, update in self.connection.updates]
        self.assertEqual([update.kind for update in updates], [case[2] for case in cases])
        self.assertTrue(all(update.title for update in updates))
        self.assertEqual(updates[1].content[0].type, "diff")

    async def test_only_write_todos_projects_to_acp_plan(self) -> None:
        self.frontend.emit(
            ToolEvent(
                session_id="todos",
                phase="start",
                name="write_todos",
                tool_call_id="todos-1",
                arguments={"todos": [{"content": "Test", "status": "in_progress"}]},
            )
        )
        await self.frontend.flush("todos")
        plans = [
            update
            for _, update in self.connection.updates
            if isinstance(update, schema.AgentPlanUpdate)
        ]
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].entries[0].content, "Test")
        self.assertEqual(plans[0].entries[0].status, "in_progress")

    async def test_approval_flushes_and_preserves_native_tool_call_ids(self) -> None:
        self.connection.permission_choices = ["approve", "reject"]
        self.frontend.emit(
            ToolEvent(
                session_id="hitl",
                phase="start",
                name="write_file",
                tool_call_id="native-7",
                arguments={"file_path": "x.txt", "content": "x"},
            )
        )
        request = ApprovalRequest(
            (
                {
                    "action_requests": [
                        {"name": "write_file", "args": {"file_path": "x.txt"}},
                        {"name": "execute", "args": {"command": "pytest"}, "id": "exec-2"},
                    ]
                },
            )
        )
        with self.frontend.bind("hitl"):
            decisions = await self.frontend.request(request)

        self.assertEqual(decisions, [{"type": "approve"}, {"type": "reject"}])
        self.assertEqual(self.connection.permissions[0]["tool_call"].tool_call_id, "native-7")
        self.assertEqual(self.connection.permissions[1]["tool_call"].tool_call_id, "exec-2")
        self.assertEqual(self.connection.activity[0][0], "update")
        self.assertEqual(self.connection.activity[1][0], "permission")

    async def test_ask_user_has_suggestions_and_open_ended_button(self) -> None:
        self.connection.permission_choices = ["choice-1"]
        request = AskUserRequest(
            {
                "question": "Which approach?",
                "options": ["A", "B", "Tell MIRA what to do differently"],
                "open_option": "Tell MIRA what to do differently",
            }
        )
        with self.frontend.bind("ask"):
            answer = await self.frontend.request(request)

        self.assertEqual(answer, "B")
        self.assertEqual([kind for kind, _ in self.connection.activity], ["permission"])
        self.assertEqual(
            self.connection.permissions[0]["tool_call"].title,
            "Which approach?",
        )
        self.assertEqual(
            [option.name for option in self.connection.permissions[0]["options"]],
            ["A", "B", "Reply in chat"],
        )

    async def test_open_reply_and_cancel_never_return_fabricated_user_text(self) -> None:
        request = AskUserRequest(
            {
                "question": "Value?",
                "options": ["A", "Tell MIRA what to do differently"],
                "open_option": "Tell MIRA what to do differently",
            }
        )
        self.connection.permission_choices = ["reply-in-chat", None]
        with self.frontend.bind("declined"):
            with self.assertRaises(ReplyInChat):
                await self.frontend.request(request)
            with self.assertRaises(InteractionCancelled):
                await self.frontend.request(request)

    async def test_artifacts_flush_as_messages_before_stable_review_buttons(self) -> None:
        self.frontend.emit(
            MessageEvent(session_id="artifact", phase="reasoning", text="thought")
        )
        self.frontend.emit(
            ToolEvent(
                session_id="artifact",
                phase="start",
                name="finalize_plan",
                tool_call_id="plan-1",
                arguments={"title": "Plan"},
            )
        )
        self.frontend.emit(
            ToolEvent(
                session_id="artifact",
                phase="completed_result",
                name="read_file",
                tool_call_id="read-1",
                result="done",
            )
        )
        plan = {"title": "Plan", "objective": "Ship it", "key_changes": ["One"]}
        self.connection.permission_choices = ["close"]
        with self.frontend.bind("artifact"):
            result = await self.frontend.request(ArtifactReviewRequest("plan", {}, plan))

        self.assertEqual(result, {"action": "close"})
        self.assertEqual(
            [kind for kind, _ in self.connection.activity[-3:]],
            ["update", "update", "permission"],
        )
        finalizer_update = self.connection.activity[-3][1]
        self.assertIsInstance(finalizer_update, schema.ToolCallProgress)
        self.assertEqual(finalizer_update.tool_call_id, "plan-1")
        self.assertEqual(finalizer_update.status, "completed")
        artifact_update = self.connection.activity[-2][1]
        self.assertIsInstance(artifact_update, schema.AgentMessageChunk)
        self.assertEqual(artifact_update.content.text, artifact_text("plan", plan))
        self.assertTrue(artifact_update.message_id)
        self.assertEqual(
            [option.name for option in self.connection.permissions[0]["options"]],
            ["Implement", "Keep", "Revise in chat"],
        )
        self.assertFalse(
            any(
                isinstance(update, schema.AgentPlanUpdate)
                for _, update in self.connection.updates
            )
        )

        update_count = len(self.connection.updates)
        self.frontend.emit(
            ToolEvent(
                session_id="artifact",
                phase="completed_result",
                name="finalize_plan",
                tool_call_id="plan-1",
                result=plan,
            )
        )
        await self.frontend.flush("artifact")
        self.assertEqual(len(self.connection.updates), update_count)

    async def test_goal_revision_in_chat_uses_turn_boundary(self) -> None:
        self.connection.permission_choices = ["reply-in-chat"]
        with self.frontend.bind("goal"):
            with self.assertRaises(ReplyInChat):
                await self.frontend.request(
                    ArtifactReviewRequest(
                        "goal",
                        {},
                        {"title": "Goal", "objective": "Ship"},
                    )
                )

    async def test_mcp_confirmation_and_retained_artifact_display(self) -> None:
        artifact = {"title": "Retained", "objective": "Keep it"}
        self.server._mira_sessions["buttons"] = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(current_goal=artifact, current_plan=None)
        )
        self.connection.permission_choices = ["always_allow", "confirm"]
        with self.frontend.bind("buttons"):
            mcp = await self.frontend.request(MCPApprovalRequest(object(), "Connect docs"))
            confirmed = await self.frontend.request(
                ConfirmationRequest("create_git_repo", "Create a repository?")
            )
            displayed = await self.frontend.request(ArtifactDisplayRequest("goal"))

        self.assertEqual(mcp, "always_allow")
        self.assertTrue(confirmed)
        self.assertEqual(displayed, "Displayed retained goal.")
        self.assertIn("MIRA Goal", self.connection.updates[-1][1].content.text)


class ACPWiringTests(unittest.TestCase):
    def test_cli_dispatches_acp_lazily(self) -> None:
        with patch("protocols.acp.run_server") as run_server:
            result = CliRunner().invoke(cli_app, ["--acp"])
        self.assertEqual(result.exit_code, 0, result.output)
        run_server.assert_called_once_with()

    def test_cli_dispatches_acp_http_for_loopback_listen(self) -> None:
        with patch("protocols.acp.run_server") as run_server:
            result = CliRunner().invoke(
                cli_app,
                ["--acp", "--listen", "127.0.0.1:8765"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        run_server.assert_called_once_with(listen="127.0.0.1:8765")

    def test_listen_requires_acp(self) -> None:
        result = CliRunner().invoke(cli_app, ["--listen", "127.0.0.1:8765"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--listen requires --acp", result.output)

    def test_listen_rejects_malformed_and_non_loopback_addresses(self) -> None:
        malformed = CliRunner().invoke(cli_app, ["--acp", "--listen", "localhost"])
        exposed = CliRunner().invoke(cli_app, ["--acp", "--listen", "0.0.0.0:8765"])

        self.assertNotEqual(malformed.exit_code, 0)
        self.assertIn("HOST:PORT", malformed.output)
        self.assertNotEqual(exposed.exit_code, 0)
        self.assertIn("loopback", exposed.output)

    def test_cli_reports_missing_stock_sdk(self) -> None:
        missing = ModuleNotFoundError("No module named 'acp'")
        missing.name = "acp"
        with patch("protocols.acp.run_server", side_effect=missing):
            result = CliRunner().invoke(cli_app, ["--acp"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Install MIRA with the 'acp' extra", result.output)

    def test_cli_reports_missing_http_extra(self) -> None:
        missing = ModuleNotFoundError("No module named 'hypercorn'")
        missing.name = "hypercorn"
        with patch("protocols.acp.run_server", side_effect=missing):
            result = CliRunner().invoke(
                cli_app,
                ["--acp", "--listen", "127.0.0.1:8765"],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Install MIRA with the 'acp-http' extra", result.output)

    def test_architecture_uses_only_stock_stable_acp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (root / "protocols" / "acp").glob("*.py")
        )
        self.assertIn('"agent-client-protocol==0.12.1"', project)
        self.assertIn('"agent-client-protocol[http]==0.12.1"', project)
        self.assertIn('"hypercorn>=0.17"', project)
        self.assertNotIn("deepagents-acp", project.lower())
        self.assertNotIn("deepagents_acp", sources)
        self.assertNotIn("AgentServerACP", sources)
        self.assertNotIn("create_elicitation", sources)
        self.assertNotIn("use_unstable_protocol", sources)
        self.assertIn("await run_agent(server)", sources)
        stdio_source = (root / "protocols" / "acp" / "server.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("acp.http", stdio_source)
        self.assertNotIn("hypercorn", stdio_source)
        http_source = (root / "protocols" / "acp" / "http.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("create_asgi_app", http_source)
        self.assertNotIn("numpy", http_source)

    def test_example_imports_without_http_dependencies_or_starting_mira(self) -> None:
        root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(root / "examples" / "acp_client.py"),
            run_name="acp_client_example",
        )

        self.assertTrue(callable(namespace["main"]))
        self.assertTrue(callable(namespace["run_stdio"]))
        self.assertTrue(callable(namespace["run_http"]))


class ACPStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_windows_preloads_numpy_before_stdio_runner(self) -> None:
        events: list[str] = []
        original_import = __import__

        def tracked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                events.append("numpy")
                return object()
            return original_import(name, *args, **kwargs)

        async def run_agent(_server: object) -> None:
            events.append("run_agent")

        async def shutdown() -> None:
            events.append("shutdown")

        server = SimpleNamespace(shutdown=shutdown)
        with (
            patch.object(acp_server.sys, "platform", "win32"),
            patch("builtins.__import__", new=tracked_import),
            patch.object(acp_server, "MiraAgent", return_value=server),
            patch.object(acp_server, "run_agent", new=run_agent),
        ):
            await acp_server.serve()
        self.assertEqual(events, ["numpy", "run_agent", "shutdown"])

    async def test_non_windows_does_not_preload_numpy(self) -> None:
        events: list[str] = []
        original_import = __import__

        def tracked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                events.append("numpy")
                return object()
            return original_import(name, *args, **kwargs)

        async def run_agent(_server: object) -> None:
            events.append("run_agent")

        async def shutdown() -> None:
            events.append("shutdown")

        server = SimpleNamespace(shutdown=shutdown)
        with (
            patch.object(acp_server.sys, "platform", "linux"),
            patch("builtins.__import__", new=tracked_import),
            patch.object(acp_server, "MiraAgent", return_value=server),
            patch.object(acp_server, "run_agent", new=run_agent),
        ):
            await acp_server.serve()
        self.assertEqual(events, ["run_agent", "shutdown"])


class ACPStdioSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdout_contains_only_json_rpc_frames(self) -> None:
        root = Path(__file__).resolve().parents[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "cli.main",
            "--acp",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {},
            },
        }
        stdout, _stderr = await asyncio.wait_for(
            process.communicate((json.dumps(request) + "\n").encode()),
            timeout=30,
        )

        lines = stdout.decode().splitlines()
        self.assertEqual(len(lines), 1, stdout.decode(errors="replace"))
        response = json.loads(lines[0])
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], 0)
        self.assertEqual(response["result"]["protocolVersion"], PROTOCOL_VERSION)

    async def test_stock_client_initializes_real_stdio_runner_and_creates_session(self) -> None:
        client = SimpleNamespace(on_connect=lambda connection: None)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as workspace:
            async with spawn_agent_process(
                client,
                sys.executable,
                "-m",
                "cli.main",
                "--acp",
                cwd=root,
            ) as (connection, _process):
                initialized = await asyncio.wait_for(
                    connection.initialize(
                        PROTOCOL_VERSION,
                        schema.ClientCapabilities(),
                    ),
                    timeout=30,
                )
                created = await asyncio.wait_for(
                    connection.new_session(workspace),
                    timeout=30,
                )

        self.assertEqual(initialized.protocol_version, PROTOCOL_VERSION)
        self.assertTrue(initialized.agent_capabilities.load_session)
        self.assertTrue(created.session_id)
        self.assertEqual(created.modes.current_mode_id, "act")


if __name__ == "__main__":
    unittest.main()
