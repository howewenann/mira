"""Real loopback tests for MIRA's experimental ACP Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import socket
import tempfile
import unittest
import warnings
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from acp import (
    PROTOCOL_VERSION,
    connect_to_agent,
    resource_link_block,
    schema,
    text_block,
)
from acp.exceptions import RequestError
from acp.http import create_http_stream
from rich.console import Console

from config.branding import VERSION, blocky_wordmark
from mira.api import (
    ApprovalRequest,
    ArtifactReviewRequest,
    AskUserRequest,
    MessageEvent,
    ToolEvent,
)
from protocols.acp.http.server import MiraAgentFactory, _server_config, serve_http
from protocols.acp.http.splash import http_splash, print_http_splash


class RecordingClient:
    def __init__(self, *permission_choices: str | None) -> None:
        self.connection = None
        self.updates: list[tuple[str, object]] = []
        self.permissions: list[dict[str, object]] = []
        self.permission_choices = list(permission_choices)

    def on_connect(self, connection: object) -> None:
        self.connection = connection

    async def session_update(
        self,
        session_id: str,
        update: object,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.updates.append((session_id, update))

    async def request_permission(self, **kwargs: object) -> object:
        self.permissions.append(dict(kwargs))
        choice = self.permission_choices.pop(0) if self.permission_choices else None
        if choice is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {
            "outcome": {
                "outcome": "selected",
                "optionId": choice,
            }
        }


class PersistentState:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}
        self.prompt_started = asyncio.Event()
        self.prompt_cancelled = asyncio.Event()


class HttpFakeSession:
    def __init__(
        self,
        application: "HttpFakeApplication",
        session_id: str,
        state: dict[str, object],
    ) -> None:
        self.application = application
        self.id = session_id
        self.state = state

    async def prompt(self, text: str) -> object:
        prompts = self.state.setdefault("prompts", [])
        assert isinstance(prompts, list)
        prompts.append(text)
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
            answers = self.state.setdefault("answers", [])
            assert isinstance(answers, list)
            answers.append(answer)
            reply = f"answer: {answer}"
        elif text == "approve":
            decisions = await self.application.frontend.request(
                ApprovalRequest(
                    (
                        {
                            "action_requests": [
                                {"name": "write_file", "args": {"file_path": "x.txt"}}
                            ]
                        },
                    )
                )
            )
            self.state["decisions"] = decisions
            reply = "permission handled"
        elif text == "goal":
            goal = {"title": "Goal", "objective": "Preserve semantics"}
            self.application.frontend.emit(
                ToolEvent(
                    session_id=self.id,
                    phase="start",
                    name="finalize_goal",
                    tool_call_id="goal-finalizer",
                    arguments=goal,
                )
            )
            self.state["goal_review"] = await self.application.frontend.request(
                ArtifactReviewRequest("goal", {}, goal)
            )
            reply = "goal reviewed"
        elif text == "block":
            self.application.store.prompt_started.set()
            await self.application.store.prompt_cancelled.wait()
            raise asyncio.CancelledError
        else:
            reply = f"reply: {text}"
        transcript = self.state.setdefault("transcript", [])
        assert isinstance(transcript, list)
        transcript.extend(
            [
                {"type": "user", "text": text},
                {"type": "assistant", "text": reply},
            ]
        )
        self.application.frontend.emit(
            MessageEvent(
                session_id=self.id,
                message_id=f"reply-{len(prompts)}",
                phase="content",
                text=reply,
            )
        )
        return SimpleNamespace(final_text=reply)

    async def set_mode(self, mode: str) -> None:
        self.state["mode"] = "planning" if mode == "plan" else "action"

    async def cancel(self) -> None:
        self.application.store.prompt_cancelled.set()

    async def close(self, *, persist: bool = True) -> None:
        del persist

    def snapshot(self) -> object:
        return SimpleNamespace(
            mode=self.state.get("mode", "action"),
            transcript=tuple(self.state.get("transcript", ())),
            current_goal=None,
            current_plan=None,
        )


class HttpFakeApplication:
    def __init__(self, workspace: Path, frontend: object, store: PersistentState) -> None:
        self.workspace = workspace
        self.frontend = frontend
        self.store = store
        self.sessions: dict[str, HttpFakeSession] = {}
        self.shutdowns = 0

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.store.sessions

    async def open_session(
        self,
        session_id: str | None = None,
        resume: bool = False,
    ) -> HttpFakeSession:
        del resume
        assert session_id is not None
        state = self.store.sessions.setdefault(session_id, {"mode": "action"})
        session = HttpFakeSession(self, session_id, state)
        self.sessions[session_id] = session
        return session

    async def shutdown(self) -> None:
        self.shutdowns += 1
        for session in self.sessions.values():
            await session.close()


async def wait_for_server(port: int) -> None:
    loop = asyncio.get_running_loop()
    for _ in range(100):
        sock = socket.socket()
        sock.setblocking(False)
        try:
            await loop.sock_connect(sock, ("127.0.0.1", port))
        except OSError:
            await asyncio.sleep(0.02)
        else:
            return
        finally:
            sock.close()
    raise TimeoutError("ACP HTTP server did not start")


async def connect(url: str, client: RecordingClient) -> tuple[object, object]:
    transport = create_http_stream(url)
    connection = connect_to_agent(client, transport)
    await connection.initialize(PROTOCOL_VERSION, schema.ClientCapabilities())
    return connection, transport


class ACPHttpTests(unittest.IsolatedAsyncioTestCase):
    def test_http_splash_is_centered_bounded_and_server_only(self) -> None:
        output = StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        console.print(http_splash("127.0.0.1:9000", terminal_width=console.width))
        rendered = output.getvalue()

        for logo_line in blocky_wordmark().splitlines():
            self.assertIn(logo_line.strip(), rendered)
        self.assertIn(VERSION, rendered)
        self.assertIn("ACP Streamable HTTP", rendered)
        self.assertIn("http://127.0.0.1:9000/acp", rendered)
        self.assertIn("loopback only", rendered)
        self.assertIn("ready", rendered)
        self.assertIn("Ctrl+C to stop", rendered)
        self.assertNotIn("session", rendered.lower())
        self.assertNotIn("model", rendered.lower())
        self.assertNotIn("workspace", rendered.lower())
        self.assertNotIn("/help", rendered)
        self.assertNotIn("Alt+Q", rendered)

        border = next(
            line for line in rendered.splitlines() if "┌" in line or "╭" in line
        )
        self.assertTrue(border.startswith(" " * 20))
        self.assertLessEqual(len(border), 120)
        self.assertEqual(len(rendered.splitlines()), 20)
        self.assertTrue(
            all(line == line.rstrip() for line in rendered.splitlines())
        )

    def test_http_splash_handles_a_narrow_terminal(self) -> None:
        output = StringIO()
        console = Console(file=output, width=32, force_terminal=False)

        console.print(http_splash("127.0.0.1:9000", terminal_width=console.width))

        rendered = output.getvalue()
        self.assertIn("transport", rendered)
        self.assertIn("Streamable", rendered)
        self.assertIn("ready", rendered)
        self.assertTrue(all(len(line) <= 32 for line in rendered.splitlines()))
        self.assertEqual(len(rendered.splitlines()), 17)

    def test_http_splash_has_three_rows_of_spacing_above_and_below(self) -> None:
        output = StringIO()
        console = Console(file=output, width=100, force_terminal=False)

        with patch("protocols.acp.http.splash.Console", return_value=console):
            print_http_splash("127.0.0.1:9000")

        lines = output.getvalue().splitlines()
        self.assertEqual(lines[:3], ["", "", ""])
        self.assertEqual(lines[-3:], ["", "", ""])

    async def test_native_hypercorn_info_is_emitted_below_the_panel(self) -> None:
        config = _server_config("127.0.0.1:9000")
        logger = config.log
        events: list[str] = []

        with (
            patch(
                "protocols.acp.http.server.print_http_splash",
                side_effect=lambda _listen: events.append("splash"),
            ),
            patch.object(
                logger.error_logger,
                "info",
                side_effect=lambda _message: events.append("hypercorn-info"),
            ),
        ):
            await logger.info(
                "Running on http://127.0.0.1:9000 (CTRL + C to quit)"
            )

        self.assertEqual(events, ["splash", "hypercorn-info"])

    async def test_splash_failure_does_not_interfere_with_hypercorn(self) -> None:
        config = _server_config("127.0.0.1:9000")
        logger = config.log

        with (
            patch(
                "protocols.acp.http.server.print_http_splash",
                side_effect=UnicodeError,
            ),
            patch.object(logger.error_logger, "info") as hypercorn_info,
        ):
            await logger.info(
                "Running on http://127.0.0.1:9000 (CTRL + C to quit)"
            )

        hypercorn_info.assert_called_once()

    def test_http_config_keeps_hypercorn_info_and_errors(self) -> None:
        config = _server_config("127.0.0.1:9000")

        self.assertEqual(config.loglevel, "INFO")
        self.assertEqual(config.errorlog, "-")

    async def test_occupied_port_does_not_print_a_false_ready_splash(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch("protocols.acp.http.server.print_http_splash") as splash:
                # Hypercorn 0.17 can leave its failed-bind probe socket for
                # garbage collection. Keep that dependency warning out of this
                # MIRA behavior test while still exercising the real bind path.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    with self.assertRaises(OSError):
                        await serve_http(f"127.0.0.1:{port}")

        splash.assert_not_called()

    async def test_http_bootstrap_rejects_non_loopback_bind(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            await serve_http("0.0.0.0:8765")

    async def test_real_http_preserves_sessions_interactions_and_connection_isolation(
        self,
    ) -> None:
        store = PersistentState()
        applications: list[HttpFakeApplication] = []
        shutdown = asyncio.Event()
        factory = MiraAgentFactory()

        async def start(*, workspace: Path, frontend: object) -> HttpFakeApplication:
            application = HttpFakeApplication(workspace, frontend, store)
            applications.append(application)
            return application

        with tempfile.TemporaryDirectory() as workspace:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}/acp"
            with patch(
                "protocols.acp.shared.agent.MiraApplication.start",
                new=AsyncMock(side_effect=start),
            ), patch("protocols.acp.http.server.print_http_splash") as splash:
                server_task = asyncio.create_task(
                    serve_http(
                        f"127.0.0.1:{port}",
                        shutdown_trigger=shutdown.wait,
                        agent_factory=factory,
                    )
                )
                await wait_for_server(port)
                first_client = RecordingClient("choice-0", "approve", "close")
                second_client = RecordingClient()
                first, first_transport = await connect(url, first_client)
                second, second_transport = await connect(url, second_client)
                try:
                    created = await first.new_session(workspace)
                    await first.prompt(created.session_id, [text_block("one")])
                    await first.prompt(created.session_id, [text_block("two")])
                    await first.set_session_mode(created.session_id, "plan")
                    await first.prompt(created.session_id, [text_block("ask")])
                    await first.prompt(created.session_id, [text_block("approve")])
                    await first.prompt(created.session_id, [text_block("goal")])
                    with self.assertRaises(RequestError):
                        await first.prompt(
                            created.session_id,
                            [resource_link_block("file:///not-text", "not text")],
                        )

                    other = await second.new_session(workspace)
                    await second.prompt(other.session_id, [text_block("other")])

                    blocked = asyncio.create_task(
                        first.prompt(created.session_id, [text_block("block")])
                    )
                    await asyncio.wait_for(store.prompt_started.wait(), timeout=5)
                    await first.cancel(created.session_id)
                    cancelled = await asyncio.wait_for(blocked, timeout=5)
                finally:
                    await first.close()
                    await first_transport.close()
                    await second.close()
                    await second_transport.close()

                self.assertEqual(cancelled.stop_reason, "cancelled")
                self.assertEqual(
                    store.sessions[created.session_id]["prompts"],
                    ["one", "two", "ask", "approve", "goal", "block"],
                )
                self.assertEqual(store.sessions[created.session_id]["mode"], "planning")
                self.assertEqual(store.sessions[created.session_id]["answers"], ["A"])
                self.assertEqual(
                    store.sessions[created.session_id]["decisions"],
                    [{"type": "approve"}],
                )
                self.assertEqual(
                    store.sessions[created.session_id]["goal_review"],
                    {"action": "close"},
                )
                self.assertEqual(len(first_client.permissions), 3)
                self.assertFalse(
                    any(
                        isinstance(update, schema.AgentPlanUpdate)
                        for _, update in first_client.updates
                    )
                )
                self.assertIn(
                    "MIRA Goal",
                    "\n".join(str(update) for _, update in first_client.updates),
                )
                self.assertTrue(
                    all(item[0] == created.session_id for item in first_client.updates)
                )
                self.assertTrue(
                    all(item[0] == other.session_id for item in second_client.updates)
                )
                self.assertEqual(factory.agent_count, 2)
                self.assertEqual(sum(app.shutdowns for app in applications), 0)

                resumed_client = RecordingClient()
                resumed, resumed_transport = await connect(url, resumed_client)
                try:
                    loaded = await resumed.load_session(workspace, created.session_id)
                finally:
                    await resumed.close()
                    await resumed_transport.close()
                self.assertEqual(loaded.modes.current_mode_id, "plan")
                replayed_text = "\n".join(str(update) for _, update in resumed_client.updates)
                self.assertIn("reply: one", replayed_text)
                self.assertIn("reply: two", replayed_text)

                shutdown.set()
                await asyncio.wait_for(server_task, timeout=10)

            splash.assert_called_once_with(f"127.0.0.1:{port}")

        self.assertEqual(factory.agent_count, 0)
        self.assertEqual(sum(app.shutdowns for app in applications), len(applications))


if __name__ == "__main__":
    unittest.main()
