"""Tests for the Textual interactive UI."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from pyfiglet import Figlet
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from textual.color import Color
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static, TextArea

from agent.compaction import PostTurnCompactionResult
from agent.context_overflow import context_overflow_error, set_context_overflow_notice
from config.metadata import ModelMetadata
from config.settings import (
    dynamic_subagent_response_schema_enabled,
    dynamic_subagents_enabled,
    execute_env_settings,
    load_settings,
    save_settings,
    set_dynamic_subagents,
    tool_always_allow,
    tool_enabled,
)
from config.version import display_version
from runtime.diagnostics import get_diagnostics_logger, setup_diagnostics_logging
from runtime.trace_stream import TraceStream
from ui.interrupts import ASK_USER_OPEN_OPTION, action_choices, action_preview, normalize_plan
from ui.app import DESTRUCTIVE_CONFIRM_CHOICES, MiraApp, append_prompt_history, read_prompt_history
from ui.renderer import Renderer
from ui.splash import HINTS, VERSION, blocky_wordmark, splash_text
from ui.terminal_colors import strip_ansi
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory, SettingsPanel, StatusBar, SubagentsPanel
from ui.widgets.settings_panel import SettingsHeaderRow
from ui.widgets.subagent_panel import SubagentRecord, append_task_cell, group_status_icon, truncate_cells
from ui.widgets.session_history import session_label


PROMPT_BUTTON_FOCUS_BACKGROUND = Color.parse("#d2a957")
PROMPT_BUTTON_FOCUS_COLOR = Color.parse("#0c0f10")


class FakeStore:
    """Store double used by app smoke tests."""

    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []
        self.clear_all_calls = 0
        self.clear_compactions_calls = 0
        self.new_sessions = 0

    def save(self, record: dict[str, Any]) -> None:
        """Ignore session saves."""
        self.saves.append(record)
        return None

    def load(self, session_id: str | None, resume: bool, workspace: Path) -> dict[str, Any]:
        """Return a new session for tests that exercise session creation."""
        self.new_sessions += 1
        record = {
            "id": session_id or f"new-thread-{self.new_sessions}",
            "title": "Untitled session",
            "workspace": str(workspace),
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "dashboard": {},
            "events": [],
        }
        self.save(record)
        return record

    def clear_all(self) -> int:
        """Record all-session clears."""
        self.clear_all_calls += 1
        return 3

    def clear_compactions(self) -> int:
        """Record compaction archive clears."""
        self.clear_compactions_calls += 1
        return 2


class FakeWorker:
    """Worker double for cancel binding tests."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def renderable_plain(widget: Any) -> str:
    """Return plain text from a Textual Static-like widget."""
    renderable = getattr(widget, "renderable", None) or getattr(widget, "_renderable", None) or getattr(widget, "content", "")
    plain = getattr(renderable, "plain", None)
    if plain is not None:
        return str(plain)
    if isinstance(renderable, str):
        return renderable
    output = StringIO()
    Console(file=output, force_terminal=False, width=120).print(renderable)
    return output.getvalue()


async def wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    """Wait until a UI side effect is visible to the test."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.02)


def make_app(workspace: Path | None = None, session: dict[str, Any] | None = None, **state_overrides: Any) -> MiraApp:
    """Return a bootstrapped app with fake agents and session state."""
    workspace = workspace or Path(".")
    session_record = session or {
        "id": "thread-1",
        "workspace": str(workspace),
        "created_at": "2026-01-01T00:00:00+00:00",
        "turns": 0,
        "dashboard": {
            "model": "test-model",
            "context": {
                "used_tokens": 5512,
                "limit_tokens": 8192,
                "percent": 67.2,
                "source": "usage_metadata",
            },
            "tokens": {"in": 45230, "out": 12991},
            "duration_seconds": 12,
        },
    }
    state = {
        "agent": "agent",
        "plan_agent": "plan-agent",
        "config": {},
        "store": FakeStore(),
        "session": session_record,
        "model_name": "test-model",
        "context_limit_tokens": 8192,
        "context_limit_source": "test",
        "checkpointer": "checkpointer",
    }
    state.update(state_overrides)
    return MiraApp(
        workspace=workspace,
        prebuilt=state,
        tool_output_chars=80,
    )


class TextualAppTests(unittest.IsolatedAsyncioTestCase):
    """Smoke tests for the Textual app shell."""

    def assert_styled_char(self, text: Text, char: str, expected_style: str) -> None:
        """Assert that one character in a Rich Text object has a style."""
        matches = [
            span
            for span in text.spans
            if text.plain[span.start : span.end] == char and expected_style in str(span.style)
        ]
        self.assertTrue(matches, f"expected {char!r} to have style {expected_style!r}")

    def test_destructive_confirm_choices_match_visible_shortcuts(self) -> None:
        """Clear confirmations should return the same shortcuts shown in their labels."""
        self.assertEqual(DESTRUCTIVE_CONFIRM_CHOICES, [("o", "OK (o)"), ("c", "Cancel (c)")])

    def test_splash_text_uses_blocky_wordmark_and_logo_width(self) -> None:
        """The startup splash should preserve the old blocky logo structure."""
        wordmark = blocky_wordmark()
        logo_width = max(len(line.rstrip()) for line in wordmark.splitlines())
        plain = splash_text(
            model_name="lmstudio:test-model",
            session_id="thread-1",
            workspace="D:\\Projects\\mira",
        ).plain

        self.assertEqual(wordmark, Figlet(font="blocky").renderText("MIRA").rstrip())
        self.assertIn("=" * logo_width, plain)
        self.assertIn("-" * logo_width, plain)
        self.assertIn(VERSION, plain)
        self.assertIn("session   thread-1", plain)
        self.assertIn("model     lmstudio:test-model", plain)
        self.assertIn("workspace D:\\Projects\\mira", plain)
        self.assertIn(HINTS, plain)

    def test_splash_version_uses_central_project_version(self) -> None:
        """The splash version should come from the central package version."""
        self.assertEqual(VERSION, display_version())

    def test_session_label_shows_latest_prompt_preview_and_timestamp(self) -> None:
        """Sidebar rows should preview the latest user prompt instead of generated titles."""
        label = session_label(
            {
                "title": "Implementation Strategy Selection Work",
                "updated_at": "2026-06-03T09:15:00",
                "turns": 7,
                "events": [
                    {"type": "user", "text": "first prompt"},
                    {"type": "assistant", "text": "done"},
                    {"type": "user", "text": "tell me a 1000 word story with a quiet ending"},
                ],
            }
        )

        self.assertEqual(label.count("\n"), 2)
        self.assertIn("tell me a 1000 word story", label)
        self.assertIn("Jun 03 09:15", label)
        self.assertNotIn("turn", label.lower())
        self.assertLessEqual(max(len(line) for line in label.splitlines()[:-1]), 34)

    async def test_bootstrapped_app_renders_stream_and_tool_events_in_chat(self) -> None:
        """Stream events and tool calls should stay in the central transcript."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.text_delta("hello")
            app.tool_call("read_file", {"path": "README.md"})
            await pilot.pause()

            self.assertTrue(app.ready)
            self.assertFalse(app.query_one(PromptBox).disabled)
            self.assertGreaterEqual(len(app.query_one(ChatLog).children), 3)
            self.assertIsNotNone(app.query_one(SessionHistory))
            self.assertEqual(renderable_plain(app.query_one("#session-sidebar-title", Static)), "Chat History")
            self.assertEqual(app.query_one("#new-chat", Button).label.plain, "+ New")
            startup = app.query_one(ChatLog).children[0]
            startup_text = renderable_plain(startup)
            self.assertIn(VERSION, startup_text)
            self.assertIn(blocky_wordmark().splitlines()[-1].rstrip(), startup_text)
            self.assertIn(HINTS, startup_text)

    async def test_new_chat_button_switches_to_fresh_saved_session(self) -> None:
        """The flat sidebar action should create a new chat without clearing the old one."""
        old_session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 2,
            "events": [{"type": "user", "text": "keep me"}],
        }
        store = FakeStore()
        app = make_app(session=old_session, store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#new-chat", Button).press()
            await wait_until(lambda: app.session["id"] == "new-thread-1")

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertEqual(old_session["events"], [{"type": "user", "text": "keep me"}])
            self.assertEqual(app.session["turns"], 0)
            self.assertEqual(app.session["events"], [])
            self.assertEqual(store.new_sessions, 1)
            self.assertIn("started new chat", rendered)
            self.assertNotIn("keep me", rendered)
            self.assertTrue(app.query_one(PromptBox).has_focus)

    async def test_new_chat_command_uses_same_session_switch(self) -> None:
        """The slash command should create a blank saved session like the sidebar action."""
        store = FakeStore()
        app = make_app(store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/new-chat"))
            await wait_until(lambda: app.session["id"] == "new-thread-1")

            self.assertEqual(app.session["turns"], 0)
            self.assertEqual(app.session["events"], [])
            self.assertEqual(store.new_sessions, 1)

    async def test_new_chat_is_blocked_while_busy(self) -> None:
        """Starting a new chat should wait until the active turn finishes."""
        store = FakeStore()
        app = make_app(store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True

            handled = app._handle_new_chat()
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertTrue(handled)
            self.assertEqual(app.session["id"], "thread-1")
            self.assertEqual(store.new_sessions, 0)
            self.assertIn("finish the current turn before starting a new chat", rendered)

    async def test_narrow_window_hides_sidebar_and_keeps_prompt_width(self) -> None:
        """Very narrow terminals should not squeeze the prompt to zero width."""
        app = make_app()

        async with app.run_test(size=(50, 20)) as pilot:
            await pilot.pause()

            self.assertFalse(app.query_one("#session-sidebar").display)
            self.assertGreater(app.query_one(PromptBox).region.width, 0)

    async def test_prompt_submission_runs_turn_and_restores_focus(self) -> None:
        """Submitting prompt text should run the turn helper and refocus input."""
        app = make_app()
        calls: list[str] = []

        async def fake_run_user_turn(**kwargs: Any) -> None:
            calls.append(kwargs["text"])
            kwargs["renderer"].text_delta("done")

        with patch("ui.app.run_user_turn", fake_run_user_turn):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)
                prompt.focus()

                await app.submit_prompt(PromptBox.Submitted(prompt, "hello\nsecond line"))
                await pilot.pause()

                self.assertEqual(calls, ["hello\nsecond line"])
                self.assertFalse(prompt.disabled)
                self.assertTrue(prompt.has_focus)

    async def test_prompt_submission_refreshes_status_after_live_usage(self) -> None:
        """Live usage updates should redraw the mounted status bar immediately."""
        app = make_app()

        async def fake_run_user_turn(**kwargs: Any) -> None:
            kwargs["session"]["dashboard"]["tokens"]["in"] += 100
            kwargs["session"]["dashboard"]["tokens"]["out"] += 20
            kwargs["renderer"].usage_updated()

        with patch("ui.app.run_user_turn", fake_run_user_turn):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "hello"))
                await pilot.pause()

                status = renderable_plain(app.query_one("#status", Static))
                self.assertIn("In 45.3k Out 13.0k", status)

    async def test_turn_failure_shows_error_report_path(self) -> None:
        """Suppressed TUI turn errors should show the generated report path."""
        app = make_app()

        async def fake_run_user_turn(**kwargs: Any) -> None:
            raise RuntimeError("turn boom")

        with (
            patch("ui.app.run_user_turn", fake_run_user_turn),
            patch("ui.app.write_error_report", return_value=Path("turn-report.txt")) as report,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app.submit_prompt(PromptBox.Submitted(app.query_one(PromptBox), "hello"))
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn("error: turn boom", rendered)
                self.assertIn("error report: turn-report.txt", rendered)
                report.assert_called_once()
                self.assertEqual(report.call_args.kwargs["source"], "tui.turn")
                self.assertEqual(report.call_args.kwargs["session_id"], "thread-1")

    async def test_plan_turn_failure_shows_error_report_path(self) -> None:
        """Approved-plan turn errors should show the generated report path."""
        app = make_app()

        async def fake_run_user_turn(**kwargs: Any) -> None:
            raise RuntimeError("plan boom")

        with (
            patch("ui.app.run_user_turn", fake_run_user_turn),
            patch("ui.app.write_error_report", return_value=Path("plan-report.txt")) as report,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app._run_turn_for_plan({"title": "Build It"})
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn("error: plan boom", rendered)
                self.assertIn("error report: plan-report.txt", rendered)
                self.assertEqual(report.call_args.kwargs["source"], "tui.plan_turn")

    async def test_plan_revision_failure_shows_error_report_path(self) -> None:
        """Plan-revision errors should show the generated report path."""
        app = make_app()

        async def fake_run_user_turn(**kwargs: Any) -> None:
            raise RuntimeError("revision boom")

        with (
            patch("ui.app.run_user_turn", fake_run_user_turn),
            patch("ui.app.write_error_report", return_value=Path("revision-report.txt")) as report,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app._run_turn_for_plan_revision({"title": "Revise It"}, "more tests")
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn("error: revision boom", rendered)
                self.assertIn("error report: revision-report.txt", rendered)
                self.assertEqual(report.call_args.kwargs["source"], "tui.plan_revision")

    async def test_startup_failure_shows_error_report_path(self) -> None:
        """Startup errors should get a report even before a real session exists."""
        async def bootstrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("startup boom")

        async def ensure_git_repository(*args: Any, **kwargs: Any) -> bool:
            return True

        app = MiraApp(
            workspace=Path("."),
            config={},
            bootstrap=bootstrap,
            ensure_git_repository=ensure_git_repository,
        )

        with patch("ui.app.write_error_report", return_value=Path("startup-report.txt")) as report:
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                await wait_until(lambda: app.status_state == "error")

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn("startup error: startup boom", rendered)
                self.assertIn("error report: startup-report.txt", rendered)
                self.assertEqual(report.call_args.kwargs["source"], "tui.startup")
                self.assertIsNone(report.call_args.kwargs["session_id"])

    async def test_session_load_failure_shows_error_report_path(self) -> None:
        """Session-switch failures should show the generated report path."""
        app = make_app()

        async def bootstrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("load boom")

        app.bootstrap = bootstrap

        with patch("ui.app.write_error_report", return_value=Path("load-report.txt")) as report:
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app._load_session("thread-2")
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn("session load error: load boom", rendered)
                self.assertIn("error report: load-report.txt", rendered)
                self.assertEqual(report.call_args.kwargs["source"], "tui.session_load")
                self.assertEqual(report.call_args.kwargs["session_id"], "thread-1")

    async def test_trace_logging_mirrors_visible_tui_activity(self) -> None:
        """Trace mode should log normal TUI renderer activity."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            logger = get_diagnostics_logger()
            old_handlers = list(logger.handlers)
            logger.handlers = []
            try:
                log_path = setup_diagnostics_logging(workspace)
                app = make_app(workspace=workspace)
                app.trace = TraceStream(logger, output_chars=80)

                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()

                    app.user_message("hello trace")
                    app.text_delta("assistant ")
                    app.text_delta("hello")
                    app.reasoning_delta("hidden thinking")
                    app.tool_call("read_file", {"path": "README.md"})
                    app.tool_result("read_file", "content" * 30)
                    app.system_message("bad thing", kind="error")
                    app.subagent_started("worker", "do a thing")
                    app.subagent_finished("worker", "done")
                    await pilot.pause()

                content = log_path.read_text(encoding="utf-8")
                self.assertIn("user:\nhello trace", content)
                self.assertIn("mira:\nassistant hello", content)
                self.assertIn("read_file:\nargs:", content)
                self.assertIn("read_file output: content", content)
                self.assertIn("error:\nbad thing", content)
                self.assertIn("subagent - worker:\nrequest: do a thing", content)
                self.assertIn("subagent - worker:\ndone", content)
                self.assertIn("truncated", content)
                self.assertIn("thinking:\nhidden thinking", content)
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = old_handlers

    async def test_trace_recovered_tool_result_matches_terminal_callback_order(self) -> None:
        """Recovered tool results should use the same callback order as mira -p."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            logger = get_diagnostics_logger()
            old_handlers = list(logger.handlers)
            logger.handlers = []
            try:
                log_path = setup_diagnostics_logging(workspace)
                app = make_app(workspace=workspace)
                app.trace = TraceStream(logger, output_chars=80)

                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()

                    app.text_delta("final answer")
                    app.model_stream_finished()
                    app.recovered_tool_result("write_todos", "updated todos")
                    app.finish_main()
                    await pilot.pause()

                content = log_path.read_text(encoding="utf-8")
                tool_index = content.index("write_todos output: updated todos")
                assistant_index = content.index("mira:\nfinal answer")
                self.assertLess(assistant_index, tool_index)
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = old_handlers

    async def test_trace_logging_is_silent_without_trace_setup(self) -> None:
        """Normal TUI activity should not create trace logs unless trace mode is configured."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            logger = get_diagnostics_logger()
            old_handlers = list(logger.handlers)
            logger.handlers = []
            try:
                app = make_app(workspace=workspace)
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()

                    app.user_message("hello no trace")
                    app.tool_call("read_file", {"path": "README.md"})
                    await pilot.pause()

                self.assertFalse((workspace / ".mira" / "_logs").exists())
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = old_handlers

    async def test_waiting_indicator_appears_after_silence_and_hides_on_output(self) -> None:
        """The transient working block should appear only during phase silence."""
        app = make_app()
        app._waiting_delay_seconds = 0.05

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.busy = True
            app.waiting_started()
            await pilot.pause(0.08)
            first = renderable_plain(app.query_one(ChatLog).children[-1])
            self.assertIn("working...", first)

            app.query_one(ChatLog).tick_waiting()
            await pilot.pause()
            second = renderable_plain(app.query_one(ChatLog).children[-1])
            self.assertIn("working...", second)
            self.assertNotEqual(first, second)

            app.text_delta("hello")
            await pilot.pause()
            blocks = list(app.query_one(ChatLog).children)
            self.assertEqual(renderable_plain(blocks[-1]), "hello")
            self.assertFalse(any("working..." in renderable_plain(block) for block in blocks))

            app.waiting_started()
            await pilot.pause(0.08)
            self.assertFalse(any("working..." in renderable_plain(block) for block in app.query_one(ChatLog).children))

            app.waiting_finished()
            await pilot.pause()
            self.assertFalse(any("working..." in renderable_plain(block) for block in app.query_one(ChatLog).children))
            app.busy = False

    async def test_slow_token_generation_does_not_show_working_between_chunks(self) -> None:
        """Once text is streaming, slow token gaps should not re-add working status."""
        app = make_app()
        app._waiting_delay_seconds = 0.05

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True

            app.waiting_started()
            await pilot.pause(0.08)
            self.assertIn("working...", renderable_plain(app.query_one(ChatLog).children[-1]))

            app.text_delta("h")
            await pilot.pause(0.12)
            app.text_delta("i")
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("hi", rendered)
            self.assertNotIn("working...", rendered)
            app.busy = False

    async def test_reasoning_stream_silence_does_not_show_working(self) -> None:
        """Slow reasoning token gaps should not re-add the working status."""
        app = make_app()
        app._waiting_delay_seconds = 0.05

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True

            app.reasoning_delta("I should delegate this.")
            await pilot.pause(0.08)

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("I should delegate this.", rendered)
            self.assertNotIn("working...", rendered)
            app.busy = False

    async def test_whitespace_reasoning_delta_does_not_create_thinking_block(self) -> None:
        """Whitespace-only reasoning should not render an empty thinking block."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            base_count = len(app.query_one(ChatLog).children)
            app.reasoning_delta("   \n")
            await pilot.pause()

            self.assertEqual(len(app.query_one(ChatLog).children), base_count)

    async def test_reasoning_delta_preserves_line_break_chunks_after_text(self) -> None:
        """TUI thinking blocks should preserve newline chunks once text exists."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.reasoning_delta("The user wants:")
            app.reasoning_delta("\n")
            app.reasoning_delta("1. Check envs")
            app.reasoning_delta("\n")
            app.reasoning_delta("2. Tell joke")
            await pilot.pause()

            reasoning_blocks = [block for block in app.query_one(ChatLog).children if "reasoning" in block.classes]
            self.assertEqual(len(reasoning_blocks), 1)
            self.assertEqual(renderable_plain(reasoning_blocks[0]), "The user wants:\n1. Check envs\n2. Tell joke")

    async def test_reasoning_starts_new_block_after_intervening_bubbles(self) -> None:
        """Any visible non-reasoning phase should separate thinking blocks."""
        cases = [
            ("assistant", lambda app: app.text_delta("mira text")),
            ("tool draft", lambda app: app.tool_call_delta("read_file", {"path": "README.md"})),
            ("tool call", lambda app: app.tool_call("read_file", {"path": "README.md"}, call_id="call-read")),
            ("delegation draft", lambda app: app.delegation_delta([{"name": "task", "args": {"description": "judge"}}])),
            ("delegation", lambda app: app.delegation_started([{"name": "task", "args": {"description": "judge"}}])),
            ("subagent", lambda app: app.subagent_started("general-purpose", "judge")),
            ("compaction", lambda app: app.compaction_started()),
            ("system", lambda app: app.system_message("status update", kind="status")),
            ("command", lambda app: app.command_output("command output")),
            ("activity", lambda app: app.model_activity()),
        ]

        for name, action in cases:
            with self.subTest(name=name):
                app = make_app()
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()

                    app.reasoning_delta("first reasoning")
                    action(app)
                    app.reasoning_delta("second reasoning")
                    await pilot.pause()

                    reasoning_blocks = [
                        block for block in app.query_one(ChatLog).children if "reasoning" in block.classes
                    ]

                    self.assertEqual(
                        [renderable_plain(block) for block in reasoning_blocks],
                        ["first reasoning", "second reasoning"],
                    )

    async def test_reasoning_starts_new_block_after_plan_bubble(self) -> None:
        """Structured plan bubbles should also separate thinking blocks."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Boundary Plan",
            "summary": ["Separate phases."],
            "key_changes": ["Render a plan bubble."],
            "test_plan": ["Check reasoning block count."],
            "assumptions": ["The plan is visible."],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.reasoning_delta("first reasoning")
            await app.present_plan(interrupt)
            app.reasoning_delta("second reasoning")
            await pilot.pause()

            reasoning_blocks = [block for block in app.query_one(ChatLog).children if "reasoning" in block.classes]
            self.assertEqual(
                [renderable_plain(block) for block in reasoning_blocks],
                ["first reasoning", "second reasoning"],
            )

    async def test_cancel_turn_detaches_reasoning_and_assistant_blocks(self) -> None:
        """New streamed output after cancellation should start fresh main bubbles."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.reasoning_delta("old thinking")
            app.text_delta("old answer")
            await pilot.pause()

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            app.reasoning_delta("new thinking")
            app.text_delta("new answer")
            await pilot.pause()

            reasoning_blocks = [block for block in app.query_one(ChatLog).children if "reasoning" in block.classes]
            assistant_blocks = [block for block in app.query_one(ChatLog).children if "assistant" in block.classes]

            self.assertEqual([renderable_plain(block) for block in reasoning_blocks], ["old thinking", "new thinking"])
            self.assertEqual([renderable_plain(block) for block in assistant_blocks], ["old answer", "new answer"])

    def test_terminal_renderer_skips_whitespace_reasoning_delta(self) -> None:
        """One-shot output should not print blank thinking sections."""
        renderer = Renderer()
        output = StringIO()

        with redirect_stdout(output):
            renderer.reasoning_delta("   \n")
            renderer.finish_main()

        self.assertEqual(output.getvalue(), "")

    def test_terminal_renderer_preserves_reasoning_line_break_chunks(self) -> None:
        """One-shot thinking output should preserve newline chunks between text."""
        renderer = Renderer()
        output = StringIO()

        with redirect_stdout(output):
            renderer.reasoning_delta("The user wants:")
            renderer.reasoning_delta("\n")
            renderer.reasoning_delta("1. Check envs\n")
            renderer.reasoning_delta("2. Tell joke")
            renderer.finish_main()

        rendered = output.getvalue()
        self.assertIn("\033[", rendered)
        self.assertEqual(strip_ansi(rendered), "\nthinking:\nThe user wants:\n1. Check envs\n2. Tell joke\n")

    async def test_terminal_renderer_filters_respond_decision(self) -> None:
        """One-shot approval prompts should not accept Respond from stale interrupts."""
        renderer = Renderer()
        interrupt = {
            "action_requests": [{"name": "execute", "args": {"command": "conda env list"}}],
            "review_configs": [
                {
                    "action_name": "execute",
                    "allowed_decisions": ["approve", "edit", "reject", "respond"],
                }
            ],
        }
        output = StringIO()

        with patch("builtins.input", side_effect=["s", "r"]) as input_mock, redirect_stdout(output):
            decisions = await renderer.ask_approvals([interrupt])

        self.assertEqual(decisions, [{"type": "reject"}])
        prompts = "\n".join(str(call.args[0]) for call in input_mock.call_args_list)
        self.assertIn("a=Approve (a), e=Edit (e), r=Reject (r)", prompts)
        self.assertNotIn("Respond", prompts)

    async def test_tool_call_streaming_activity_suppresses_working(self) -> None:
        """Tool-call JSON chunks should show activity instead of silent waiting."""
        app = make_app()
        app._waiting_delay_seconds = 0.05

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True

            app.waiting_started()
            await pilot.pause(0.03)
            app.model_activity()
            await pilot.pause(0.08)

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("preparing tool call...", rendered)
            self.assertNotIn("working...", rendered)

            app.tool_call("ls", {"path": "/"}, call_id="call-1")
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertNotIn("preparing tool call...", rendered)
            self.assertIn("call:", rendered)
            app.busy = False

    async def test_cancel_turn_removes_waiting_and_model_activity(self) -> None:
        """Cancellation should clear transient status bubbles before the next turn."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.query_one(ChatLog).show_waiting()
            await pilot.pause()
            self.assertIn("working...", renderable_plain(app.query_one(ChatLog).children[-1]))

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertNotIn("working...", rendered)

            app.busy = True
            app.turn_worker = FakeWorker()
            app.model_activity()
            await pilot.pause()
            self.assertIn("preparing tool call...", renderable_plain(app.query_one(ChatLog).children[-1]))

            app._cancel_turn()
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertNotIn("preparing tool call...", rendered)

    async def test_unknown_context_usage_renders_pending(self) -> None:
        """A context limit without provider usage should not pretend to be measured."""
        session = {
            "id": "thread-pending",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "dashboard": {},
            "events": [],
        }
        app = make_app(
            session=session,
            context_limit_tokens=10000,
            context_limit_source="lmstudio.api.v1.loaded_instance",
        )

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            status = renderable_plain(app.query_one("#status", Static))
            self.assertIn("pending", status)
            self.assertIn("?/10.0k", status)
            self.assertNotIn("14/10.0k", status)

    async def test_blank_leading_assistant_text_does_not_create_empty_block(self) -> None:
        """Leading blank assistant deltas should be ignored until real text arrives."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            base_count = len(app.query_one(ChatLog).children)
            app.text_delta("\n\n")
            await pilot.pause()
            self.assertEqual(len(app.query_one(ChatLog).children), base_count)

            app.text_delta("\n\n## Summary")
            await pilot.pause()
            text = renderable_plain(app.query_one(ChatLog).children[-1])
            self.assertEqual(text, "## Summary")

    async def test_restore_session_skips_blank_reasoning_blocks(self) -> None:
        """Blank persisted reasoning should not render as empty thinking boxes."""
        session = {
            "id": "thread-blank-reasoning",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 1,
            "dashboard": {},
            "events": [
                {"id": 1, "type": "user", "mode": "action", "text": "hello"},
                {"id": 2, "type": "reasoning", "mode": "action", "text": ""},
                {"id": 3, "type": "reasoning", "mode": "action", "text": "   \n"},
                {"id": 4, "type": "reasoning", "mode": "action", "text": "real thought"},
            ],
        }
        app = make_app(session=session)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            reasoning_blocks = [block for block in app.query_one(ChatLog).children if "reasoning" in block.classes]
            self.assertEqual(len(reasoning_blocks), 1)
            self.assertEqual(renderable_plain(reasoning_blocks[0]), "real thought")

    async def test_assistant_text_preserves_internal_markdown_newlines(self) -> None:
        """Assistant text should keep markdown paragraph breaks after visible text starts."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.text_delta("## Summary")
            app.text_delta("\n\nThe eval tool passed.")
            await pilot.pause()

            text = renderable_plain(app.query_one(ChatLog).children[-1])
            self.assertEqual(text, "## Summary\n\nThe eval tool passed.")

    async def test_compaction_status_renders_and_completes(self) -> None:
        """Context compaction status should render and then mark completion."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.compaction_started()
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            first = renderable_plain(block)

            app.query_one(ChatLog).tick_compaction()
            await pilot.pause()
            animated = renderable_plain(block)

            app.compaction_finished()
            await pilot.pause()
            done = renderable_plain(block)

            self.assertIn("compacting context...", first)
            self.assertIn("compacting context...", animated)
            self.assertNotEqual(first, animated)
            self.assertIn("context compacted", done)

    async def test_cancel_turn_stops_compaction_spinner(self) -> None:
        """Cancelled compaction status should become terminal and non-animated."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.compaction_started()
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            self.assertIn("compacting context...", renderable_plain(block))

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()
            cancelled = renderable_plain(block)

            self.assertIn("context compaction cancelled", cancelled)
            self.assertNotIn("compacting context...", cancelled)

            app.query_one(ChatLog).tick_compaction()
            await pilot.pause()
            self.assertEqual(renderable_plain(block), cancelled)

    async def test_discard_reasoning_removes_current_thinking_block(self) -> None:
        """Late compaction detection should retract already-rendered reasoning."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.reasoning_delta("I am reviewing a normal request.")
            await pilot.pause()
            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("I am reviewing a normal request.", rendered)

            app.discard_reasoning()
            await pilot.pause()
            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertNotIn("I am reviewing a normal request.", rendered)

    async def test_info_message_renders_as_dedicated_block(self) -> None:
        """Info messages should use the first-class info box."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.system_message("context notice", kind="info")
            await pilot.pause()

            block = app.query_one(ChatLog).children[-1]
            self.assertEqual(str(getattr(block, "border_title", "")), "info")
            self.assertIn("info", block.classes)
            self.assertIn("context notice", renderable_plain(block))

    async def test_context_overflow_notice_stays_separate_from_compaction_block(self) -> None:
        """Context-overflow copy should render as info, not inside compacting status."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            set_context_overflow_notice("Provider context limit reached. Compacting older context and retrying.")
            app.compaction_started()
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            info = blocks[-2]
            compaction = blocks[-1]
            self.assertEqual(str(getattr(info, "border_title", "")), "info")
            self.assertIn("Provider context limit reached", renderable_plain(info))
            self.assertEqual(str(getattr(compaction, "border_title", "")), "mira")
            self.assertIn("compacting context...", renderable_plain(compaction))
            self.assertNotIn("Provider context limit", renderable_plain(compaction))

    async def test_compaction_reasoning_notice_is_not_rendered_as_info(self) -> None:
        """Leaked compaction reasoning must not render as an info notice."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            set_context_overflow_notice(
                "The user wants me to extract the most important context from this conversation history. "
                "Key information to extract: Session intent, Summary, Artifacts, Next Steps."
            )
            app.compaction_started()
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            self.assertEqual(str(getattr(blocks[-1], "border_title", "")), "mira")
            self.assertIn("compacting context...", renderable_plain(blocks[-1]))
            self.assertNotIn("Key information to extract", "\n".join(renderable_plain(block) for block in blocks))

    async def test_escaped_context_overflow_renders_info_and_ready_state(self) -> None:
        """An escaped context overflow should not become a red error block."""
        app = make_app()
        notice = "Context limit pressure detected. Compacting older context and retrying."

        async def fake_run_user_turn(**kwargs: Any) -> None:
            raise context_overflow_error("provider context limit reached", notice)

        with patch("ui.app.run_user_turn", fake_run_user_turn):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "hello"))
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertIn(notice, rendered)
                self.assertNotIn("MIRA simulated a context overflow", rendered)
                self.assertFalse(any("error" in block.classes for block in app.query_one(ChatLog).children))
                self.assertEqual(app.status_state, "ready")
                self.assertFalse(prompt.disabled)

    async def test_waiting_indicator_reappears_after_compaction_if_still_busy(self) -> None:
        """After compaction, MIRA should show working again while waiting silently."""
        app = make_app()
        app._waiting_delay_seconds = 0.05

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True

            app.compaction_started()
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]

            app.compaction_finished()
            await pilot.pause(0.08)

            blocks = list(app.query_one(ChatLog).children)
            self.assertIn("context compacted", renderable_plain(block))
            self.assertIn("working...", renderable_plain(blocks[-1]))
            app.busy = False

    async def test_startup_loading_splash_renders_before_bootstrap_finishes(self) -> None:
        """Startup should show branded progress before async bootstrap completes."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def bootstrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
            started.set()
            await release.wait()
            return {
                "agent": "agent",
                "plan_agent": "plan-agent",
                "config": {},
                "store": FakeStore(),
                "session": {
                    "id": "thread-startup",
                    "workspace": ".",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "turns": 0,
                },
                "model_name": "test-model",
                "context_limit_tokens": 8192,
                "context_limit_source": "test",
                "checkpointer": "checkpointer",
            }

        async def ensure_git_repository(*args: Any, **kwargs: Any) -> bool:
            return True

        app = MiraApp(
            workspace=Path("."),
            config={},
            bootstrap=bootstrap,
            ensure_git_repository=ensure_git_repository,
        )

        async with app.run_test(size=(100, 30)) as pilot:
            await asyncio.wait_for(started.wait(), timeout=2)
            await pilot.pause()

            self.assertFalse(app.ready)
            loading_text = renderable_plain(app.query_one(ChatLog).children[0])
            self.assertIn(VERSION, loading_text)
            self.assertIn("loading", loading_text)
            self.assertIn("loading model metadata", loading_text)

            release.set()
            await pilot.pause()
            self.assertTrue(app.ready)

    async def test_alt_q_cancels_running_turn(self) -> None:
        """Alt+Q should confirm before cancelling a running turn."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            worker = FakeWorker()
            app.busy = True
            app.turn_worker = worker

            await pilot.press("alt+q")
            await pilot.pause()
            self.assertFalse(worker.cancelled)
            self.assertEqual(renderable_plain(app.query_one("#prompt-panel-title", Static)), "Cancel Turn?")
            self.assertEqual(
                [button.label.plain for button in app.query_one(PromptPanel).query(Button)],
                ["Yes (y)", "No (n)"],
            )

            await pilot.press("n")
            await pilot.pause()
            self.assertFalse(worker.cancelled)

            await pilot.press("alt+q")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

            self.assertTrue(worker.cancelled)
            self.assertEqual(app.status_state, "cancelling")

    async def test_alt_q_confirms_idle_exit(self) -> None:
        """Alt+Q should not quit an idle app without confirmation."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as exit_app:
                await pilot.press("alt+q")
                await pilot.pause()
                self.assertEqual(renderable_plain(app.query_one("#prompt-panel-title", Static)), "Exit MIRA?")
                self.assertEqual(
                    [button.label.plain for button in app.query_one(PromptPanel).query(Button)],
                    ["Yes (y)", "No (n)"],
                )

                await pilot.press("n")
                await pilot.pause()
                exit_app.assert_not_called()

                await pilot.press("alt+q")
                await pilot.pause()
                await pilot.press("y")
                await pilot.pause()
                exit_app.assert_called_once()

    async def test_ctrl_c_copies_prompt_selection(self) -> None:
        """Ctrl+C should copy focused text instead of interrupting."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            prompt.value = "copy this"
            prompt.select_all()

            with patch.object(app, "action_interrupt_or_quit") as interrupt:
                await pilot.press("ctrl+c")
                await pilot.pause()

            self.assertEqual(getattr(app, "_clipboard", ""), "copy this")
            interrupt.assert_not_called()

    async def test_alt_q_cancels_running_turn_during_prompt(self) -> None:
        """Alt+Q should still cancel when another in-window prompt is active."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt_task = asyncio.create_task(
                app._prompt_choice("Approval", "Run this tool?", [("y", "y yes"), ("n", "n no")])
            )
            await pilot.pause()

            worker = FakeWorker()
            app.busy = True
            app.turn_worker = worker

            with patch.object(app, "exit") as exit_app:
                await pilot.press("alt+q")
                await pilot.pause()

                self.assertTrue(worker.cancelled)
                self.assertEqual(app.status_state, "cancelling")
                exit_app.assert_not_called()

            await pilot.press("n")
            self.assertEqual(await prompt_task, "n")

    async def test_loading_past_session_replays_ordered_events(self) -> None:
        """Selecting an older session should rebuild its visible transcript."""
        workspace = Path(".")
        past_session = {
            "id": "past-1",
            "workspace": str(workspace),
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 2,
            "events": [
                {
                    "id": 1,
                    "type": "compaction",
                    "cutoff_index": 2,
                    "file_path": "/.mira/conversation_history/past-1.md",
                    "summary": "Older turns compacted for replay testing.",
                    "created_at": "2026-01-01T00:01:00+00:00",
                },
                {
                    "id": 2,
                    "type": "info",
                    "mode": "action",
                    "created_at": "2026-01-01T00:01:30+00:00",
                    "text": "Provider context limit reached.",
                },
                {
                    "id": 3,
                    "type": "user",
                    "mode": "planning",
                    "created_at": "2026-01-01T00:02:00+00:00",
                    "text": "make a plan",
                },
                {
                    "id": 4,
                    "type": "assistant",
                    "mode": "planning",
                    "created_at": "2026-01-01T00:02:01+00:00",
                    "text": "plan saved",
                },
            ],
        }

        async def bootstrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "agent": "agent",
                "plan_agent": "plan-agent",
                "config": {},
                "store": FakeStore(),
                "session": past_session,
                "model_name": "test-model",
                "context_limit_tokens": 8192,
                "context_limit_source": "test",
                "checkpointer": "checkpointer",
            }

        app = make_app(workspace=workspace)
        app.bootstrap = bootstrap

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            await app._load_session("past-1")
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            rendered = "\n".join(renderable_plain(block) for block in blocks)
            titles = [str(getattr(block, "border_title", "")) for block in blocks]

            self.assertIn("session compacted", titles)
            self.assertIn("info", titles)
            self.assertIn("you (plan)", titles)
            self.assertIn("mira", titles)
            self.assertIn("Older turns compacted", rendered)
            self.assertIn("Provider context limit reached.", rendered)
            self.assertIn("/.mira/conversation_history/past-1.md", rendered)
            self.assertIn("make a plan", rendered)
            self.assertIn("plan saved", rendered)

    async def test_restore_session_timestamps_all_persisted_bubble_titles(self) -> None:
        """Persisted session events should replay with JSON-backed timestamp titles."""
        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-06-24T09:00:00+08:00",
            "turns": 1,
            "events": [
                {
                    "id": 1,
                    "type": "user",
                    "mode": "action",
                    "created_at": "2026-06-24T09:01:00+08:00",
                    "text": "hello",
                },
                {
                    "id": 2,
                    "type": "user",
                    "mode": "planning",
                    "created_at": "2026-06-24T09:02:00+08:00",
                    "text": "plan it",
                },
                {
                    "id": 3,
                    "type": "assistant",
                    "mode": "action",
                    "created_at": "2026-06-24T09:03:00+08:00",
                    "text": "hi",
                },
                {
                    "id": 4,
                    "type": "reasoning",
                    "mode": "action",
                    "created_at": "2026-06-24T09:04:00+08:00",
                    "text": "thinking",
                },
                {
                    "id": 5,
                    "type": "info",
                    "mode": "action",
                    "created_at": "2026-06-24T09:05:00+08:00",
                    "text": "info",
                },
                {
                    "id": 6,
                    "type": "system_error",
                    "mode": "action",
                    "created_at": "2026-06-24T09:06:00+08:00",
                    "text": "broken",
                },
                {
                    "id": 7,
                    "type": "interrupted",
                    "mode": "action",
                    "created_at": "2026-06-24T09:07:00+08:00",
                    "text": "stopped",
                },
                {
                    "id": 8,
                    "type": "tool_call",
                    "mode": "action",
                    "created_at": "2026-06-24T09:08:00+08:00",
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "call_id": "call-1",
                },
                {
                    "id": 9,
                    "type": "tool_result",
                    "mode": "action",
                    "created_at": "2026-06-24T09:09:00+08:00",
                    "name": "read_file",
                    "output": "contents",
                    "call_id": "call-1",
                },
                {
                    "id": 10,
                    "type": "delegation",
                    "mode": "action",
                    "created_at": "2026-06-24T09:10:00+08:00",
                    "calls": [
                        {
                            "name": "task",
                            "args": {"description": "check timestamps", "subagent_type": "general-purpose"},
                        }
                    ],
                },
                {
                    "id": 11,
                    "type": "subagent",
                    "mode": "action",
                    "created_at": "2026-06-24T09:11:00+08:00",
                    "name": "general-purpose [luna]",
                    "status": "DONE",
                    "task_input": "check timestamps",
                    "output": "done",
                },
                {
                    "id": 12,
                    "type": "compaction",
                    "mode": "action",
                    "created_at": "2026-06-24T09:12:00+08:00",
                    "summary": "older context",
                    "file_path": ".mira/conversation_history/thread.md",
                    "cutoff_index": 3,
                },
            ],
        }
        app = make_app(session=session)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            titles = [str(getattr(block, "border_title", "")).replace("\\", "") for block in blocks]
            subtitles = [str(getattr(block, "border_subtitle", "")) for block in blocks]

            self.assertIn("you", titles)
            self.assertIn("you (plan)", titles)
            self.assertIn("mira", titles)
            self.assertIn("thinking", titles)
            self.assertIn("info", titles)
            self.assertIn("error", titles)
            self.assertIn("warning", titles)
            self.assertIn("tool - read_file", titles)
            self.assertIn("task", titles)
            self.assertIn("subagent - general-purpose [luna]", titles)
            self.assertIn("session compacted", titles)
            for minute in range(1, 13):
                if minute == 9:
                    continue
                self.assertIn(f"2026-06-24 09:{minute:02d}", subtitles)

    async def test_unchanged_context_metadata_does_not_rebuild_agents(self) -> None:
        """A matching refreshed context window should avoid rebuilding both agents."""
        app = make_app(config={"llm_provider": "lmstudio", "llm_model": "local-model"})

        async def infer_metadata(config: dict[str, Any], model: Any | None = None) -> ModelMetadata:
            return ModelMetadata(8192, "lmstudio.api.v1.loaded_instance")

        with (
            patch("agent.llm.get_llm", return_value=type("Model", (), {"profile": {}})()),
            patch("config.metadata.infer_model_metadata", infer_metadata),
            patch("agent.factory.build_agent") as build_agent,
            patch("agent.factory.build_plan_agent") as build_plan_agent,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                await app._refresh_model_metadata()

        build_agent.assert_not_called()
        build_plan_agent.assert_not_called()

    async def test_changed_context_metadata_rebuilds_agents_once(self) -> None:
        """A changed context window should rebuild action and planning agents once."""
        store = FakeStore()
        app = make_app(store=store, config={"llm_provider": "lmstudio", "llm_model": "local-model"})

        async def infer_metadata(config: dict[str, Any], model: Any | None = None) -> ModelMetadata:
            return ModelMetadata(10000, "lmstudio.api.v1.loaded_instance")

        with (
            patch("agent.llm.get_llm", return_value=type("Model", (), {"profile": {}})()),
            patch("config.metadata.infer_model_metadata", infer_metadata),
            patch("agent.factory.build_agent", return_value="new-agent") as build_agent,
            patch("agent.factory.build_plan_agent", return_value="new-plan-agent") as build_plan_agent,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                await app._refresh_model_metadata()

        self.assertEqual(app.agent, "new-agent")
        self.assertEqual(app.plan_agent, "new-plan-agent")
        self.assertEqual(app.context_limit_tokens, 10000)
        self.assertEqual(app.context_limit_source, "lmstudio.api.v1.loaded_instance")
        self.assertEqual(app.config["llm_inferred_context_tokens"], 10000)
        self.assertEqual(build_agent.call_count, 1)
        self.assertEqual(build_plan_agent.call_count, 1)
        self.assertEqual(len(store.saves), 1)

    async def test_help_command_renders_as_one_command_panel(self) -> None:
        """The help command should not create one chat block per command."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)

            await app.submit_prompt(PromptBox.Submitted(prompt, "/help"))
            await pilot.pause()

            command_blocks = [child for child in app.query_one(ChatLog).children if "command" in child.classes]
            self.assertEqual(len(command_blocks), 1)
            output = renderable_plain(command_blocks[0])
            self.assertIn("Commands", output)
            self.assertIn("/help", output)
            self.assertIn("/compact", output)
            self.assertIn("/subagents", output)

    async def test_compact_command_updates_action_thread_without_adding_turn(self) -> None:
        app = make_app()
        compact_result = PostTurnCompactionResult(compacted=True, reason="compacted")

        with (
            patch("ui.app.compact_after_turn", new=AsyncMock(return_value=compact_result)) as compact,
            patch("ui.app.sync_deepagents_compaction", new=AsyncMock(return_value=True)) as sync,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "/compact"))
                await wait_until(lambda: not app.busy)
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)

        compact.assert_awaited_once_with("agent", "thread-1")
        sync.assert_awaited_once_with(app.session, "agent", "thread-1")
        self.assertEqual(app.session["turns"], 0)
        self.assertEqual(len(app.store.saves), 1)
        self.assertIn("context compacted", rendered)

    async def test_compact_command_uses_active_planning_thread(self) -> None:
        app = make_app()
        compact_result = PostTurnCompactionResult(reason="nothing_to_compact")

        with patch("ui.app.compact_after_turn", new=AsyncMock(return_value=compact_result)) as compact:
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app.mode["planning"] = True
                app.mode["plan_thread_id"] = "thread-1:plan:7"
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "/compact"))
                await wait_until(lambda: not app.busy)
                await pilot.pause()

        compact.assert_awaited_once_with("plan-agent", "thread-1:plan:7")
        self.assertEqual(app.session["turns"], 0)

    async def test_compact_command_reports_noop_without_syncing_session(self) -> None:
        app = make_app()
        compact_result = PostTurnCompactionResult(reason="nothing_to_compact")

        with (
            patch("ui.app.compact_after_turn", new=AsyncMock(return_value=compact_result)),
            patch("ui.app.sync_deepagents_compaction", new=AsyncMock()) as sync,
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "/compact"))
                await wait_until(lambda: not app.busy)
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)

        sync.assert_not_awaited()
        self.assertEqual(app.store.saves, [])
        self.assertIn("nothing to compact", rendered)

    async def test_compact_command_reports_failure_and_restores_prompt(self) -> None:
        app = make_app()

        with (
            patch("ui.app.compact_after_turn", new=AsyncMock(side_effect=RuntimeError("summary failed"))),
            patch.object(app, "_write_error_report", return_value=Path("compaction-error.json")),
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)

                await app.submit_prompt(PromptBox.Submitted(prompt, "/compact"))
                await wait_until(lambda: not app.busy)
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertFalse(prompt.disabled)

        self.assertIn("context compaction failed", rendered)
        self.assertIn("compaction error: summary failed", rendered)
        self.assertIn("compaction-error.json", rendered)

    async def test_session_command_renders_as_one_status_bubble(self) -> None:
        """The session command should group its details into one chat block."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)

            await app.submit_prompt(PromptBox.Submitted(prompt, "/session"))
            await pilot.pause()

            system_blocks = [child for child in app.query_one(ChatLog).children if "system" in child.classes]
            self.assertEqual(len(system_blocks), 1)
            output = renderable_plain(system_blocks[0])
            self.assertIn("session: thread-1", output)
            self.assertIn("mode: action", output)
            self.assertIn("turns: 0", output)

    async def test_reload_command_rebuilds_agents(self) -> None:
        """The reload command should rebuild agents and refresh visible metadata."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, patch.dict(os.environ, {}, clear=True):
            workspace = Path(directory)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "MIRA_LLM_PROVIDER=lmstudio",
                        "MIRA_LLM_MODEL=visual-model",
                    ]
                ),
                encoding="utf-8",
            )
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            action_agent = type("Agent", (), {"mira_tool_specs": [{"name": "fresh_tool", "description": "Fresh."}]})()
            plan_agent = type("PlanAgent", (), {"mira_tool_specs": [{"name": "plan_tool", "description": "Plan."}]})()

            async def infer_metadata(config: dict[str, Any], model: Any | None = None) -> ModelMetadata:
                return ModelMetadata(10000, "reload-test")

            with (
                patch("agent.llm.get_llm", return_value=type("Model", (), {"profile": {}})()),
                patch("config.metadata.infer_model_metadata", infer_metadata),
                patch("agent.factory.build_agent", return_value=action_agent) as build_agent,
                patch("agent.factory.build_plan_agent", return_value=plan_agent) as build_plan_agent,
            ):
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()
                    prompt = app.query_one(PromptBox)

                    await app.submit_prompt(PromptBox.Submitted(prompt, "/reload"))
                    await wait_until(lambda: app.agent is action_agent)
                    await pilot.pause(0.3)
                    rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                    status = renderable_plain(app.query_one(StatusBar))
                    startup_blocks = [child for child in app.query_one(ChatLog).children if "startup" in child.classes]

            self.assertIs(app.plan_agent, plan_agent)
            self.assertEqual(app.model_name, "lmstudio:visual-model")
            self.assertEqual(app.context_limit_tokens, 10000)
            self.assertEqual(app.context_limit_source, "reload-test")
            self.assertEqual(build_agent.call_count, 1)
            self.assertEqual(build_plan_agent.call_count, 1)
            self.assertEqual(app.mode["action_tools"], [{"name": "fresh_tool", "description": "Fresh."}])
            self.assertEqual(app.mode["planning_tools"], [{"name": "plan_tool", "description": "Plan."}])
            self.assertEqual(len(startup_blocks), 1)
            self.assertIn("model     lmstudio:visual-model", rendered)
            self.assertNotIn("model     loading", rendered)
            self.assertNotIn("- starting", rendered)
            self.assertIn("lmstudio:visual-model", status)
            self.assertIn("agents reloaded", rendered)

    async def test_reload_command_reload_dotenv_with_override(self) -> None:
        """The reload command should let edited .env values replace loaded env values."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, patch.dict(os.environ, {}, clear=True):
            workspace = Path(directory)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "MIRA_LLM_PROVIDER=lmstudio",
                        "MIRA_LLM_MODEL=from-env-file",
                    ]
                ),
                encoding="utf-8",
            )
            os.environ["MIRA_LLM_PROVIDER"] = "lmstudio"
            os.environ["MIRA_LLM_MODEL"] = "already-loaded"
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async def rebuild(**kwargs: Any) -> None:
                return None

            async def infer_metadata(config: dict[str, Any], model: Any | None = None) -> ModelMetadata:
                return ModelMetadata(8192, "reload-test")

            with (
                patch("agent.llm.get_llm", return_value=type("Model", (), {"profile": {}})()),
                patch("config.metadata.infer_model_metadata", infer_metadata),
            ):
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()
                    app._rebuild_agents = rebuild

                    await app._handle_reload_command()

            self.assertEqual(app.config["llm_model"], "from-env-file")
            self.assertEqual(os.environ["MIRA_LLM_MODEL"], "from-env-file")

    async def test_reload_command_is_refused_while_busy(self) -> None:
        """The reload command should not rebuild agents during an active turn."""
        app = make_app()
        calls: list[str] = []

        async def rebuild(**kwargs: Any) -> None:
            calls.append("rebuild")

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._rebuild_agents = rebuild
            app.busy = True

            handled = await app._handle_reload_command()
            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)

        self.assertTrue(handled)
        self.assertEqual(calls, [])
        self.assertIn("finish the current turn before reloading agents", rendered)

    async def test_approval_prompt_uses_in_window_panel_with_arrow_keys(self) -> None:
        """Approval prompts should stay in the app layout and accept arrow keys."""
        app = make_app()
        interrupt = {
            "action_requests": [
                {
                    "name": "write_file",
                    "args": {"file_path": "test.txt", "content": "hello world"},
                }
            ]
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            with patch.object(app, "push_screen_wait", side_effect=AssertionError("modal prompt used")):
                task = asyncio.create_task(app.ask_approvals([interrupt]))
                await pilot.pause()

                panel = app.query_one(PromptPanel)
                await wait_until(lambda: len(list(panel.query(Button))) == 3)
                await wait_until(
                    lambda: len(list(panel.query(Button))) == 3 and list(panel.query(Button))[0].has_focus
                )
                await wait_until(lambda: not panel._reflow_running and len(panel._button_positions) == 3)
                buttons = list(panel.query(Button))
                self.assertTrue(panel.display)
                self.assertEqual(len(buttons), 3)
                self.assertTrue(buttons[0].has_focus)
                self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
                self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
                self.assertEqual(buttons[0].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
                self.assertEqual(buttons[0].styles.color, PROMPT_BUTTON_FOCUS_COLOR)
                self.assertNotEqual(buttons[1].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
                self.assertNotEqual(buttons[2].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)

                for _ in range(3):
                    await pilot.press("right")
                    await pilot.pause()
                    if list(panel.query(Button))[1].has_focus:
                        break
                await wait_until(lambda: list(panel.query(Button))[1].has_focus)
                buttons = list(panel.query(Button))
                self.assertTrue(buttons[1].has_focus)
                self.assertEqual(buttons[1].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
                self.assertNotEqual(buttons[0].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
                self.assertNotEqual(buttons[2].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
                self.assertFalse(app.query_one(PromptBox).has_focus)

                for _ in range(3):
                    await pilot.press("right")
                    await pilot.pause()
                    if list(panel.query(Button))[2].has_focus:
                        break
                await wait_until(lambda: list(panel.query(Button))[2].has_focus)
                buttons = list(panel.query(Button))
                self.assertTrue(buttons[2].has_focus)

                await pilot.press("enter")
                decisions = await asyncio.wait_for(task, timeout=2)
                await pilot.pause()

                self.assertEqual(decisions, [{"type": "reject"}])
                self.assertFalse(panel.display)

    async def test_approval_prompt_filters_respond_decision(self) -> None:
        """Respond should not be surfaced even if an interrupt advertises it."""
        app = make_app()
        interrupt = {
            "action_requests": [
                {
                    "name": "edit_file",
                    "args": {"file_path": "loop.txt", "old_string": "a", "new_string": "b"},
                }
            ],
            "review_configs": [
                {
                    "action_name": "edit_file",
                    "allowed_decisions": ["approve", "edit", "reject", "respond"],
                }
            ],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app.ask_approvals([interrupt]))
            await pilot.pause()

            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Approve (a)", "Edit (e)", "Reject (r)"],
            )
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            self.assertTrue(all(button.variant == "default" for button in buttons))

            await pilot.press("r")

            self.assertEqual(
                await asyncio.wait_for(task, timeout=2),
                [{"type": "reject"}],
            )

    def test_action_choices_filters_respond_decision(self) -> None:
        """Interrupt helpers should hide stale/upstream respond choices."""
        interrupt = {
            "action_requests": [{"name": "execute", "args": {"command": "conda env list"}}],
            "review_configs": [
                {
                    "action_name": "execute",
                    "allowed_decisions": ["approve", "edit", "reject", "respond"],
                }
            ],
        }

        choices = action_choices(interrupt, interrupt["action_requests"][0], 0)

        self.assertEqual(choices, [("a", "Approve (a)"), ("e", "Edit (e)"), ("r", "Reject (r)")])

    async def test_prompt_box_uses_up_down_history_from_workspace_file(self) -> None:
        """The main prompt should navigate workspace history with Up and Down."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            history_path = workspace / ".mira" / "history.txt"
            history_path.parent.mkdir()
            history_path.write_text(
                "\n# earlier\n+first prompt\n\n# later\n+second prompt\n",
                encoding="utf-8",
            )
            app = make_app(workspace=workspace)

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)
                prompt.focus()
                prompt.value = "draft"

                await pilot.press("up")
                self.assertEqual(prompt.value, "second prompt")
                await pilot.press("up")
                self.assertEqual(prompt.value, "first prompt")
                await pilot.press("down")
                self.assertEqual(prompt.value, "second prompt")
                await pilot.press("down")
                self.assertEqual(prompt.value, "draft")

    def test_prompt_history_helpers_read_and_append_history_file(self) -> None:
        """History helpers should preserve the workspace history file format."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            history_path = Path(directory) / ".mira" / "history.txt"
            append_prompt_history(history_path, "first prompt")
            append_prompt_history(history_path, "second prompt\nwith paste")

            self.assertEqual(read_prompt_history(history_path), ["first prompt", "second prompt\nwith paste"])

    async def test_clear_chat_command_cancel_keeps_session(self) -> None:
        """Cancelling /clear-chat should leave the active saved transcript untouched."""
        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 2,
            "events": [{"type": "user", "text": "keep me"}],
        }
        store = FakeStore()
        app = make_app(session=session, store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_prompt_choice", return_value="c"):
                handled = await app._handle_history_command("/clear-chat")
            await pilot.pause()

            self.assertTrue(handled)
            self.assertEqual(app.session["turns"], 2)
            self.assertEqual(app.session["events"], [{"type": "user", "text": "keep me"}])
            self.assertEqual(store.saves, [])

    async def test_clear_chat_command_confirm_resets_current_session(self) -> None:
        """Confirming /clear-chat should clear the current transcript and save it."""
        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 2,
            "dashboard": {"tokens": {"in": 10, "out": 5}},
            "events": [{"type": "user", "text": "clear me"}],
        }
        store = FakeStore()
        app = make_app(session=session, store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_prompt_choice", return_value="o"):
                handled = await app._handle_history_command("/clear-chat")
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertTrue(handled)
            self.assertEqual(app.session["turns"], 0)
            self.assertEqual(app.session["events"], [])
            self.assertEqual(app.session["dashboard"]["tokens"], {"in": 0, "out": 0})
            self.assertEqual(len(store.saves), 1)
            self.assertIn("current chat history cleared", rendered)
            self.assertNotIn("clear me", rendered)

    async def test_clear_chat_confirmation_accepts_ok_shortcut(self) -> None:
        """The current-chat clear should visibly support O to confirm."""
        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 1,
            "events": [{"type": "user", "text": "clear me"}],
        }
        store = FakeStore()
        app = make_app(session=session, store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            task = asyncio.create_task(app.submit_prompt(PromptBox.Submitted(prompt, "/clear-chat")))
            await pilot.pause()
            await asyncio.wait_for(task, timeout=2)

            await wait_until(lambda: len(list(app.query_one(PromptPanel).query(Button))) == 2)
            buttons = list(app.query_one(PromptPanel).query(Button))
            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            self.assertIn("Press O to confirm, C or Esc to cancel.", message)

            await pilot.press("o")
            await wait_until(lambda: app.session["events"] == [])

            self.assertEqual(app.session["events"], [])
            self.assertEqual(len(store.saves), 1)

    async def test_clear_all_chats_requires_confirmation_and_keeps_active_clean_session(self) -> None:
        """The all-chat clear should require confirmation before deleting sessions."""
        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 1,
            "events": [{"type": "user", "text": "clear all"}],
        }
        store = FakeStore()
        app = make_app(session=session, store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_prompt_choice", return_value="c"):
                await app._handle_history_command("/clear-all-chats")
            self.assertEqual(store.clear_all_calls, 0)
            self.assertEqual(store.clear_compactions_calls, 0)
            self.assertEqual(app.session["events"], [{"type": "user", "text": "clear all"}])

            with patch.object(app, "_prompt_choice", return_value="o"):
                await app._handle_history_command("/clear-all-chats")
            await pilot.pause()

            self.assertEqual(store.clear_all_calls, 1)
            self.assertEqual(store.clear_compactions_calls, 1)
            self.assertEqual(app.session["turns"], 0)
            self.assertEqual(app.session["events"], [])
            self.assertGreaterEqual(len(store.saves), 1)
            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("cleared 3 saved chat sessions and 2 compaction files", rendered)

    async def test_clear_all_chats_confirmation_accepts_ok_shortcut(self) -> None:
        """The all-chat confirmation should work from the ok/cancel choice box."""
        store = FakeStore()
        app = make_app(store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            task = asyncio.create_task(app.submit_prompt(PromptBox.Submitted(prompt, "/clear-all-chats")))
            await pilot.pause()
            await asyncio.wait_for(task, timeout=2)

            await wait_until(lambda: len(list(app.query_one(PromptPanel).query(Button))) == 2)
            buttons = list(app.query_one(PromptPanel).query(Button))
            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            self.assertIn("Press O to confirm, C or Esc to cancel.", message)

            await pilot.press("o")
            await wait_until(lambda: store.clear_all_calls == 1)

            self.assertEqual(store.clear_all_calls, 1)
            self.assertEqual(store.clear_compactions_calls, 1)

    async def test_clear_all_chats_confirmation_accepts_cancel_shortcut(self) -> None:
        """The all-chat confirmation should visibly support C to cancel."""
        store = FakeStore()
        app = make_app(store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            task = asyncio.create_task(app.submit_prompt(PromptBox.Submitted(prompt, "/clear-all-chats")))
            await pilot.pause()
            await asyncio.wait_for(task, timeout=2)

            await wait_until(lambda: len(list(app.query_one(PromptPanel).query(Button))) == 2)
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))

            await pilot.press("c")
            await wait_until(lambda: not app.query_one(PromptPanel).display)

            self.assertEqual(store.clear_all_calls, 0)
            self.assertEqual(store.clear_compactions_calls, 0)

    async def test_clear_errors_command_confirm_deletes_error_reports(self) -> None:
        """The error-report clear should delete only .mira/_errors files."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            errors = workspace / ".mira" / "_errors"
            session_errors = errors / "thread-1"
            session_errors.mkdir(parents=True)
            (session_errors / "report.txt").write_text("boom", encoding="utf-8")
            (errors / "latest_error.txt").write_text("latest", encoding="utf-8")
            settings = workspace / ".mira" / "settings.yml"
            settings.write_text("keep: true\n", encoding="utf-8")
            app = make_app(workspace=workspace)

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                with patch.object(app, "_prompt_choice", return_value="o"):
                    handled = await app._handle_history_command("/clear-errors")
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertTrue(handled)
                self.assertFalse(errors.exists())
                self.assertEqual(settings.read_text(encoding="utf-8"), "keep: true\n")
                self.assertIn("cleared 2 error report files", rendered)

    async def test_clear_errors_command_cancel_keeps_error_reports(self) -> None:
        """Cancelling the error-report clear should leave reports untouched."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            report = workspace / ".mira" / "_errors" / "thread-1" / "report.txt"
            report.parent.mkdir(parents=True)
            report.write_text("boom", encoding="utf-8")
            app = make_app(workspace=workspace)

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                with patch.object(app, "_prompt_choice", return_value="c"):
                    handled = await app._handle_history_command("/clear-errors")
                await pilot.pause()

                rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
                self.assertTrue(handled)
                self.assertTrue(report.exists())
                self.assertIn("clear error reports cancelled", rendered)

    async def test_clear_chat_commands_keep_error_reports(self) -> None:
        """Chat-history clears should not delete diagnostic error reports."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            report = workspace / ".mira" / "_errors" / "thread-1" / "report.txt"
            report.parent.mkdir(parents=True)
            report.write_text("boom", encoding="utf-8")
            session = {
                "id": "thread-1",
                "workspace": str(workspace),
                "created_at": "2026-01-01T00:00:00+00:00",
                "turns": 1,
                "events": [{"type": "user", "text": "clear me"}],
            }
            app = make_app(workspace=workspace, session=session, store=FakeStore())

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                with patch.object(app, "_prompt_choice", return_value="o"):
                    await app._handle_history_command("/clear-chat")
                self.assertTrue(report.exists())

                with patch.object(app, "_prompt_choice", return_value="o"):
                    await app._handle_history_command("/clear-all-chats")
                self.assertTrue(report.exists())

    async def test_clear_prompts_confirmation_accepts_escape_cancel(self) -> None:
        """The prompt-history clear should visibly support Esc to cancel."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            history_path = workspace / ".mira" / "history.txt"
            history_path.parent.mkdir()
            history_path.write_text("\n# earlier\n+first prompt\n\n", encoding="utf-8")
            app = make_app(workspace=workspace)

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)
                task = asyncio.create_task(app.submit_prompt(PromptBox.Submitted(prompt, "/clear-prompts")))
                await pilot.pause()
                await asyncio.wait_for(task, timeout=2)

                await wait_until(lambda: len(list(app.query_one(PromptPanel).query(Button))) == 2)
                buttons = list(app.query_one(PromptPanel).query(Button))
                message = renderable_plain(app.query_one("#prompt-panel-message", Static))
                self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
                self.assertIn("Press O to confirm, C or Esc to cancel.", message)

                await pilot.press("escape")
                await wait_until(lambda: not app.query_one(PromptPanel).display)

                self.assertEqual(read_prompt_history(history_path), ["first prompt"])

    async def test_clear_prompts_command_clears_disk_and_prompt_history(self) -> None:
        """Confirming /clear-prompts should clear history.txt and in-memory prompt history."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            history_path = workspace / ".mira" / "history.txt"
            history_path.parent.mkdir()
            history_path.write_text("\n# earlier\n+first prompt\n\n", encoding="utf-8")
            app = make_app(workspace=workspace)

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)
                self.assertEqual(read_prompt_history(history_path), ["first prompt"])

                with patch.object(app, "_prompt_choice", return_value="o"):
                    await app._handle_history_command("/clear-prompts")
                await pilot.pause()

                prompt.value = "draft"
                prompt.focus()
                await pilot.press("up")
                self.assertEqual(history_path.read_text(encoding="utf-8"), "")
                self.assertEqual(prompt.value, "draft")

    async def test_destructive_history_command_is_refused_while_busy(self) -> None:
        """Destructive history commands should not run during an active turn."""
        store = FakeStore()
        app = make_app(store=store)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.busy = True
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/clear-chat"))
            await pilot.pause()

            rendered = "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            self.assertIn("finish the current turn before clearing history", rendered)
            self.assertEqual(store.saves, [])

    async def test_settings_command_toggles_tool_and_rebuilds_agents(self) -> None:
        """Keyboard changes should save settings and rebuild agents."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            calls: list[dict[str, Any]] = []

            async def rebuild(**kwargs: Any) -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                prompt = app.query_one(PromptBox)
                with patch.object(app, "_rebuild_agents", rebuild):
                    await app.submit_prompt(PromptBox.Submitted(prompt, "/settings"))
                    await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                    await wait_until(lambda: len(app.query_one(SettingsPanel).query(Button)) >= 8)
                    self.assertFalse(app.query_one(PromptPanel).display)
                    app.query_one("#settings-toggle-always_allow-edit_file", Button).focus()
                    await pilot.press("y")
                    await pilot.pause()

            loaded = load_settings(workspace)
            self.assertTrue(tool_always_allow(loaded, "edit_file"))
            self.assertEqual(len(calls), 1)
            self.assertTrue(tool_always_allow(calls[0], "edit_file"))

    async def test_settings_command_supports_click_toggling(self) -> None:
        """Clicking a toggle button should toggle it in place."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                async def rebuild(**kwargs: Any) -> None:
                    return None

                with patch.object(app, "_rebuild_agents", rebuild):
                    app._handle_settings_command()
                    await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                    panel = app.query_one(SettingsPanel)
                    await wait_until(lambda: len(panel.query(Button)) >= 3)
                    panel.query_one("#settings-toggle-always_allow-edit_file", Button).press()
                    await pilot.pause()

            self.assertTrue(tool_always_allow(load_settings(workspace), "edit_file"))

    async def test_settings_command_disables_git_without_deleting_git_directory(self) -> None:
        """Turning off Git protection should save settings and leave .git untouched."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            git_dir = workspace / ".git"
            git_dir.mkdir()
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: len(panel.query(Button)) >= 3)
                panel.query_one("#settings-toggle-git-git_protection", Button).focus()
                await pilot.press("n")
                await pilot.pause()

            self.assertTrue(git_dir.exists())
            self.assertFalse(load_settings(workspace)["hitl"]["git_protection"]["enabled"])

    async def test_settings_panel_renders_default_rows_and_closes(self) -> None:
        """The settings panel should include default rows and close with q."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: len(panel.query(Button)) >= 10)
                rendered = "\n".join(renderable_plain(child) for child in panel.query(Static))
                static_labels = {renderable_plain(child).strip() for child in panel.query(Static)}
                buttons = {button.id: button for button in panel.query(Button)}

                self.assertNotIn("Config", rendered)
                self.assertIn("Settings", rendered)
                self.assertIn("enable", rendered)
                self.assertIn("always allow", rendered)
                self.assertIn("System Settings", rendered)
                self.assertIn("Inbuilt Tools", rendered)
                self.assertIn("Execute Environment", rendered)
                self.assertIn("Run commands in", rendered)
                self.assertIn("Press Enter/click to change", rendered)
                self.assertIn("Additional env var names", rendered)
                self.assertIn("Examples only. Use comma-separated names.", rendered)
                self.assertIn("Custom Tools", rendered)
                self.assertNotIn("Tool", static_labels)
                self.assertIn("Git Protection", rendered)
                self.assertIn("Dynamic subagents", rendered)
                self.assertIn("Response schemas", rendered)
                self.assertIn("write_file", rendered)
                self.assertIn("edit_file", rendered)
                self.assertIn("eval", rendered)
                self.assertIn("task", rendered)
                self.assertIn("execute", rendered)
                self.assertIn("settings-toggle-git-git_protection", buttons)
                self.assertIn("settings-toggle-system-dynamic_subagents", buttons)
                self.assertIn("settings-toggle-response_schema-response_schema", buttons)
                self.assertIn("settings-toggle-enabled-edit_file", buttons)
                self.assertIn("settings-toggle-always_allow-edit_file", buttons)
                self.assertIn("settings-toggle-enabled-write_file", buttons)
                self.assertIn("settings-toggle-enabled-execute", buttons)
                self.assertIn("settings-toggle-always_allow-execute", buttons)
                self.assertIn("settings-execute-env-mode", buttons)
                self.assertIn("settings-close", buttons)
                self.assertEqual(str(buttons["settings-execute-env-mode"].label), "system shell >")
                self.assertEqual(panel.query_one("#settings-execute-env-allow", Input).value, "")
                self.assertEqual(
                    panel.query_one("#settings-execute-env-allow", Input).placeholder,
                    "<CUDA_HOME, HF_HOME, REQUESTS_CA_BUNDLE>",
                )
                self.assertTrue(panel.query_one("#settings-execute-env-allow", Input).display)
                self.assertEqual(str(buttons["settings-toggle-git-git_protection"].label), "yes")
                self.assertEqual(str(buttons["settings-toggle-system-dynamic_subagents"].label), "no")
                self.assertEqual(str(buttons["settings-toggle-response_schema-response_schema"].label), "yes")
                self.assertTrue(buttons["settings-toggle-response_schema-response_schema"].disabled)
                self.assertEqual(str(buttons["settings-toggle-enabled-edit_file"].label), "yes")
                self.assertFalse(buttons["settings-toggle-enabled-edit_file"].disabled)
                self.assertEqual(str(buttons["settings-toggle-enabled-execute"].label), "no")
                self.assertFalse(buttons["settings-toggle-enabled-execute"].disabled)
                self.assertEqual(str(buttons["settings-toggle-always_allow-execute"].label), "-")
                self.assertTrue(buttons["settings-toggle-always_allow-execute"].disabled)
                self.assertEqual(str(buttons["settings-toggle-always_allow-edit_file"].label), "no")
                self.assertEqual(str(buttons["settings-toggle-always_allow-write_file"].label), "no")

                panel.query_one("#settings-close", Button).press()
                await wait_until(lambda: len(app.query(SettingsPanel)) == 0)
                self.assertTrue(app.query_one(PromptBox).has_focus)

    async def test_tool_section_headers_match_system_spacing(self) -> None:
        """Tool tables should use the compact System Settings row progression."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 55)) as pilot:
                await pilot.pause()
                app.mode.setdefault("resources", {})["tools"] = [
                    {
                        "name": "project_status",
                        "path": "/.mira/tools/status.py",
                        "source": "project",
                        "replaces": "",
                    }
                ]
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: len(panel.query(SettingsHeaderRow)) == 3)
                await pilot.pause()

                labels = {
                    renderable_plain(widget).strip(): widget
                    for widget in panel.query(Static)
                    if renderable_plain(widget).strip()
                }
                system_header, inbuilt_header, custom_header = list(panel.query(SettingsHeaderRow))

                for section_name, header, first_row in (
                    ("System Settings", system_header, labels["Git Protection"]),
                    ("Inbuilt Tools", inbuilt_header, labels["write_file"]),
                ):
                    section = labels[section_name]
                    self.assertEqual(header.region.y, section.region.y + section.region.height)
                    self.assertEqual(first_row.region.y, header.region.y + header.region.height)

                custom_section = labels["Custom Tools"]
                custom_first_row = labels["project_status"]
                self.assertEqual(custom_header.region.y, custom_section.region.y + custom_section.region.height)
                self.assertEqual(custom_first_row.region.y, custom_header.region.y + custom_header.region.height)

                enable_x = [
                    header.query_one(".settings-column-label.enabled", Static).region.x
                    for header in (system_header, inbuilt_header, custom_header)
                ]
                self.assertEqual(enable_x, [enable_x[0]] * 3)

                execute_section = labels["Execute Environment"]
                inbuilt_section = labels["Inbuilt Tools"]
                last_system_row = labels["Response schemas"]
                last_inbuilt_row = labels["execute"]
                run_commands = labels["Run commands in"]
                execute_help = labels["Examples only. Use comma-separated names."]
                self.assertEqual(
                    inbuilt_section.region.y,
                    last_system_row.region.y + last_system_row.region.height + 1,
                )
                self.assertEqual(
                    execute_section.region.y,
                    last_inbuilt_row.region.y + last_inbuilt_row.region.height + 1,
                )
                self.assertEqual(run_commands.region.y, execute_section.region.y + execute_section.region.height)
                self.assertEqual(custom_section.region.y, execute_help.region.y + execute_help.region.height + 2)

    async def test_settings_panel_can_disable_inbuilt_tools(self) -> None:
        """Inbuilt tool enable buttons should save disabled state and lock approvals."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            calls = []

            async def rebuild() -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: "settings-toggle-enabled-edit_file" in {button.id for button in panel.query(Button)})

                panel.query_one("#settings-toggle-enabled-edit_file", Button).press()
                await pilot.pause()

                loaded = load_settings(workspace)
                buttons = {button.id: button for button in panel.query(Button)}
                self.assertFalse(tool_enabled(loaded, "edit_file"))
                self.assertEqual(str(buttons["settings-toggle-enabled-edit_file"].label), "no")
                self.assertEqual(str(buttons["settings-toggle-always_allow-edit_file"].label), "-")
                self.assertTrue(buttons["settings-toggle-always_allow-edit_file"].disabled)
                self.assertEqual(len(calls), 1)

    async def test_settings_panel_toggles_dynamic_subagents(self) -> None:
        """Dynamic subagents should save as a system setting and rebuild agents."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            calls = []

            async def rebuild() -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: "settings-toggle-system-dynamic_subagents" in {button.id for button in panel.query(Button)})

                panel.query_one("#settings-toggle-system-dynamic_subagents", Button).press()
                await wait_until(lambda: dynamic_subagents_enabled(load_settings(workspace)))

                buttons = {button.id: button for button in panel.query(Button)}
                self.assertEqual(str(buttons["settings-toggle-system-dynamic_subagents"].label), "yes")
                self.assertFalse(buttons["settings-toggle-response_schema-response_schema"].disabled)
                self.assertEqual(len(calls), 1)

    async def test_settings_panel_toggles_dynamic_response_schemas(self) -> None:
        """Dynamic response schemas should save independently and rebuild agents."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            settings = set_dynamic_subagents(load_settings(workspace), True)
            save_settings(workspace, settings)
            app = make_app(workspace=workspace, config={"settings": settings})
            calls = []

            async def rebuild() -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                button_id = "settings-toggle-response_schema-response_schema"
                await wait_until(lambda: button_id in {button.id for button in panel.query(Button)})

                panel.query_one(f"#{button_id}", Button).press()
                await wait_until(lambda: not dynamic_subagent_response_schema_enabled(load_settings(workspace)))

                buttons = {button.id: button for button in panel.query(Button)}
                self.assertEqual(str(buttons[button_id].label), "no")
                self.assertEqual(len(calls), 1)

    async def test_settings_panel_execute_env_cycle_preserves_scroll(self) -> None:
        """Changing execute env mode should not jump the settings body back to the top."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 18)) as pilot:
                await pilot.pause()
                app.mode.setdefault("resources", {})["tools"] = [
                    {
                        "name": "project_status",
                        "path": "/.mira/tools/status.py",
                        "source": "project",
                        "replaces": "",
                    }
                ]
                async def rebuild() -> None:
                    return None

                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: len(panel.query("#settings-execute-env-mode")) > 0)
                body = panel.query_one("#settings-body", VerticalScroll)
                original_panel = panel
                original_body = body
                body.set_scroll(None, 5)
                await pilot.pause()
                before = body.scroll_y

                panel.query_one("#settings-execute-env-mode", Button).press()
                await wait_until(lambda: execute_env_settings(load_settings(workspace))["mode"] == "conda_name")
                await pilot.pause()

                panel = app.query_one(SettingsPanel)
                body = panel.query_one("#settings-body", VerticalScroll)
                self.assertIs(panel, original_panel)
                self.assertIs(body, original_body)
                self.assertGreater(before, 0)
                self.assertGreater(body.scroll_y, 0)
                self.assertTrue(panel.query_one("#settings-execute-env-name-row").display)
                self.assertEqual(str(panel.query_one("#settings-execute-env-mode", Button).label), "conda env name >")
                self.assertTrue(panel.query_one("#settings-execute-env-mode", Button).has_focus)

    async def test_settings_panel_saves_execute_env_fields_without_placeholders(self) -> None:
        """Execute env fields should save explicit values and leave examples inert."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            calls = []

            async def rebuild() -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: len(panel.query("#settings-execute-env-mode")) > 0)

                panel.query_one("#settings-execute-env-mode", Button).press()
                await wait_until(lambda: len(panel.query("#settings-execute-env-name")) > 0)
                name_input = panel.query_one("#settings-execute-env-name", Input)
                self.assertEqual(name_input.value, "")
                self.assertEqual(name_input.placeholder, "<my_project_env>")
                name_input.value = "orbit_wars"
                await panel.submit_execute_env_input(Input.Submitted(name_input, "orbit_wars"))
                await wait_until(lambda: execute_env_settings(load_settings(workspace))["name"] == "orbit_wars")

                panel = app.query_one(SettingsPanel)
                allow_input = panel.query_one("#settings-execute-env-allow", Input)
                allow_input.value = "CUDA_HOME, HF_HOME, TOKEN=value"
                await panel.submit_execute_env_input(Input.Submitted(allow_input, "CUDA_HOME, HF_HOME, TOKEN=value"))
                await wait_until(lambda: execute_env_settings(load_settings(workspace))["allow"] == ["CUDA_HOME", "HF_HOME"])

            saved = execute_env_settings(load_settings(workspace))
            self.assertEqual(saved["mode"], "conda_name")
            self.assertEqual(saved["name"], "orbit_wars")
            self.assertEqual(saved["allow"], ["CUDA_HOME", "HF_HOME"])
            self.assertNotIn("my_project_env", str(saved))
            self.assertGreaterEqual(len(calls), 2)

    async def test_settings_panel_confirm_cancel_keeps_execute_disabled(self) -> None:
        """Cancelling the execute warning should leave settings unchanged."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            prompts: list[str] = []

            async def reject_execute(title: str, message: str, choices: list[tuple[str, str]]) -> str:
                prompts.append(message)
                return "n"

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._prompt_choice = reject_execute
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: "settings-toggle-enabled-execute" in {button.id for button in panel.query(Button)})

                panel.query_one("#settings-toggle-enabled-execute", Button).press()
                await pilot.pause()

                self.assertTrue(prompts)
                self.assertIn("LocalShellBackend", prompts[0])
                self.assertIn("small OS shell environment allowlist", prompts[0])
                self.assertIn("not your full environment or API keys", prompts[0])
                self.assertFalse(tool_enabled(load_settings(workspace), "execute"))
                self.assertEqual(str(panel.query_one("#settings-toggle-enabled-execute", Button).label), "no")

    async def test_settings_panel_confirm_enable_execute_rebuilds_agents(self) -> None:
        """Accepting the execute warning should save settings and rebuild agents."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})
            calls = []

            async def accept_execute(title: str, message: str, choices: list[tuple[str, str]]) -> str:
                return "y"

            async def rebuild() -> None:
                calls.append(dict(app.config or {}))

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app._prompt_choice = accept_execute
                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: "settings-toggle-enabled-execute" in {button.id for button in panel.query(Button)})

                panel.query_one("#settings-toggle-enabled-execute", Button).press()
                await pilot.pause()

                buttons = {button.id: button for button in panel.query(Button)}
                self.assertTrue(tool_enabled(load_settings(workspace), "execute"))
                self.assertEqual(str(buttons["settings-toggle-enabled-execute"].label), "yes")
                self.assertFalse(buttons["settings-toggle-always_allow-execute"].disabled)
                self.assertEqual(len(calls), 1)

    async def test_settings_panel_can_disable_custom_tools(self) -> None:
        """Custom tools should support enabled toggles and disabled approval cells."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            app = make_app(workspace=workspace, config={"settings": load_settings(workspace)})

            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app.mode.setdefault("resources", {})["tools"] = [
                    {
                        "name": "project_status",
                        "path": "/.mira/tools/status.py",
                        "source": "project",
                        "replaces": "",
                    }
                ]
                calls = []

                async def rebuild() -> None:
                    calls.append(dict(app.config or {}))

                app._rebuild_agents = rebuild
                app._handle_settings_command()
                await wait_until(lambda: len(app.query(SettingsPanel)) > 0)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: "settings-toggle-enabled-project_status" in {button.id for button in panel.query(Button)})

                panel.query_one("#settings-toggle-enabled-project_status", Button).press()
                await pilot.pause()

                loaded = load_settings(workspace)
                buttons = {button.id: button for button in panel.query(Button)}
                self.assertFalse(tool_enabled(loaded, "project_status"))
                self.assertEqual(str(buttons["settings-toggle-always_allow-project_status"].label), "-")
                self.assertTrue(buttons["settings-toggle-always_allow-project_status"].disabled)
                self.assertEqual(len(calls), 1)

    def test_action_preview_shows_key_value_rows(self) -> None:
        """Approval previews should show scan-friendly rows with truncated values."""
        text = action_preview(
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/.mira/memories/AGENTS.md",
                    "old_string": "old " * 100,
                    "new_string": "new " * 100,
                    "replace_all": False,
                },
            }
        )

        self.assertIn("target", text)
        self.assertIn("/.mira/memories/AGENTS.md", text)
        self.assertIn("old_string", text)
        self.assertIn("new_string", text)
        self.assertIn("replace_all", text)
        self.assertIn("truncated", text)
        self.assertIn("Press e to inspect or edit full args", text)
        self.assertLess(len(text), 700)

    async def test_long_approval_text_keeps_buttons_above_prompt_box(self) -> None:
        """Large approval bodies should not push the action buttons offscreen."""
        app = make_app()
        interrupt = {
            "action_requests": [
                {
                    "name": "edit_file",
                    "args": {
                        "file_path": "ui/dialogs.py",
                        "old_string": "x" * 1200,
                        "new_string": "y" * 1200,
                    },
                }
            ]
        }

        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause()
            task = asyncio.create_task(app.ask_approvals([interrupt]))
            await pilot.pause()

            panel = app.query_one(PromptPanel)
            body = app.query_one("#prompt-panel-body")
            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            prompt_y = app.query_one(PromptBox).region.y
            panel_bottom = panel.region.y + panel.region.height
            self.assertEqual(body.styles.overflow_y, "auto")
            self.assertIn("target", message)
            self.assertIn("ui/dialogs.py", message)
            self.assertIn("truncated", message)
            self.assertIn("Press e to inspect or edit full args", message)
            self.assertLess(len(message), 700)
            for button in panel.query(Button):
                self.assertLess(button.region.y, prompt_y)
                self.assertLessEqual(button.region.y + button.region.height, panel_bottom)

            await pilot.press("r")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), [{"type": "reject"}])

    async def test_json_edit_flow_returns_edits_and_rejects_invalid_json(self) -> None:
        """Edited approval args should save valid JSON and reject invalid JSON."""
        app = make_app()
        action = {"name": "write_file", "args": {"file_path": "test.txt", "content": "hello"}}

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app.edit_decision(action))
            await pilot.pause()
            editor = app.query_one("#prompt-panel-editor", TextArea)
            await wait_until(lambda: editor.has_focus)
            self.assertTrue(editor.has_focus)
            editor.text = '{"file_path": "test.txt", "content": "bye"}'
            await wait_until(lambda: len(list(app.query_one(PromptPanel).query("#prompt-save"))) > 0)
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Save", "Cancel"],
            )
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            app.query_one("#prompt-save", Button).press()
            decision = await asyncio.wait_for(task, timeout=2)

            self.assertEqual(
                decision,
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "write_file",
                        "args": {"file_path": "test.txt", "content": "bye"},
                    },
                },
            )

            invalid_task = asyncio.create_task(app.edit_decision(action))
            await pilot.pause()
            app.query_one("#prompt-panel-editor", TextArea).text = "{bad json"
            await wait_until(lambda: len(list(app.query_one(PromptPanel).query("#prompt-save"))) > 0)
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Save", "Cancel"],
            )
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            app.query_one("#prompt-save", Button).press()
            self.assertEqual(await asyncio.wait_for(invalid_task, timeout=2), {"type": "reject"})

    async def test_tool_result_attaches_to_existing_tool_call(self) -> None:
        """Tool results should update the existing tool block instead of adding a new panel."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.tool_call("eval", {"code": "1 + 1"}, call_id="call-1")
            app.tool_result("eval", "<result>2</result>", call_id="call-1")
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            self.assertEqual(len(blocks), 2)
            text = renderable_plain(blocks[-1])
            self.assertIn("call:", text)
            self.assertIn("output:", text)

    async def test_tool_call_delta_updates_in_place_when_final_call_arrives(self) -> None:
        """Draft tool-call args should finalize in the same transcript block."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.tool_call_delta("read_file", {"path": "REA"}, call_id="call-read")
            await pilot.pause()
            draft_blocks = list(app.query_one(ChatLog).children)
            draft = renderable_plain(draft_blocks[-1])
            self.assertIn("draft:", draft)
            self.assertIn("REA", draft)

            app.tool_call("read_file", {"path": "README.md"}, call_id="call-read")
            await pilot.pause()
            final_blocks = list(app.query_one(ChatLog).children)
            final = renderable_plain(final_blocks[-1])

            self.assertEqual(len(final_blocks), len(draft_blocks))
            self.assertIn("call:", final)
            self.assertNotIn("draft:", final)
            self.assertIn("README.md", final)

    async def test_cancel_turn_discards_tool_call_drafts(self) -> None:
        """Tool-call drafts from a cancelled turn should not be reused."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.tool_call_delta("read_file", {"path": "old"})
            await pilot.pause()
            old_draft = app.query_one(ChatLog).children[-1]
            self.assertIn("old", renderable_plain(old_draft))

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            self.assertNotIn(old_draft, list(app.query_one(ChatLog).children))

            app.tool_call_delta("read_file", {"path": "new"})
            await pilot.pause()
            tool_blocks = [block for block in app.query_one(ChatLog).children if "tool-call" in block.classes]

            self.assertEqual(len(tool_blocks), 1)
            self.assertIn("new", renderable_plain(tool_blocks[0]))
            self.assertNotIn("old", renderable_plain(tool_blocks[0]))

    async def test_delegation_delta_updates_in_place_when_final_call_arrives(self) -> None:
        """Draft task requests should finalize in the same transcript block."""
        app = make_app()
        draft_call = [{"name": "task", "args": {"description": "summarize"}}]
        final_call = [{"name": "task", "args": {"description": "summarize README"}}]

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_delta(draft_call)
            await pilot.pause()
            draft_blocks = list(app.query_one(ChatLog).children)
            draft = renderable_plain(draft_blocks[-1])
            self.assertIn("preparing 1 subagent", draft)
            self.assertIn("summarize", draft)

            app.delegation_started(final_call)
            await pilot.pause()
            final_blocks = list(app.query_one(ChatLog).children)
            final = renderable_plain(final_blocks[-1])

            self.assertEqual(len(final_blocks), len(draft_blocks))
            self.assertIn("delegating to 1 subagent", final)
            self.assertIn("summarize README", final)

    async def test_cancel_turn_discards_delegation_drafts(self) -> None:
        """Delegation drafts from a cancelled turn should not promote later."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_delta([{"id": "task-old", "name": "task", "args": {"description": "old task"}}])
            await pilot.pause()
            old_draft = app.query_one(ChatLog).children[-1]
            self.assertIn("old task", renderable_plain(old_draft))

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            self.assertNotIn(old_draft, list(app.query_one(ChatLog).children))

            app.delegation_started([{"id": "task-new", "name": "task", "args": {"description": "new task"}}])
            await pilot.pause()
            delegation_blocks = [block for block in app.query_one(ChatLog).children if "delegation" in block.classes]

            self.assertEqual(len(delegation_blocks), 1)
            self.assertIn("new task", renderable_plain(delegation_blocks[0]))
            self.assertNotIn("old task", renderable_plain(delegation_blocks[0]))

    async def test_empty_delegation_delta_renders_info_placeholder(self) -> None:
        """Empty task drafts should show a live info placeholder, not an empty task box."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_delta([{"id": "task-1", "name": "task", "args": {}}])
            await pilot.pause()
            blocks = list(app.query_one(ChatLog).children)
            block = blocks[-1]
            rendered = renderable_plain(block)

            self.assertEqual(str(getattr(block, "border_title", "")), "info")
            self.assertIn("info", block.classes)
            self.assertIn("preparing subagent tasks...", rendered)
            self.assertNotIn("request:", rendered)

    async def test_delegation_delta_promotes_info_placeholder_to_task(self) -> None:
        """The first readable task request should replace the info placeholder in place."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_delta([{"id": "task-1", "name": "task", "args": {}}])
            await pilot.pause()
            info_blocks = list(app.query_one(ChatLog).children)

            app.delegation_delta(
                [
                    {"id": "task-1", "name": "task", "args": {"description": "scary story"}},
                    {"id": "task-2", "name": "task", "args": {}},
                ]
            )
            await pilot.pause()
            task_blocks = list(app.query_one(ChatLog).children)
            block = task_blocks[-1]
            rendered = renderable_plain(block)

            self.assertEqual(len(task_blocks), len(info_blocks))
            self.assertEqual(str(getattr(block, "border_title", "")), "task")
            self.assertIn("delegation", block.classes)
            self.assertIn("scary story", rendered)
            self.assertIn("drafting request...", rendered)

    async def test_delegation_started_calls_update_one_task_block(self) -> None:
        """One-by-one task calls should coalesce into one visible task block."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_started([{"id": "task-1", "name": "task", "args": {"description": "scary story"}}])
            await pilot.pause()
            first_blocks = list(app.query_one(ChatLog).children)

            app.delegation_started([{"id": "task-2", "name": "task", "args": {"description": "funny story"}}])
            await pilot.pause()
            second_blocks = list(app.query_one(ChatLog).children)
            final = renderable_plain(second_blocks[-1])

            self.assertEqual(len(second_blocks), len(first_blocks))
            self.assertIn("delegating to 2 subagents", final)
            self.assertIn("scary story", final)
            self.assertIn("funny story", final)

    async def test_live_subagent_panel_removes_existing_delegation_draft(self) -> None:
        """Opening the panel should remove the transient task draft for the same work."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.delegation_delta([{"id": "task-1", "name": "task", "args": {"description": "judge haiku"}}])
            await pilot.pause()
            draft = app.query_one(ChatLog).children[-1]
            self.assertIn("judge haiku", renderable_plain(draft))

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "judge haiku")
            await pilot.pause()

            delegation_blocks = [block for block in app.query_one(ChatLog).children if "delegation" in block.classes]
            panel_text = renderable_plain(app.query_one(SubagentsPanel).query_one("#subagents-tasks", Static))

            self.assertNotIn(draft, list(app.query_one(ChatLog).children))
            self.assertEqual(delegation_blocks, [])
            self.assertIn("general-purpose [one]", panel_text)
            self.assertIn("judge haiku", panel_text)

    async def test_live_subagent_panel_suppresses_sequential_delegation_bubbles(self) -> None:
        """Sequential task starts should update the panel instead of stacking task bubbles."""
        app = make_app()

        async with app.run_test(size=(140, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            for index in range(1, 4):
                calls = [
                    {
                        "id": f"task-{task_index}",
                        "name": "task",
                        "args": {"description": f"judge haiku pair {task_index}"},
                    }
                    for task_index in range(1, index + 1)
                ]
                app.subagent_started(f"general-purpose [{index}]", f"judge haiku pair {index}")
                app.delegation_started(calls)
                await pilot.pause()

            delegation_blocks = [block for block in app.query_one(ChatLog).children if "delegation" in block.classes]
            panel_text = renderable_plain(app.query_one(SubagentsPanel).query_one("#subagents-tasks", Static))

            self.assertEqual(delegation_blocks, [])
            self.assertEqual(panel_text.count("RUNNING"), 3)
            self.assertIn("general-purpose [1]", panel_text)
            self.assertIn("general-purpose [2]", panel_text)
            self.assertIn("general-purpose [3]", panel_text)
            self.assertIn("judge haiku pair 3", panel_text)

    async def test_subagent_request_update_fills_running_block(self) -> None:
        """A subagent block that started blank should accept late request text."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose [one]", "")
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            self.assertNotIn("request:", renderable_plain(block))

            app.subagent_request_updated("general-purpose [one]", "write scary story")
            await pilot.pause()

            text = renderable_plain(block)
            self.assertIn("request:", text)
            self.assertIn("write scary story", text)

    async def test_dynamic_subagent_origin_is_quiet(self) -> None:
        """Origin metadata should not be shown as user-visible classification."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose [one]", "", origin="dynamic_tool_subagent")
            await pilot.pause()

            block = app.query_one(ChatLog).children[-1]
            title = str(getattr(block, "border_title", "")).replace("\\", "")
            text = renderable_plain(block)

            self.assertEqual(title, "subagent - general-purpose [one]")
            self.assertNotIn("source:", text)
            self.assertNotIn("eval/tool-created subagent", text)
            self.assertNotEqual(title, "task")

    async def test_late_subagent_request_clears_dynamic_origin(self) -> None:
        """A late top-level task request should restore ordinary subagent display."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose [one]", "", origin="dynamic_tool_subagent")
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]

            app.subagent_request_updated("general-purpose [one]", "write scary story")
            await pilot.pause()

            title = str(getattr(block, "border_title", "")).replace("\\", "")
            text = renderable_plain(block)

            self.assertEqual(title, "subagent - general-purpose [one]")
            self.assertNotIn("eval/tool-created subagent", text)
            self.assertIn("write scary story", text)

    async def test_tool_result_waits_for_call_and_then_attaches(self) -> None:
        """Out-of-order tool results should attach once the tool call is rendered."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.tool_result("eval", "<result>5</result>\nsecond line", call_id="call-2")
            await pilot.pause()
            self.assertEqual(len(app.query_one(ChatLog).children), 1)

            app.tool_call("eval", {"code": "2 + 3"}, call_id="call-2")
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            self.assertEqual(len(blocks), 2)
            text = renderable_plain(blocks[-1])
            self.assertIn("call:", text)
            self.assertIn("output:", text)
            self.assertIn("second line", text)

    async def test_tool_results_without_ids_attach_by_name_order(self) -> None:
        """Tool results without ids should attach to the oldest unresolved matching call."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.tool_call("eval", {}, call_id="call-empty")
            app.tool_call("eval", {"code": "2 + 2"}, call_id="call-code")
            app.tool_result("eval", "missing code")
            app.tool_result("eval", "<result>4</result>")
            await pilot.pause()

            blocks = list(app.query_one(ChatLog).children)
            first = renderable_plain(blocks[-2])
            second = renderable_plain(blocks[-1])
            self.assertIn("call: {}", first)
            self.assertIn("output:", first)
            self.assertIn("missing code", first)
            self.assertIn("call:", second)
            self.assertIn("2 + 2", second)
            self.assertIn("<result>4</result>", second)

    async def test_subagent_labels_keep_cute_suffix_and_running_animation(self) -> None:
        """Subagent display should keep readable nicknames and animate running status."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose", "look for README")
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            title = str(getattr(block, "border_title", "")).replace("\\", "")
            first = renderable_plain(block)

            self.assertIn("general-purpose [", title)
            self.assertIn("RUNNING", first)

            app.query_one(ChatLog).tick_subagents()
            await pilot.pause()
            second = renderable_plain(block)

            self.assertNotEqual(first, second)
            app.subagent_finished("general-purpose", "README.md\nDone.")
            await pilot.pause()
            done = renderable_plain(block)
            self.assertIn("DONE", done)
            self.assertIn("output:", done)
            self.assertIn("README.md", done)

    async def test_repeated_unsuffixed_subagents_get_separate_blocks(self) -> None:
        """Live subagents should render in the bottom panel while they run."""
        app = make_app()

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "inspect README")
            app.subagent_started("general-purpose [two]", "inspect pyproject")
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            rendered = renderable_plain(panel.query_one("#subagents-tasks", Static))
            chat_blocks = [block for block in app.query_one(ChatLog).children if "subagent" in block.classes]

            self.assertTrue(panel.display)
            self.assertEqual(chat_blocks, [])
            self.assertEqual(rendered.count("RUNNING"), 2)
            self.assertGreaterEqual(rendered.count("["), 2)
            self.assertIn("inspect README", rendered)
            self.assertIn("inspect pyproject", rendered)
            self.assertNotIn("MODEL", rendered)
            self.assertNotIn("Group", renderable_plain(panel.query_one("#subagents-groups", Static)))

    async def test_subagent_panel_uses_symbol_controls_without_ctrl_g(self) -> None:
        """The panel should use compact visible controls only."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "inspect README")
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            toggle = panel.query_one("#subagents-panel-toggle", Static)
            close = panel.query_one("#subagents-panel-close", Static)
            self.assertEqual(renderable_plain(toggle), "[-]")
            self.assertEqual(renderable_plain(close), "x")
            self.assertFalse(close.display)
            self.assertFalse(any(binding.key == "ctrl+g" for binding in app.BINDINGS))

            panel.toggle()
            await pilot.pause()
            self.assertEqual(renderable_plain(toggle), "[+]")
            self.assertFalse(panel.query_one("#subagents-panel-body").display)
            first_header = renderable_plain(panel.query_one("#subagents-panel-header", Static))
            panel.tick()
            await pilot.pause()
            second_header = renderable_plain(panel.query_one("#subagents-panel-header", Static))
            self.assertNotEqual(first_header[0], second_header[0])
            self.assertIn("subagents", second_header)

            app.subagent_started("general-purpose [two]", "inspect pyproject")
            await pilot.pause()
            self.assertEqual(renderable_plain(toggle), "[-]")
            self.assertTrue(panel.query_one("#subagents-panel-body").display)

            panel.toggle()
            await pilot.pause()
            self.assertEqual(renderable_plain(toggle), "[+]")
            self.assertFalse(panel.query_one("#subagents-panel-body").display)

            await pilot.click("#subagents-panel-header")
            await pilot.pause()
            self.assertEqual(renderable_plain(toggle), "[-]")
            self.assertTrue(panel.query_one("#subagents-panel-body").display)

            app.subagent_finished("general-purpose [one]", "README summary")
            app.subagent_finished("general-purpose [two]", "pyproject summary")
            await pilot.pause()
            self.assertTrue(close.display)

    async def test_subagent_panel_keeps_fixed_columns_across_terminal_widths(self) -> None:
        """Task truncation should preserve one-line status and time columns."""
        identity = "general-purpose [persimmon-sparrow]"
        long_hint = "Inspect README and explain " + "every relevant architecture detail " * 8
        app = make_app()

        async with app.run_test(size=(160, 24)) as pilot:
            await pilot.pause()

            app.eval_subagent_started(identity, long_hint, eval_id="eval-a", row_id="row-a")
            app.eval_subagent_started(
                "general-purpose [second-runner]",
                long_hint,
                eval_id="eval-a",
                row_id="row-b",
            )
            await pilot.pause()

            for width in (160, 100, 50):
                with self.subTest(width=width):
                    if width != 160:
                        await pilot.resize_terminal(width, 24)
                    await pilot.pause()

                    panel = app.query_one(SubagentsPanel)
                    rendered = renderable_plain(panel.query_one("#subagents-tasks", Static))
                    self.assertIn("Group 1", renderable_plain(panel.query_one("#subagents-groups", Static)))
                    lines = rendered.rstrip("\n").splitlines()
                    self.assertEqual(len(lines), 3)
                    status_col = lines[0].index("STATUS")
                    time_col = lines[0].index("TIME")

                    for row in lines[1:]:
                        self.assertEqual(row[status_col : status_col + 11].strip(), "RUNNING")
                        self.assertTrue(row[time_col:].strip().endswith("s"))
                        task_cell = row[3:status_col].rstrip()
                        self.assertTrue(task_cell.endswith("..."))

    async def test_subagent_panel_styles_identity_with_purple_type_colour(self) -> None:
        """The subagent type and coolname should use MIRA's purple identity colour."""
        identity = "general-purpose [persimmon-sparrow]"
        text = Text()
        append_task_cell(text, SubagentRecord(key="one", name=identity, hint="inspect README"), 80)
        start = text.plain.index(identity)
        end = start + len(identity)
        matching = [
            span for span in text.spans if span.start <= start and span.end >= end and "#B7A4E8" in str(span.style).upper()
        ]

        self.assertTrue(matching)

    def test_subagent_task_truncation_uses_terminal_cell_width(self) -> None:
        """Wide task characters should retain exact table width and ellipsis."""
        value = truncate_cells("任务说明 with extra detail", 12)

        self.assertEqual(cell_len(value), 12)
        self.assertTrue(value.endswith("..."))

    def test_subagent_group_status_icons_have_status_colours(self) -> None:
        """Group status icons should mirror task-row status colours."""
        self.assertEqual(group_status_icon(done=1, total=1, failed=0, cancelled=0), ("v", "bold green"))
        self.assertEqual(group_status_icon(done=1, total=1, failed=1, cancelled=0), ("x", "bold red"))
        self.assertEqual(group_status_icon(done=1, total=1, failed=0, cancelled=1), ("-", "bold yellow"))
        self.assertEqual(group_status_icon(done=0, total=1, failed=0, cancelled=0), ("*", "bold yellow"))

    def test_subagent_group_status_spans_colour_only_the_icon(self) -> None:
        """The left group list should colour v/x/- without tinting labels."""
        panel = SubagentsPanel()
        panel.start_subagent("general-purpose [done]", "done", eval_id="eval-done", row_id="done")
        panel.start_subagent("general-purpose [error]", "error", eval_id="eval-error", row_id="error")
        panel.start_subagent("general-purpose [cancelled]", "cancelled", eval_id="eval-cancelled", row_id="cancelled")
        panel.finish_subagent("general-purpose [done]", eval_id="eval-done", row_id="done")
        panel.finish_subagent(
            "general-purpose [error]",
            "failed",
            eval_id="eval-error",
            row_id="error",
            status="ERROR",
        )
        panel.finish_subagent(
            "general-purpose [cancelled]",
            "cancelled",
            eval_id="eval-cancelled",
            row_id="cancelled",
            status="CANCELLED",
        )

        text = panel._render_groups()

        self.assert_styled_char(text, "v", "bold green")
        self.assert_styled_char(text, "x", "bold red")
        self.assert_styled_char(text, "-", "bold yellow")
        self.assertFalse(
            [
                span
                for span in text.spans
                if "Group" in text.plain[span.start : span.end] and any(color in str(span.style) for color in ("green", "red", "yellow"))
            ]
        )

    async def test_repeated_unsuffixed_subagent_errors_update_matching_blocks(self) -> None:
        """Error-like subagent output should finish only the matching active bubble."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose", "inspect README")
            app.subagent_started("general-purpose", "inspect pyproject")
            await pilot.pause()
            blocks = [block for block in app.query_one(ChatLog).children if "subagent" in block.classes]

            app.subagent_finished("general-purpose", "error: subagent_type: Field required")
            await pilot.pause()
            first = renderable_plain(blocks[0])
            second = renderable_plain(blocks[1])

            self.assertIn("DONE", first)
            self.assertIn("error: subagent_type: Field required", first)
            self.assertIn("RUNNING", second)
            self.assertNotIn("subagent_type: Field required", second)

            app.subagent_finished("general-purpose", "pyproject summary")
            await pilot.pause()
            second_done = renderable_plain(blocks[1])

            self.assertIn("DONE", second_done)
            self.assertIn("pyproject summary", second_done)

    async def test_subagent_panel_collapses_closes_and_reopens_for_new_activity(self) -> None:
        """Completed panel state should collapse on the next prompt and reset on new work."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "inspect README")
            app.subagent_finished("general-purpose [one]", "README summary")
            app.stop_subagent_live()
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            self.assertTrue(panel.display)
            self.assertTrue(panel.query_one("#subagents-panel-body").display)

            panel.prepare_turn()
            await pilot.pause()
            self.assertTrue(panel.display)
            self.assertFalse(panel.query_one("#subagents-panel-body").display)

            panel.close()
            await pilot.pause()
            self.assertFalse(panel.display)

            app.start_subagent_live()
            app.subagent_started("general-purpose [two]", "py")
            await pilot.pause()

            rendered = renderable_plain(panel.query_one("#subagents-tasks", Static))
            self.assertTrue(panel.display)
            self.assertTrue(panel.query_one("#subagents-panel-body").display)
            self.assertIn("py", rendered)
            self.assertNotIn("inspect README", rendered)

    async def test_eval_subagents_are_grouped_without_showing_eval_ids(self) -> None:
        """Eval-created subagents should use user-facing group labels."""
        app = make_app()

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()

            app.eval_subagent_started(
                "general-purpose [haiku 1]",
                "Generate haiku 1",
                eval_id="eval-round-a",
                row_id="task-a",
                model="claude-haiku",
            )
            app.eval_subagent_finished(
                "general-purpose [haiku 1]",
                eval_id="eval-round-a",
                row_id="task-a",
                duration_ms=1500,
            )
            app.eval_subagent_started(
                "general-purpose [final]",
                "judge",
                eval_id="eval-round-b",
                row_id="task-b",
                model="claude-haiku",
            )
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            header = renderable_plain(panel.query_one("#subagents-panel-header", Static))
            groups = renderable_plain(panel.query_one("#subagents-groups", Static))
            tasks = renderable_plain(panel.query_one("#subagents-tasks", Static))

            self.assertIn("dynamic subagents", header)
            self.assertIn("2 groups", header)
            self.assertIn("Group 1", groups)
            self.assertIn("Group 2", groups)
            self.assertIn("RUNNING", tasks)
            self.assertIn("general-purpose [", tasks)
            self.assertIn("judge", tasks)
            self.assertNotIn("MODEL", tasks)
            self.assertNotIn("claude-haiku", tasks)
            self.assertNotIn("final]", tasks)
            self.assertNotIn("eval-round", groups)
            self.assertNotIn("eval-round", tasks)

    async def test_eval_subagent_failure_retry_reuses_group(self) -> None:
        """A failed eval batch retry should reuse the same user-facing group."""
        app = make_app()

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()

            app.eval_subagent_started(
                "general-purpose [draft]",
                "write long story description that should not become identity",
                eval_id="eval-failed",
                row_id="row-a",
                label="draft",
            )
            app.eval_subagent_cancelled(
                "general-purpose [draft]",
                "tool failed",
                eval_id="eval-failed",
                row_id="row-a",
            )
            app.eval_subagent_started(
                "general-purpose [retry]",
                "write long story description that should not become identity",
                eval_id="eval-retry",
                row_id="row-b",
                label="retry",
            )
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            groups = renderable_plain(panel.query_one("#subagents-groups", Static))
            tasks = renderable_plain(panel.query_one("#subagents-tasks", Static))

            self.assertIn("Group 1", groups)
            self.assertNotIn("Group 2", groups)
            self.assertIn("retry", tasks)
            self.assertNotIn("draft]", tasks)
            self.assertNotIn("eval-", groups)
            self.assertNotIn("eval-", tasks)

    async def test_mixed_regular_and_eval_subagents_share_panel_sections(self) -> None:
        """Mixed workflows should show regular tasks plus eval groups in one panel."""
        app = make_app()

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "inspect README")
            app.eval_subagent_started(
                "general-purpose [judge]",
                "judge README summary",
                eval_id="eval-judge",
                row_id="judge-a",
                label="judge",
            )
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            header = renderable_plain(panel.query_one("#subagents-panel-header", Static))
            groups = renderable_plain(panel.query_one("#subagents-groups", Static))

            self.assertIn("subagents", header)
            self.assertNotIn("dynamic subagents", header)
            self.assertIn("Tasks", groups)
            self.assertIn("Group 1", groups)

    async def test_subagent_group_list_click_selects_visible_group_rows(self) -> None:
        """Clicking the left group list should switch the visible task table."""
        app = make_app()

        async with app.run_test(size=(160, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [regular]", "inspect README")
            app.eval_subagent_started(
                "general-purpose [judge]",
                "judge README summary",
                eval_id="eval-judge",
                row_id="judge-a",
                label="judge",
            )
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            tasks_widget = panel.query_one("#subagents-tasks", Static)
            initial = renderable_plain(tasks_widget)
            self.assertIn("judge", initial)
            self.assertNotIn("general-purpose [regular]", initial)

            await pilot.click("#subagents-groups", offset=(1, 1))
            await pilot.pause()
            regular = renderable_plain(tasks_widget)
            self.assertIn("general-purpose [regular]", regular)
            self.assertNotIn("general-purpose [judge]", regular)

            await pilot.click("#subagents-groups", offset=(1, 2))
            await pilot.pause()
            grouped = renderable_plain(tasks_widget)
            self.assertIn("judge", grouped)
            self.assertNotIn("general-purpose [regular]", grouped)

    async def test_subagent_group_line_header_and_invalid_rows_do_not_change_selection(self) -> None:
        """Header and invalid group lines should leave the selection alone."""
        panel = SubagentsPanel()
        panel.start_subagent("general-purpose [regular]", "inspect README")
        panel.start_subagent(
            "general-purpose [judge]",
            "judge README summary",
            eval_id="eval-judge",
            row_id="judge-a",
            label="judge",
        )

        before = [record.name for record in panel._displayed_records()]
        panel.select_group_line(0)
        self.assertEqual([record.name for record in panel._displayed_records()], before)
        panel.select_group_line(99)
        self.assertEqual([record.name for record in panel._displayed_records()], before)
        panel.select_group_line(-1)
        self.assertEqual([record.name for record in panel._displayed_records()], before)

    async def test_live_subagent_panel_marks_running_rows_cancelled(self) -> None:
        """Cancelling a live turn should stop panel spinners and mark rows terminal."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.start_subagent_live()
            app.subagent_started("general-purpose [one]", "look for dead code")
            await pilot.pause()

            panel = app.query_one(SubagentsPanel)
            running = renderable_plain(panel.query_one("#subagents-tasks", Static))
            self.assertIn("RUNNING", running)

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            cancelled = renderable_plain(panel.query_one("#subagents-tasks", Static))
            self.assertIn("CANCELLED", cancelled)
            self.assertNotIn("RUNNING", cancelled)
            self.assertFalse(panel.has_running_subagents())

            panel.tick()
            await pilot.pause()
            self.assertEqual(renderable_plain(panel.query_one("#subagents-tasks", Static)), cancelled)

    async def test_cancelled_subagent_blocks_stop_animating(self) -> None:
        """Cancelling a turn should leave active subagents in a terminal state."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.subagent_started("general-purpose", "look for dead code")
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            running = renderable_plain(block)
            self.assertIn("RUNNING", running)

            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            cancelled = renderable_plain(block)
            self.assertIn("CANCELLED", cancelled)
            self.assertNotIn("RUNNING", cancelled)
            self.assertFalse(app.query_one(ChatLog).has_running_subagents())

            app.query_one(ChatLog).tick_subagents()
            await pilot.pause()
            self.assertEqual(renderable_plain(block), cancelled)
            self.assertEqual(app.status_state, "cancelling")

    async def test_restore_session_with_subagent_output_keeps_single_subagent_block(self) -> None:
        """Persisted subagent output should restore inside the existing compact block."""
        session = {
            "id": "thread-subagent-output",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 1,
            "dashboard": {},
            "events": [
                {
                    "id": 1,
                    "type": "subagent",
                    "mode": "action",
                    "name": "general-purpose [one]",
                    "status": "DONE",
                    "task_input": "find dead code",
                    "output": "No dead code found.",
                }
            ],
        }
        app = make_app(session=session)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            subagent_blocks = [block for block in app.query_one(ChatLog).children if "subagent" in block.classes]
            self.assertFalse(app.query_one(SubagentsPanel).display)
            self.assertEqual(len(subagent_blocks), 1)
            rendered = renderable_plain(subagent_blocks[0])
            self.assertIn("DONE", rendered)
            self.assertIn("No dead code found.", rendered)
            duplicate_blocks = [
                block
                for block in app.query_one(ChatLog).children
                if ("assistant" in block.classes or "tool-result" in block.classes)
                and "No dead code found." in renderable_plain(block)
            ]
            self.assertEqual(duplicate_blocks, [])

    async def test_ask_user_choice_and_open_text_flow(self) -> None:
        """ask_user should support both concrete choices and open-ended text."""
        app = make_app()
        choice_interrupt = {
            "type": "ask_user",
            "question": "Which path?",
            "options": ["Use A", "Use B", ASK_USER_OPEN_OPTION],
        }
        open_interrupt = {
            "type": "ask_user",
            "question": "What next?",
            "options": ["Use A", ASK_USER_OPEN_OPTION],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            choice_task = asyncio.create_task(app.ask_user(choice_interrupt))
            await pilot.pause()
            await pilot.press("2")
            self.assertEqual(await asyncio.wait_for(choice_task, timeout=2), "Use B")

            open_task = asyncio.create_task(app.ask_user(open_interrupt))
            await pilot.pause()
            await wait_until(lambda: len(list(app.query_one(PromptPanel).query(Button))) >= 2)
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertTrue(str(buttons[-1].label).startswith("2 Tell MIRA"))
            await pilot.press("2")
            await pilot.pause()
            answer = app.query_one("#prompt-panel-input", Input)
            await wait_until(lambda: answer.has_focus)
            self.assertTrue(answer.has_focus)
            answer.value = "Try the safer patch"
            await pilot.press("enter")
            self.assertEqual(await asyncio.wait_for(open_task, timeout=2), "Try the safer patch")

    async def test_text_prompt_action_buttons_use_action_labels(self) -> None:
        """Freeform PromptPanel actions should use readable action labels."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app._prompt_text("Question", "Tell MIRA what to do differently"))
            await pilot.pause()
            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == 2)

            buttons = list(panel.query(Button))
            self.assertEqual([button.label.plain for button in buttons], ["Submit", "Cancel"])
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))

            app.query_one("#prompt-cancel", Button).press()
            self.assertIsNone(await asyncio.wait_for(task, timeout=2))

    async def test_ask_user_uses_vertical_buttons_and_preserves_question(self) -> None:
        """ask_user choices should be vertical buttons separate from the question text."""
        app = make_app()
        interrupt = {
            "type": "ask_user",
            "question": "Which path?\nPick one.",
            "options": ["Minimal change (Recommended)", "Focused refactor", "Planning only"],
        }

        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app.ask_user(interrupt))
            await pilot.pause()
            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == 4)
            await wait_until(lambda: all(row.region.height == 1 for row in panel.query(".prompt-panel-button-row")))

            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            buttons = list(panel.query(Button))
            rows = list(panel.query(".prompt-panel-button-row"))
            row_heights = {row.region.height for row in rows}

            self.assertEqual(message, "Which path?\nPick one.")
            self.assertNotIn("Minimal change", message)
            self.assertEqual(len(rows), 4)
            self.assertEqual(row_heights, {1})
            self.assertTrue(all(button.compact for button in buttons))
            self.assertTrue(all(not button.flat for button in buttons))
            self.assertTrue(all(str(button.label) for button in buttons))
            self.assertTrue(all(button.styles.content_align_horizontal == "left" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            self.assertTrue(buttons[0].has_focus)
            self.assertEqual(buttons[0].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
            self.assertEqual(buttons[0].styles.color, PROMPT_BUTTON_FOCUS_COLOR)
            self.assertTrue(all(button.region.width >= row.region.width - 2 for button, row in zip(buttons, rows)))
            self.assertTrue(str(buttons[-1].label).startswith("4 Tell MIRA"))

            await pilot.press("1")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "Minimal change (Recommended)")

    async def test_ask_user_large_choice_set_scrolls_and_remains_selectable(self) -> None:
        """ask_user should keep larger choice sets accessible in a vertical scroll area."""
        app = make_app()
        interrupt = {
            "type": "ask_user",
            "question": "Choose a target.",
            "options": [f"Target {index}" for index in range(1, 11)],
        }

        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app.ask_user(interrupt))
            await pilot.pause()
            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == 11)

            button_area = app.query_one("#prompt-panel-buttons", Vertical)
            rows = list(panel.query(".prompt-panel-button-row"))
            self.assertEqual(len(rows), 11)
            self.assertLessEqual(button_area.region.height, 4)

            await wait_until(lambda: len(list(panel.query(Button))) == 11 and list(panel.query(Button))[0].has_focus)
            for _ in range(9):
                await pilot.press("down")
                await pilot.pause()
            await pilot.press("enter")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "Target 10")

    async def test_ask_user_vertical_buttons_fit_narrow_and_truncate_long_labels(self) -> None:
        """ask_user should not overflow narrow prompts with long labels."""
        app = make_app()
        interrupt = {
            "type": "ask_user",
            "question": "What should be tested?",
            "options": [
                "Test database initialization and schema creation",
                "Test email ingestion without segmentation",
                "Test processing extraction and reporting",
            ],
        }

        async with app.run_test(size=(48, 30)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app.ask_user(interrupt))
            await pilot.pause()
            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == 4)
            await wait_until(lambda: all(row.region.height == 1 for row in panel.query(".prompt-panel-button-row")))
            await wait_until(lambda: any(str(button.label).endswith("...") for button in panel.query(Button)))
            buttons = list(panel.query(Button))
            rows = list(panel.query(".prompt-panel-button-row"))
            panel_right = panel.region.x + panel.region.width

            self.assertTrue(all(button.region.x + button.region.width <= panel_right for button in buttons))
            self.assertTrue(all(row.region.height == 1 for row in rows))
            self.assertTrue(all(button.compact for button in buttons))
            self.assertTrue(all(not button.flat for button in buttons))
            self.assertTrue(all(str(button.label) for button in buttons))
            self.assertTrue(any(str(button.label).endswith("...") for button in buttons[:-1]))
            self.assertTrue(str(buttons[-1].label).startswith("4 Tell MIRA"))

            await pilot.press("4")
            await pilot.pause()
            answer = app.query_one("#prompt-panel-input", Input)
            await wait_until(lambda: answer.has_focus)
            answer.value = "Run a shorter smoke test"
            await pilot.press("enter")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "Run a shorter smoke test")

    async def test_choice_prompt_buttons_wrap_in_narrow_viewports(self) -> None:
        """Shared choice dialogs should wrap options instead of hiding later buttons."""
        app = make_app()
        choices = [(str(index), f"{index} Option {index}") for index in range(1, 9)]

        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app._prompt_choice("Choices", "Pick one.", choices))
            await pilot.pause()

            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == len(choices))
            await wait_until(
                lambda: len({row.region.y for row in panel.query(".prompt-panel-button-row")}) > 1
            )
            await wait_until(lambda: all(row.region.height == 1 for row in panel.query(".prompt-panel-button-row")))
            rows = list(panel.query(".prompt-panel-button-row"))
            buttons = list(panel.query(Button))
            row_positions = {row.region.y for row in rows}
            panel_right = panel.region.x + panel.region.width
            self.assertGreater(len(rows), 1)
            self.assertGreater(len(row_positions), 1, [row.region for row in rows])
            self.assertEqual(str(buttons[-1].label), "8 Option 8")
            self.assertTrue(all(row.region.height == 1 for row in rows))
            self.assertTrue(all(button.compact for button in buttons))
            self.assertTrue(all(not button.flat for button in buttons))
            self.assertTrue(all(str(button.label) for button in buttons))
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            self.assertEqual(buttons[0].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
            self.assertNotEqual(buttons[1].styles.background, PROMPT_BUTTON_FOCUS_BACKGROUND)
            self.assertTrue(
                all(button.region.x + button.region.width <= panel_right for button in buttons),
                {
                    "panel": panel.region,
                    "rows": [row.region for row in rows],
                    "buttons": [(button.region, str(button.label)) for button in buttons],
                },
            )

            await pilot.press("8")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "8")

    async def test_choice_prompt_reflows_to_use_wide_viewports(self) -> None:
        """Choice rows should be based on measured width, not a static row guess."""
        choices = [(str(index), f"{index} Option {index}") for index in range(1, 9)]

        async def row_count(width: int) -> int:
            app = make_app()
            async with app.run_test(size=(width, 40)) as pilot:
                await pilot.pause()
                task = asyncio.create_task(app._prompt_choice("Choices", "Pick one.", choices))
                await pilot.pause()
                panel = app.query_one(PromptPanel)
                await wait_until(lambda: len(list(panel.query(Button))) == len(choices))
                count = len(list(panel.query(".prompt-panel-button-row")))
                await pilot.press("8")
                self.assertEqual(await asyncio.wait_for(task, timeout=2), "8")
                return count

        narrow_rows = await row_count(80)
        wide_rows = await row_count(140)
        self.assertLess(wide_rows, narrow_rows)

    async def test_wrapped_choice_prompt_down_arrow_moves_between_rows(self) -> None:
        """Arrow navigation should understand wrapped prompt button rows."""
        app = make_app()
        choices = [(letter, f"{letter} Choice") for letter in ("a", "b", "c", "d", "e", "f")]

        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()

            task = asyncio.create_task(app._prompt_choice("Choices", "Pick one.", choices))
            await pilot.pause()

            panel = app.query_one(PromptPanel)
            await wait_until(lambda: len(list(panel.query(Button))) == len(choices))
            await wait_until(lambda: any(button.has_focus for button in panel.query(Button)))
            await wait_until(lambda: not panel._reflow_running and len(panel._button_rows) >= 2)
            rows = list(panel.query(Horizontal))
            self.assertGreaterEqual(len(rows), 2)
            first_row_buttons = list(rows[0].query(Button))
            second_row_buttons = list(rows[1].query(Button))
            self.assertTrue(first_row_buttons[0].has_focus)

            await pilot.press("down")
            await pilot.pause()
            await wait_until(lambda: list(panel.query(Horizontal))[1].query(Button).first().has_focus)
            rows = list(panel.query(Horizontal))
            second_row_buttons = list(rows[1].query(Button))
            self.assertTrue(second_row_buttons[0].has_focus)

            await pilot.press("enter")
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "c")

    async def test_present_plan_bubble_discards_to_inactive_history(self) -> None:
        """Structured plans should render with real buttons and resolve in place."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Ephemeral Structured Planning",
            "summary": ["Use a temporary plan bubble."],
            "key_changes": ["Add present_plan.", "Remove /plans."],
            "test_plan": ["Verify the plan bubble controls."],
            "assumptions": ["Plans are temporary UI artifacts."],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            self.assertEqual(await app.present_plan(interrupt), "Plan presented for user review.")
            await pilot.pause()

            self.assertIsNotNone(app.mode["current_plan"])
            rendered = "\n".join(renderable_plain(block) for block in app.query(".plan-body"))
            self.assertIn("Ephemeral Structured Planning", rendered)
            self.assertIn("Summary", rendered)
            self.assertIn("Key Changes", rendered)
            self.assertIn("Test Plan", rendered)
            self.assertIn("Assumptions", rendered)
            buttons = list(app.query(".plan-action"))
            self.assertEqual(len(buttons), 3)
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Implement (i)", "Revise (r)", "Discard (d)"],
            )
            self.assertEqual(len(app.query(".plan-actions")), 1)
            self.assertTrue(all(button.compact for button in buttons))
            self.assertTrue(all(button.region.height == 1 for button in buttons))
            self.assertTrue(all(all(edge[0] == "" for edge in button.styles.border) for button in buttons))

            discard_button = app.query_one("#plan-discard-plan-1", Button)
            discard_button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#plan-discard-plan-1")
            await wait_until(lambda: app.mode["current_plan"] is None)

            self.assertIsNone(app.mode["current_plan"])
            self.assertEqual(len([button for button in app.query(".plan-action") if button.display]), 0)
            rendered = "\n".join(renderable_plain(block) for block in app.query(".plan-body"))
            self.assertIn("Status: discarded", rendered)

    async def test_present_plan_shortcuts_match_visible_button_labels(self) -> None:
        """Presented plan controls should focus and honor visible keyboard controls."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Shortcut Plan",
            "summary": ["Keep shortcut labels honest."],
            "key_changes": ["Handle i, r, and d."],
            "test_plan": ["Press each visible shortcut."],
            "assumptions": ["The plan button row has focus."],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await app.present_plan(interrupt)
            await pilot.pause()

            first_button = app.query_one("#plan-implement-plan-1", Button)
            await wait_until(lambda: first_button.has_focus)

            revise_button = app.query_one("#plan-revise-plan-1", Button)
            discard_button = app.query_one("#plan-discard-plan-1", Button)
            await pilot.press("right")
            await wait_until(lambda: revise_button.has_focus)
            await pilot.press("right")
            await wait_until(lambda: discard_button.has_focus)
            await pilot.press("right")
            await wait_until(lambda: first_button.has_focus)
            await pilot.press("left")
            await wait_until(lambda: discard_button.has_focus)
            await pilot.press("right")
            await wait_until(lambda: first_button.has_focus)

            app.query_one("#prompt").focus()
            await wait_until(lambda: getattr(app.focused, "id", None) == "prompt")
            await pilot.click(".plan-body")
            await wait_until(lambda: first_button.has_focus)

            with patch.object(app, "_handle_plan_action", new_callable=AsyncMock) as handle_action:
                for count, (shortcut, action) in enumerate(
                    (("i", "implement"), ("r", "revise"), ("d", "discard")),
                    start=1,
                ):
                    await pilot.press(shortcut)
                    await wait_until(lambda: handle_action.await_count == count)
                    self.assertEqual(handle_action.await_args_list[-1].args, (action, "plan-1"))

                await pilot.press("enter")
                await wait_until(lambda: handle_action.await_count == 4)
                self.assertEqual(handle_action.await_args_list[-1].args, ("implement", "plan-1"))

            await pilot.press("escape")
            await wait_until(lambda: getattr(app.focused, "id", None) == "prompt")

    async def test_cancel_turn_keeps_unrelated_plan_bubble_active(self) -> None:
        """Cancelling a later turn should not discard an active plan bubble."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Cancellation Plan",
            "summary": ["Keep the active plan intact."],
            "key_changes": ["Cancel an unrelated turn."],
            "test_plan": ["Verify plan actions remain visible."],
            "assumptions": ["The cancellation is unrelated."],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            await app.present_plan(interrupt)
            await pilot.pause()
            self.assertEqual(len([button for button in app.query(".plan-action") if button.display]), 3)

            app.reasoning_delta("unrelated turn thinking")
            app.busy = True
            app.turn_worker = FakeWorker()
            app._cancel_turn()
            await pilot.pause()

            self.assertIsNotNone(app.mode["current_plan"])
            self.assertEqual(len([button for button in app.query(".plan-action") if button.display]), 3)
            rendered = "\n".join(renderable_plain(block) for block in app.query(".plan-body"))
            self.assertIn("Cancellation Plan", rendered)
            self.assertNotIn("Status:", rendered)

    def test_normalize_plan_fills_all_structured_sections(self) -> None:
        """Partial plan payloads should not drop sections in logs or replay."""
        plan = normalize_plan({"title": "Partial", "summary": ["One."]})

        self.assertEqual(plan["summary"], ["One."])
        self.assertEqual(plan["key_changes"], ["List the key implementation changes."])
        self.assertEqual(plan["test_plan"], ["Describe the tests or checks to create."])
        self.assertEqual(plan["assumptions"], ["No additional assumptions identified."])

    async def test_present_plan_revise_cancel_keeps_plan_active(self) -> None:
        """Cancelling a revision prompt should leave the current plan actionable."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Palindrome Plan",
            "summary": ["Create a palindrome helper."],
            "key_changes": ["Add palindrome.py."],
            "test_plan": ["Add unit tests for palindrome inputs."],
            "assumptions": ["Use Python."],
        }

        with patch.object(app, "_prompt_text", return_value=None):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app.present_plan(interrupt)
                await pilot.pause()
                await app._handle_plan_action("revise", "plan-1")
                await pilot.pause()

                self.assertIsNotNone(app.mode["current_plan"])
                self.assertEqual(len([button for button in app.query(".plan-action") if button.display]), 3)
                rendered = "\n".join(renderable_plain(block) for block in app.query(".plan-body"))
                self.assertNotIn("Status: revision requested", rendered)

    async def test_present_plan_revise_runs_planning_turn_with_plan_context(self) -> None:
        """Submitted revision feedback should become a visible planning turn with old-plan context."""
        app = make_app()
        calls: list[dict[str, Any]] = []
        interrupt = {
            "type": "present_plan",
            "title": "Palindrome Plan",
            "summary": ["Create a palindrome helper."],
            "key_changes": ["Add palindrome.py."],
            "test_plan": ["Add unit tests for palindrome inputs."],
            "assumptions": ["Use Python."],
        }

        async def fake_run_user_turn(**kwargs: Any) -> None:
            calls.append(kwargs)

        with (
            patch.object(app, "_prompt_text", return_value="include a testing plan"),
            patch("ui.app.run_user_turn", fake_run_user_turn),
        ):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()

                await app.present_plan(interrupt)
                await pilot.pause()
                await app._handle_plan_action("revise", "plan-1")
                await wait_until(lambda: len(calls) == 1)

                self.assertIsNone(app.mode["current_plan"])
                self.assertTrue(app.mode["planning"])
                self.assertEqual(calls[0]["display_text"], "Revise plan: include a testing plan")
                self.assertTrue(calls[0]["record_user"])
                self.assertIn("Revise this structured plan.", calls[0]["text"])
                self.assertIn("Title: Palindrome Plan", calls[0]["text"])
                self.assertIn("- Add palindrome.py.", calls[0]["text"])
                self.assertIn("Test Plan:\n- Add unit tests for palindrome inputs.", calls[0]["text"])
                self.assertIn("User feedback:\ninclude a testing plan", calls[0]["text"])
                rendered = "\n".join(renderable_plain(block) for block in app.query(".plan-body"))
                self.assertIn("Status: revision requested", rendered)

    async def test_present_plan_revise_button_opens_feedback_prompt(self) -> None:
        """The Revise button should ask for feedback before resolving the plan."""
        app = make_app()
        interrupt = {
            "type": "present_plan",
            "title": "Palindrome Plan",
            "summary": ["Create a palindrome helper."],
            "key_changes": ["Add palindrome.py."],
            "test_plan": ["Add unit tests for palindrome inputs."],
            "assumptions": ["Use Python."],
        }

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            await app.present_plan(interrupt)
            await pilot.pause()
            app.query_one("#plan-revise-plan-1", Button).press()
            await wait_until(lambda: app.query_one(PromptPanel).display)

            self.assertEqual(renderable_plain(app.query_one("#prompt-panel-title", Static)), "Revise Plan")
            self.assertIn("What should MIRA change", renderable_plain(app.query_one("#prompt-panel-message", Static)))
            await wait_until(lambda: len(list(app.query_one(PromptPanel).query("#prompt-cancel"))) > 0)
            app.query_one("#prompt-cancel", Button).press()
            await wait_until(lambda: not app.query_one(PromptPanel).display)
            await wait_until(
                lambda: "kept plan" in "\n".join(renderable_plain(block) for block in app.query_one(ChatLog).children)
            )

    async def test_git_prompt_booleans_use_in_window_choices(self) -> None:
        """Startup Git prompts should keep returning the expected booleans."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            create_task = asyncio.create_task(app.ask_create_git_repo("Initialize Git?"))
            await pilot.pause()
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Yes (y)", "No (n)"],
            )
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            await pilot.press("y")
            self.assertTrue(await asyncio.wait_for(create_task, timeout=2))

            continue_task = asyncio.create_task(app.ask_continue_without_git("Continue without Git?"))
            await pilot.pause()
            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual(
                [button.label.plain for button in buttons],
                ["Continue (c)", "Exit (e)"],
            )
            self.assertTrue(all(button.styles.content_align_horizontal == "center" for button in buttons))
            self.assertTrue(all(button.styles.content_align_vertical == "middle" for button in buttons))
            await pilot.press("e")
            self.assertFalse(await asyncio.wait_for(continue_task, timeout=2))


if __name__ == "__main__":
    unittest.main()
