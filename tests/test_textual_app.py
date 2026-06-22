"""Tests for the Textual interactive UI."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pyfiglet import Figlet
from rich.console import Console
from textual.widgets import Button, Input, Static, TextArea

from agent.context_overflow import context_overflow_error, set_context_overflow_notice
from config.metadata import ModelMetadata
from config.settings import load_settings, tool_always_allow, tool_enabled
from ui.interrupts import ASK_USER_OPEN_OPTION, action_preview
from ui.app import DESTRUCTIVE_CONFIRM_CHOICES, MiraApp, append_prompt_history, read_prompt_history
from ui.renderer import Renderer
from ui.splash import HINTS, VERSION, blocky_wordmark, splash_text
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory, SettingsPanel
from ui.widgets.session_history import session_label


class FakeStore:
    """Store double used by app smoke tests."""

    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []
        self.clear_all_calls = 0
        self.clear_compactions_calls = 0

    def save(self, record: dict[str, Any]) -> None:
        """Ignore session saves."""
        self.saves.append(record)
        return None

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
            startup = app.query_one(ChatLog).children[0]
            startup_text = renderable_plain(startup)
            self.assertIn(VERSION, startup_text)
            self.assertIn(blocky_wordmark().splitlines()[-1].rstrip(), startup_text)
            self.assertIn(HINTS, startup_text)

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

        self.assertEqual(output.getvalue(), "\nthinking:\nThe user wants:\n1. Check envs\n2. Tell joke\n")

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

    async def test_ctrl_c_action_cancels_running_turn(self) -> None:
        """The VS Code-friendly interrupt binding should confirm before cancelling."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            worker = FakeWorker()
            app.busy = True
            app.turn_worker = worker

            app.action_interrupt_or_quit()
            await pilot.pause()
            self.assertFalse(worker.cancelled)
            self.assertEqual(renderable_plain(app.query_one("#prompt-panel-title", Static)), "Cancel Turn?")

            await pilot.press("n")
            await pilot.pause()
            self.assertFalse(worker.cancelled)

            app.action_interrupt_or_quit()
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

            self.assertTrue(worker.cancelled)
            self.assertEqual(app.status_state, "cancelling")

    async def test_ctrl_c_action_confirms_idle_exit(self) -> None:
        """Ctrl+C should not quit an idle app without confirmation."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as exit_app:
                app.action_interrupt_or_quit()
                await pilot.pause()
                self.assertEqual(renderable_plain(app.query_one("#prompt-panel-title", Static)), "Exit MIRA?")

                await pilot.press("n")
                await pilot.pause()
                exit_app.assert_not_called()

                app.action_interrupt_or_quit()
                await pilot.pause()
                await pilot.press("y")
                await pilot.pause()
                exit_app.assert_called_once()

    async def test_ctrl_c_action_cancels_running_turn_during_prompt(self) -> None:
        """Ctrl+C should still cancel when another in-window prompt is active."""
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
                app.action_interrupt_or_quit()
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
            self.assertIn("/subagents", output)

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
                buttons = list(panel.query(Button))
                self.assertTrue(panel.display)
                self.assertEqual(len(buttons), 3)
                self.assertTrue(buttons[0].has_focus)

                await pilot.press("right")
                await pilot.pause()
                self.assertTrue(buttons[1].has_focus)
                self.assertFalse(app.query_one(PromptBox).has_focus)

                await pilot.press("right")
                await pilot.pause()
                self.assertTrue(buttons[2].has_focus)

                await pilot.press("enter")
                decisions = await asyncio.wait_for(task, timeout=2)
                await pilot.pause()

                self.assertEqual(decisions, [{"type": "reject"}])
                self.assertFalse(panel.display)

    async def test_approval_prompt_supports_respond_decision(self) -> None:
        """Respond decisions should return a synthetic successful tool result."""
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
                ["Approve (a)", "Edit (e)", "Reject (r)", "Respond (s)"],
            )
            self.assertTrue(all(button.variant == "default" for button in buttons))

            await pilot.press("s")
            await pilot.pause()
            answer = app.query_one("#prompt-panel-input", Input)
            self.assertTrue(answer.has_focus)
            answer.value = "Stop retrying; the file change is not needed."
            await pilot.press("enter")

            self.assertEqual(
                await asyncio.wait_for(task, timeout=2),
                [{"type": "respond", "message": "Stop retrying; the file change is not needed."}],
            )

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

            buttons = list(app.query_one(PromptPanel).query(Button))
            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
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

            buttons = list(app.query_one(PromptPanel).query(Button))
            message = renderable_plain(app.query_one("#prompt-panel-message", Static))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])
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

            buttons = list(app.query_one(PromptPanel).query(Button))
            self.assertEqual([button.label.plain for button in buttons], ["OK (o)", "Cancel (c)"])

            await pilot.press("c")
            await wait_until(lambda: not app.query_one(PromptPanel).display)

            self.assertEqual(store.clear_all_calls, 0)
            self.assertEqual(store.clear_compactions_calls, 0)

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
                buttons = {button.id: button for button in panel.query(Button)}

                self.assertNotIn("Config", rendered)
                self.assertIn("Settings", rendered)
                self.assertIn("enabled", rendered)
                self.assertIn("always allow", rendered)
                self.assertIn("System", rendered)
                self.assertIn("Inbuilt Tools", rendered)
                self.assertIn("Custom Tools", rendered)
                self.assertIn("Git Protection", rendered)
                self.assertIn("write_file", rendered)
                self.assertIn("edit_file", rendered)
                self.assertIn("eval", rendered)
                self.assertIn("task", rendered)
                self.assertIn("execute", rendered)
                self.assertIn("settings-toggle-git-git_protection", buttons)
                self.assertIn("settings-toggle-enabled-edit_file", buttons)
                self.assertIn("settings-toggle-always_allow-edit_file", buttons)
                self.assertIn("settings-toggle-enabled-write_file", buttons)
                self.assertIn("settings-toggle-enabled-execute", buttons)
                self.assertIn("settings-toggle-always_allow-execute", buttons)
                self.assertIn("settings-close", buttons)
                self.assertEqual(str(buttons["settings-toggle-git-git_protection"].label), "yes")
                self.assertEqual(str(buttons["settings-toggle-enabled-edit_file"].label), "yes")
                self.assertTrue(buttons["settings-toggle-enabled-edit_file"].disabled)
                self.assertEqual(str(buttons["settings-toggle-enabled-execute"].label), "no")
                self.assertFalse(buttons["settings-toggle-enabled-execute"].disabled)
                self.assertEqual(str(buttons["settings-toggle-always_allow-execute"].label), "no")
                self.assertTrue(buttons["settings-toggle-always_allow-execute"].disabled)
                self.assertEqual(str(buttons["settings-toggle-always_allow-edit_file"].label), "no")
                self.assertEqual(str(buttons["settings-toggle-always_allow-write_file"].label), "no")

                panel.query_one("#settings-close", Button).press()
                await wait_until(lambda: len(app.query(SettingsPanel)) == 0)
                self.assertTrue(app.query_one(PromptBox).has_focus)

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
            self.assertTrue(editor.has_focus)
            editor.text = '{"file_path": "test.txt", "content": "bye"}'
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
            await pilot.press("2")
            await pilot.pause()
            answer = app.query_one("#prompt-panel-input", Input)
            self.assertTrue(answer.has_focus)
            answer.value = "Try the safer patch"
            await pilot.press("enter")
            self.assertEqual(await asyncio.wait_for(open_task, timeout=2), "Try the safer patch")

    async def test_git_prompt_booleans_use_in_window_choices(self) -> None:
        """Startup Git prompts should keep returning the expected booleans."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            create_task = asyncio.create_task(app.ask_create_git_repo("Initialize Git?"))
            await pilot.pause()
            await pilot.press("y")
            self.assertTrue(await asyncio.wait_for(create_task, timeout=2))

            continue_task = asyncio.create_task(app.ask_continue_without_git("Continue without Git?"))
            await pilot.pause()
            await pilot.press("e")
            self.assertFalse(await asyncio.wait_for(continue_task, timeout=2))


if __name__ == "__main__":
    unittest.main()
