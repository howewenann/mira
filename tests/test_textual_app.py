"""Tests for the Textual interactive UI."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pyfiglet import Figlet
from textual.widgets import Button, Input, Static, TextArea

from ui.interrupts import ASK_USER_OPEN_OPTION
from ui.app import MiraApp, append_prompt_history, read_prompt_history
from ui.splash import HINTS, VERSION, blocky_wordmark, splash_text
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory
from ui.widgets.session_history import session_label


class FakeStore:
    """Store double used by app smoke tests."""

    def save(self, record: dict[str, Any]) -> None:
        """Ignore session saves."""
        return None


def renderable_plain(widget: Any) -> str:
    """Return plain text from a Textual Static-like widget."""
    renderable = getattr(widget, "renderable", None) or getattr(widget, "_renderable", None) or getattr(widget, "content", "")
    return str(getattr(renderable, "plain", renderable))


def make_app(workspace: Path | None = None) -> MiraApp:
    """Return a bootstrapped app with fake agents and session state."""
    workspace = workspace or Path(".")
    return MiraApp(
        workspace=workspace,
        prebuilt={
            "agent": "agent",
            "plan_agent": "plan-agent",
            "store": FakeStore(),
            "session": {
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
            },
            "model_name": "test-model",
            "session_model": None,
            "context_limit_tokens": 8192,
            "context_limit_source": "test",
        },
        tool_output_chars=80,
    )


class TextualAppTests(unittest.IsolatedAsyncioTestCase):
    """Smoke tests for the Textual app shell."""

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

    def test_session_label_shows_title_and_timestamp_without_turns(self) -> None:
        """Sidebar rows should reserve space for timestamp instead of turns."""
        label = session_label(
            {
                "title": "Implementation Strategy Selection Work",
                "updated_at": "2026-06-03T09:15:00",
                "turns": 7,
            }
        )

        self.assertEqual(label.count("\n"), 1)
        self.assertIn("Jun 03 09:15", label)
        self.assertNotIn("turn", label.lower())
        self.assertLessEqual(len(label.splitlines()[0]), 24)

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

                await app.submit_prompt(Input.Submitted(prompt, "hello"))
                await pilot.pause()

                self.assertEqual(calls, ["hello"])
                self.assertFalse(prompt.disabled)
                self.assertTrue(prompt.has_focus)

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
            append_prompt_history(history_path, "second prompt")

            self.assertEqual(read_prompt_history(history_path), ["first prompt", "second prompt"])

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
            prompt_y = app.query_one(PromptBox).region.y
            panel_bottom = panel.region.y + panel.region.height
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
