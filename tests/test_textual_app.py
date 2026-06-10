"""Tests for the Textual interactive UI."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pyfiglet import Figlet
from rich.console import Console
from textual.widgets import Button, Input, Static, TextArea

from config.metadata import ModelMetadata
from ui.interrupts import ASK_USER_OPEN_OPTION, action_text
from ui.app import MiraApp, append_prompt_history, read_prompt_history
from ui.splash import HINTS, VERSION, blocky_wordmark, splash_text
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory
from ui.widgets.session_history import session_label


class FakeStore:
    """Store double used by app smoke tests."""

    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        """Ignore session saves."""
        self.saves.append(record)
        return None


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

                await app.submit_prompt(PromptBox.Submitted(prompt, "hello\nsecond line"))
                await pilot.pause()

                self.assertEqual(calls, ["hello\nsecond line"])
                self.assertFalse(prompt.disabled)
                self.assertTrue(prompt.has_focus)

    async def test_compaction_status_animates_in_chat_log(self) -> None:
        """Context compaction status should show a live spinner while running."""
        app = make_app()

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            app.compaction_started()
            await pilot.pause()
            block = app.query_one(ChatLog).children[-1]
            first = renderable_plain(block)

            app.query_one(ChatLog).tick_compaction()
            await pilot.pause()
            second = renderable_plain(block)

            app.compaction_finished()
            await pilot.pause()
            done = renderable_plain(block)

        self.assertIn("compacting context...", first)
        self.assertIn("compacting context...", second)
        self.assertNotEqual(first, second)
        self.assertIn("context compacted", done)

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
                    "type": "user",
                    "mode": "planning",
                    "created_at": "2026-01-01T00:02:00+00:00",
                    "text": "make a plan",
                },
                {
                    "id": 3,
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
            self.assertIn("you (plan)", titles)
            self.assertIn("mira", titles)
            self.assertIn("Older turns compacted", rendered)
            self.assertIn("/.mira/conversation_history/past-1.md", rendered)
            self.assertIn("make a plan", rendered)
            self.assertIn("plan saved", rendered)

    async def test_unchanged_context_metadata_does_not_rebuild_agents(self) -> None:
        """A matching refreshed context window should avoid rebuilding both agents."""
        app = make_app()

        async def infer_metadata(config: dict[str, Any]) -> ModelMetadata:
            return ModelMetadata(8192, "lmstudio.api.v1.loaded_instance")

        with (
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
        app = make_app(store=store)

        async def infer_metadata(config: dict[str, Any]) -> ModelMetadata:
            return ModelMetadata(10000, "lmstudio.api.v1.loaded_instance")

        with (
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
        self.assertNotIn("llm_context_tokens", app.config)
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
            self.assertEqual([button.label.plain for button in buttons], ["a approve", "e edit", "r reject", "s respond"])

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

    def test_action_text_previews_large_json_args(self) -> None:
        """Approval text should show useful compact JSON, not full raw payloads."""
        text = action_text(
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

        self.assertIn("edit_file", text)
        self.assertIn("target: /.mira/memories/AGENTS.md", text)
        self.assertIn('"old_string"', text)
        self.assertIn('"new_string"', text)
        self.assertIn('"replace_all": false', text)
        self.assertIn("truncated", text)
        self.assertIn("Full args available with e edit.", text)
        self.assertLess(len(text), 900)

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
            self.assertEqual(body.styles.overflow_y, "hidden")
            self.assertIn("target: ui/dialogs.py", message)
            self.assertIn("truncated", message)
            self.assertLess(len(message), 900)
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
