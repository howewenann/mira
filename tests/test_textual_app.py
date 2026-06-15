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
from ui.interrupts import ASK_USER_OPEN_OPTION, action_preview
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
            second = renderable_plain(block)

            app.compaction_finished()
            await pilot.pause()
            done = renderable_plain(block)

        self.assertIn("compacting context...", first)
        self.assertIn("compacting context...", second)
        self.assertEqual(first, second)
        self.assertIn("context compacted", done)

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
