"""Tests for the Textual interactive UI."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from textual.widgets import Input

from ui.app import MiraApp
from ui.widgets import ChatLog, PromptBox, SessionHistory


class FakeStore:
    """Store double used by app smoke tests."""

    def save(self, record: dict[str, Any]) -> None:
        """Ignore session saves."""
        return None


def make_app() -> MiraApp:
    """Return a bootstrapped app with fake agents and session state."""
    return MiraApp(
        workspace=Path("."),
        prebuilt={
            "agent": "agent",
            "plan_agent": "plan-agent",
            "store": FakeStore(),
            "session": {
                "id": "thread-1",
                "workspace": ".",
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


if __name__ == "__main__":
    unittest.main()
