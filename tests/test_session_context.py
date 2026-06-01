"""Tests for durable session context helpers."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from session import context
from session.store import SessionStore


class FakeModel:
    """Simple async model double for title and summary generation."""

    def __init__(self, outputs: list[str]) -> None:
        """Store scripted model outputs."""
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> str:
        """Record the prompt and return the next scripted output."""
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class SessionContextTests(unittest.IsolatedAsyncioTestCase):
    """Tests for one-file session resume data."""

    def test_new_session_id_is_timestamped(self) -> None:
        """Default session ids should sort alphabetically by creation time."""
        record = SessionStore(Path(".")).new(session_id=None, workspace=Path("workspace"))

        self.assertRegex(record["id"], r"^\d{8}-\d{6}[+-]\d{4}-[0-9a-f]{8}$")

    def test_explicit_and_legacy_session_ids_still_load_exactly(self) -> None:
        """Readable ids should not break old UUID session files or explicit ids."""
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))

            explicit = store.load("thread-1", resume=False, workspace=Path("workspace"))
            legacy = {
                "id": "0dc61ead7e38",
                "title": "Legacy Session",
                "workspace": "workspace",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "turns": 0,
                "context_policy": context.context_policy(),
                "summary": None,
                "messages": [],
            }
            store.save(legacy)
            loaded = store.load("0dc61ead7e38", resume=False, workspace=Path("workspace"))

        self.assertEqual(explicit["id"], "thread-1")
        self.assertEqual(loaded["id"], "0dc61ead7e38")
        self.assertFalse(re.match(r"^\d{8}-\d{6}[+-]\d{4}-[0-9a-f]{8}$", loaded["id"]))

    def test_new_session_has_v1_context_fields(self) -> None:
        """New records should include readable V1 session fields."""
        record = SessionStore(Path(".")).new(
            session_id="thread-1",
            workspace=Path("workspace"),
            policy={"max_chars": 40, "recent_messages": 2, "summary_max_chars": 500},
        )

        self.assertEqual(list(record.keys()), [
            "id",
            "title",
            "workspace",
            "created_at",
            "updated_at",
            "turns",
            "context_policy",
            "summary",
            "messages",
        ])
        self.assertEqual(record["title"], "Untitled session")
        self.assertEqual(record["messages"], [])
        self.assertIsNone(record["summary"])

    async def test_title_generation_refreshes_after_early_turns(self) -> None:
        """The title should mature after the first real follow-up turn."""
        record = {
            "id": "thread-1",
            "title": "Untitled session",
            "messages": [],
            "context_policy": context.context_policy(),
            "turns": 1,
        }
        context.append_turn(record, "add resume", "done", "action")
        model = FakeModel(["MIRA Session Kickoff", "Durable Resume Work"])

        await context.update_title(record, model)
        self.assertEqual(record["title"], "MIRA Session Kickoff")

        record["turns"] = 2
        context.append_turn(record, "make resume durable", "updated session code", "action")
        await context.update_title(record, model)

        self.assertEqual(record["title"], "Durable Resume Work")
        self.assertEqual(len(model.prompts), 2)

    async def test_title_generation_skips_between_periodic_updates(self) -> None:
        """The LLM should not be called on every completed turn."""
        record = {
            "id": "thread-1",
            "title": "Durable Resume Work",
            "messages": [],
            "context_policy": context.context_policy(),
            "turns": 3,
        }
        context.append_turn(record, "next task", "done", "action")
        model = FakeModel(["Unexpected Title"])

        await context.update_title(record, model)

        self.assertEqual(record["title"], "Durable Resume Work")
        self.assertEqual(len(model.prompts), 0)

    async def test_title_generation_runs_on_periodic_turns(self) -> None:
        """Longer sessions should occasionally refresh their title."""
        record = {
            "id": "thread-1",
            "title": "Durable Resume Work",
            "messages": [],
            "context_policy": context.context_policy(),
            "turns": 5,
        }
        context.append_turn(record, "add local session filenames", "done", "action")
        model = FakeModel(["Local Session Filenames"])

        await context.update_title(record, model)

        self.assertEqual(record["title"], "Local Session Filenames")
        self.assertEqual(len(model.prompts), 1)

    async def test_compaction_keeps_recent_messages_and_structured_summary(self) -> None:
        """Long sessions should compact older messages into continuation state."""
        record = {
            "id": "thread-1",
            "title": "Title",
            "messages": [],
            "context_policy": {"max_chars": 10, "recent_messages": 2, "summary_max_chars": 1000},
        }
        context.append_turn(record, "first request", "first response", "action")
        context.append_turn(record, "second request", "second response", "action")
        model = FakeModel(
            [
                """
                {
                  "objective": "Resume sessions",
                  "current_status": "Compacting older context",
                  "important_decisions": ["Use one JSON file"],
                  "user_preferences": ["Readable session files"],
                  "relevant_files": ["session/context.py"],
                  "next_steps": ["Inject context on resume"]
                }
                """
            ]
        )

        await context.compact_if_needed(record, model)

        self.assertEqual([message["content"] for message in record["messages"]], ["second request", "second response"])
        self.assertEqual(record["summary"]["through_message"], 2)
        self.assertEqual(record["summary"]["state"]["objective"], "Resume sessions")
        self.assertEqual(record["summary"]["state"]["next_steps"], ["Inject context on resume"])

    def test_will_compact_detects_long_sessions_with_older_messages(self) -> None:
        """The compaction predicate should be true only when older messages exist."""
        record = {
            "messages": [],
            "context_policy": {"max_chars": 10, "recent_messages": 2, "summary_max_chars": 1000},
        }
        context.append_turn(record, "short", "ok", "action")
        self.assertFalse(context.will_compact(record))

        context.append_turn(record, "a much longer request", "a much longer response", "action")
        self.assertTrue(context.will_compact(record))

    def test_resume_context_injects_once(self) -> None:
        """Resumed session context should be added to only one request."""
        record = {
            "resume_context_pending": True,
            "summary": {
                "version": 1,
                "kind": "llm_compaction",
                "through_message": 2,
                "updated_at": "now",
                "state": {
                    "objective": "Resume sessions",
                    "current_status": "Testing",
                    "important_decisions": [],
                    "user_preferences": [],
                    "relevant_files": [],
                    "next_steps": ["Continue"],
                },
            },
            "messages": [
                {
                    "id": 3,
                    "role": "user",
                    "mode": "action",
                    "created_at": "now",
                    "content": "recent request",
                }
            ],
        }

        first = context.with_resume_context(record, "next request")
        second = context.with_resume_context(record, "another request")

        self.assertIn("Previous MIRA session context:", first)
        self.assertIn("Resume sessions", first)
        self.assertIn("recent request", first)
        self.assertEqual(second, "another request")


if __name__ == "__main__":
    unittest.main()
