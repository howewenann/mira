"""Tests for durable session transcript helpers."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from session import context
from session.dashboard import apply_turn_usage
from session.store import SessionStore


class Snapshot:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class AgentWithState:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.configs: list[dict[str, Any]] = []

    async def aget_state(self, config: dict[str, Any]) -> Snapshot:
        self.configs.append(config)
        return Snapshot(self.values)


class Message:
    def __init__(self, content: str) -> None:
        self.content = content


class SessionContextTests(unittest.IsolatedAsyncioTestCase):
    def test_new_session_id_is_timestamped(self) -> None:
        record = SessionStore(Path(".")).new(session_id=None, workspace=Path("workspace"))

        self.assertRegex(record["id"], r"^\d{8}-\d{6}[+-]\d{4}-[0-9a-f]{8}$")

    def test_explicit_session_ids_load_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))

            explicit = store.load("thread-1", resume=False, workspace=Path("workspace"))
            custom = {
                "id": "custom-session",
                "title": "Custom Session",
                "workspace": "workspace",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "turns": 0,
                "dashboard": {},
                "compactions": [],
                "messages": [],
            }
            store.save(custom)
            loaded = store.load("custom-session", resume=False, workspace=Path("workspace"))

        self.assertEqual(explicit["id"], "thread-1")
        self.assertEqual(loaded["id"], "custom-session")
        self.assertFalse(re.match(r"^\d{8}-\d{6}[+-]\d{4}-[0-9a-f]{8}$", loaded["id"]))

    def test_new_session_shape_is_readable(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))

        self.assertEqual(
            list(record.keys()),
            [
                "id",
                "title",
                "workspace",
                "created_at",
                "updated_at",
                "turns",
                "dashboard",
                "compactions",
                "messages",
            ],
        )
        self.assertEqual(record["title"], "Untitled session")
        self.assertEqual(record["dashboard"]["context"]["percent"], 0.0)
        self.assertEqual(record["compactions"], [])
        self.assertEqual(record["messages"], [])

    def test_dashboard_usage_is_persisted_in_session_shape(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 5512,
                    "output_tokens": 91,
                    "context_tokens": 5512,
                    "source": "usage_metadata",
                }
            },
        )()

        apply_turn_usage(record, result, model_name="lmstudio:gemma", context_limit_tokens=8192)
        normalized = context.normalize_session(record)

        self.assertEqual(normalized["dashboard"]["model"], "lmstudio:gemma")
        self.assertEqual(normalized["dashboard"]["tokens"], {"in": 5512, "out": 91})
        self.assertEqual(normalized["dashboard"]["context"]["percent"], 67.3)

    def test_title_uses_recent_topic(self) -> None:
        record = {"title": "Untitled session", "messages": []}
        context.append_turn(record, "hello", "Hello", "action")
        context.update_title(record)
        self.assertEqual(record["title"], "Untitled session")

        context.append_turn(record, "help me debug qwen reasoning_content", "done", "action")
        context.update_title(record)
        self.assertEqual(record["title"], "Debug Qwen reasoning_content")

        context.append_turn(record, "now check deepagents compact_conversation history", "done", "action")
        context.update_title(record)
        title = record["title"].lower()
        self.assertIn("deepagents", title)
        self.assertIn("compact_conversation", title)

    async def test_deepagents_compaction_event_is_copied_once(self) -> None:
        record = {"compactions": []}
        agent = AgentWithState(
            {
                "_summarization_event": {
                    "cutoff_index": 12,
                    "file_path": "/.mira/conversation_history/thread-1.md",
                    "summary_message": Message(
                        "A condensed summary follows:\n\n<summary>\nDebugged Qwen helper latency.\n</summary>"
                    ),
                }
            }
        )

        await context.sync_deepagents_compaction(record, agent, "thread-1")
        await context.sync_deepagents_compaction(record, agent, "thread-1")

        self.assertEqual(agent.configs[0], {"configurable": {"thread_id": "thread-1"}})
        self.assertEqual(len(record["compactions"]), 1)
        self.assertEqual(record["compactions"][0]["cutoff_index"], 12)
        self.assertEqual(record["compactions"][0]["file_path"], "/.mira/conversation_history/thread-1.md")
        self.assertEqual(record["compactions"][0]["summary"], "Debugged Qwen helper latency.")

    def test_resume_context_injects_once(self) -> None:
        record = {
            "resume_context_pending": True,
            "compactions": [
                {
                    "cutoff_index": 8,
                    "file_path": "/.mira/conversation_history/thread-1.md",
                    "summary": "Earlier work debugged session latency.",
                    "created_at": "now",
                }
            ],
            "messages": [
                {
                    "id": 9,
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
        self.assertIn("Earlier work debugged session latency.", first)
        self.assertIn("/.mira/conversation_history/thread-1.md", first)
        self.assertIn("recent request", first)
        self.assertEqual(second, "another request")


if __name__ == "__main__":
    unittest.main()
