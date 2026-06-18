"""Tests for durable session transcript helpers."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from session import context
from session.dashboard import apply_context_usage, apply_turn_usage, ensure_dashboard
from session.recorder import SessionRecorder
from session.store import SessionStore
from runtime import runner
from tests.test_runner import COMPACTION_SUMMARY, FakeAgent, FakeStream, Message as StreamMessage, RunTurnRenderer
from ui.repl import run_user_turn


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


class AgentWithFailingState(FakeAgent):
    async def aget_state(self, config: dict[str, Any]) -> Snapshot:
        raise TypeError("'MockValSer' object is not an instance of 'SchemaSerializer'")


class AgentWithFailingTurn:
    async def astream_events(self, payload: Any, config: dict[str, Any], version: str) -> FakeStream:
        raise RuntimeError("main turn failed")

    async def aget_state(self, config: dict[str, Any]) -> Snapshot:
        raise TypeError("'MockValSer' object is not an instance of 'SchemaSerializer'")


class Store:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        self.saved.append(record)


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
                "events": [],
            }
            store.save(custom)
            loaded = store.load("custom-session", resume=False, workspace=Path("workspace"))

        self.assertEqual(explicit["id"], "thread-1")
        self.assertEqual(loaded["id"], "custom-session")
        self.assertFalse(re.match(r"^\d{8}-\d{6}[+-]\d{4}-[0-9a-f]{8}$", loaded["id"]))

    def test_session_store_clear_all_deletes_only_session_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)
            store.save(store.new(session_id="one", workspace=Path("workspace")))
            store.save(store.new(session_id="two", workspace=Path("workspace")))
            note = root / "notes.md"
            note.write_text("keep", encoding="utf-8")

            removed = store.clear_all()

            self.assertEqual(removed, 2)
            self.assertEqual(list(root.glob("*.json")), [])
            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_session_store_clear_compactions_deletes_conversation_history_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mira = Path(directory) / ".mira"
            store = SessionStore(mira / "_sessions")
            archive = mira / "conversation_history" / "nested"
            archive.mkdir(parents=True)
            first = archive / "one.md"
            second = mira / "conversation_history" / "two.md"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")
            other = mira / "tools" / "keep.py"
            other.parent.mkdir()
            other.write_text("keep", encoding="utf-8")

            removed = store.clear_compactions()

            self.assertEqual(removed, 2)
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(other.read_text(encoding="utf-8"), "keep")

    def test_session_store_delete_one_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save(store.new(session_id="one", workspace=Path("workspace")))
            store.save(store.new(session_id="two", workspace=Path("workspace")))

            self.assertTrue(store.delete("one"))
            self.assertFalse(store.delete("missing"))

            self.assertFalse(store.path("one").exists())
            self.assertTrue(store.path("two").exists())

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
                "events",
            ],
        )
        self.assertEqual(record["title"], "Untitled session")
        self.assertEqual(record["dashboard"]["context"]["percent"], 0.0)
        self.assertEqual(record["events"], [])

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

    def test_dashboard_context_uses_total_tokens_without_changing_token_totals(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 8200,
                    "output_tokens": 1424,
                    "total_tokens": 9624,
                    "context_tokens": 9624,
                    "source": "response_metadata.stats",
                }
            },
        )()

        apply_turn_usage(record, result, model_name="lmstudio:qwen3.5-9b", context_limit_tokens=10000)
        normalized = context.normalize_session(record)

        self.assertEqual(normalized["dashboard"]["tokens"], {"in": 8200, "out": 1424})
        self.assertEqual(normalized["dashboard"]["context"]["used_tokens"], 9624)
        self.assertEqual(normalized["dashboard"]["context"]["percent"], 96.2)

    def test_dashboard_context_uses_request_floor_above_low_provider_total(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 1400,
                    "output_tokens": 67,
                    "total_tokens": 1467,
                    "context_tokens": 1467,
                    "context_floor_tokens": 10013,
                    "source": "usage_metadata",
                }
            },
        )()

        apply_turn_usage(record, result, model_name="lmstudio:qwen3.5-27b-mtp", context_limit_tokens=12000)

        self.assertEqual(record["dashboard"]["tokens"], {"in": 1400, "out": 67})
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 10013)
        self.assertEqual(record["dashboard"]["context"]["percent"], 83.4)
        self.assertEqual(record["dashboard"]["context"]["source"], "request_floor.count_tokens")

    def test_dashboard_estimate_does_not_lower_provider_context_usage(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        provider_result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 8400,
                    "output_tokens": 1400,
                    "total_tokens": 9800,
                    "context_tokens": 9800,
                    "source": "usage_metadata",
                }
            },
        )()
        compacted_result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 8200,
                    "output_tokens": 100,
                    "total_tokens": 8300,
                    "context_tokens": 8300,
                    "source": "usage_metadata",
                }
            },
        )()

        apply_turn_usage(record, provider_result, model_name="lmstudio:qwen", context_limit_tokens=10000)
        apply_context_usage(
            record,
            7,
            model_name="lmstudio:qwen",
            context_limit_tokens=10000,
            source="langchain_approx.count_tokens",
        )
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 9800)

        apply_turn_usage(record, compacted_result, model_name="lmstudio:qwen", context_limit_tokens=10000)
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 8300)

    def test_dashboard_limit_source_does_not_claim_context_usage(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))

        ensure_dashboard(
            record,
            model_name="lmstudio:qwen",
            context_limit_tokens=10000,
            context_limit_source="lmstudio.api.v1.loaded_instance",
        )

        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 0)
        self.assertEqual(record["dashboard"]["context"]["limit_tokens"], 10000)
        self.assertEqual(record["dashboard"]["context"]["source"], "unknown")

    def test_title_uses_recent_user_prompts_with_cap(self) -> None:
        record = {"title": "Untitled session", "events": []}
        context.append_event(record, {"type": "user", "mode": "action", "text": "hello"})
        context.append_event(record, {"type": "assistant", "mode": "action", "text": "Hello"})
        context.update_title(record)
        self.assertEqual(record["title"], "hello")

        context.append_event(record, {"type": "user", "mode": "action", "text": "help me debug qwen reasoning_content"})
        context.append_event(record, {"type": "assistant", "mode": "action", "text": "done"})
        context.update_title(record)
        self.assertEqual(record["title"], "help me debug qwen reasoning_content hello")

        context.append_event(
            record,
            {"type": "user", "mode": "action", "text": "now check deepagents compact_conversation history"},
        )
        context.append_event(record, {"type": "assistant", "mode": "action", "text": "done"})
        context.update_title(record)
        self.assertEqual(record["title"], "now check deepagents compact_conversation histor")

    async def test_deepagents_compaction_event_is_copied_once(self) -> None:
        record = {"events": []}
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
        compactions = context.normalize_compactions(record["events"])
        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0]["cutoff_index"], 12)
        self.assertEqual(compactions[0]["file_path"], "/.mira/conversation_history/thread-1.md")
        self.assertEqual(compactions[0]["summary"], "Debugged Qwen helper latency.")

    async def test_compaction_summary_string_is_copied(self) -> None:
        record = {"events": []}
        agent = AgentWithState(
            {
                "_summarization_event": {
                    "cutoff_index": 4,
                    "file_path": "/.mira/conversation_history/thread-1.md",
                    "summary": "Earlier messages were summarized.",
                }
            }
        )

        await context.sync_deepagents_compaction(record, agent, "thread-1")

        compactions = context.normalize_compactions(record["events"])
        self.assertEqual(compactions[0]["summary"], "Earlier messages were summarized.")

    async def test_compaction_sync_scrubs_leaked_reasoning_events(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "reasoning",
                    "created_at": "2026-06-18T05:01:45+00:00",
                    "mode": "action",
                    "text": (
                        "The user wants me to extract context from the conversation history. "
                        "Looking at the messages provided:\n\n"
                        "## SESSION INTENT\nWrite a story.\n\n"
                        "## SUMMARY\nThe task was completed.\n\n"
                        "## ARTIFACTS\nFile created.\n\n"
                        "## NEXT STEPS\nNone."
                    ),
                },
                {
                    "id": 2,
                    "type": "compaction",
                    "created_at": "2026-06-18T05:15:40+00:00",
                    "cutoff_index": 2,
                    "file_path": "/.mira/conversation_history/thread-1.md",
                    "summary": "Write a story.",
                },
                {
                    "id": 3,
                    "type": "info",
                    "created_at": "2026-06-18T05:15:47+00:00",
                    "mode": "action",
                    "text": (
                        "The user wants me to extract the most important context from this "
                        "conversation history. Let me analyze what's happened:\n\n"
                        "Key information to extract:\n"
                        "- Session intent: User wants a short story written to a file\n"
                        "- Summary: Story content was created\n"
                        "- Artifacts: File /mira-short-story.txt\n"
                        "- Next Steps: Verify the file was written successfully\n\n"
                        "Let me structure this properly according to the instructions."
                    ),
                },
            ]
        }
        agent = AgentWithState({})

        changed = await context.sync_deepagents_compaction(record, agent, "thread-1")

        self.assertTrue(changed)
        self.assertEqual([event["type"] for event in record["events"]], ["compaction"])

    def test_recorder_does_not_save_compaction_reasoning_as_info(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.info(
            "The user wants me to extract the most important context from this conversation history. "
            "Key information to extract: Session intent, Summary, Artifacts, Next Steps."
        )

        self.assertEqual(record["events"], [])

    def test_recorder_does_not_duplicate_streamed_assistant_final_text(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.text_delta("hello")
        recorder.finish_main()
        recorder.ensure_assistant("hello")

        messages = context.normalize_messages(record["events"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "hello")

    def test_recorder_updates_streamed_assistant_with_full_final_text(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.text_delta("hel")
        recorder.finish_main()
        recorder.ensure_assistant("hello")

        messages = context.normalize_messages(record["events"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "hello")

    def test_recorder_preserves_subagent_request_on_terminal_events(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.subagent_started("general-purpose [one]", "summarize README")
        recorder.subagent_finished("general-purpose [one]", "done")
        recorder.subagent_started("general-purpose [two]", "find tests")
        recorder.subagent_cancelled("general-purpose [two]", "cancelled")

        subagents = [event for event in context.normalize_events(record["events"]) if event["type"] == "subagent"]
        self.assertEqual(subagents[1]["status"], "DONE")
        self.assertEqual(subagents[1]["task_input"], "summarize README")
        self.assertEqual(subagents[2]["status"], "RUNNING")
        self.assertEqual(subagents[3]["status"], "CANCELLED")
        self.assertEqual(subagents[3]["task_input"], "find tests")

    def test_recorder_updates_blank_running_subagent_request(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.subagent_started("general-purpose [one]", "")
        recorder.subagent_request_updated("general-purpose [one]", "write scary story")
        recorder.subagent_finished("general-purpose [one]", "done")

        subagents = [event for event in context.normalize_events(record["events"]) if event["type"] == "subagent"]
        self.assertEqual(subagents[0]["status"], "RUNNING")
        self.assertEqual(subagents[0]["task_input"], "write scary story")
        self.assertEqual(subagents[1]["status"], "DONE")
        self.assertEqual(subagents[1]["task_input"], "write scary story")

    def test_recorder_deduplicates_delegation_events(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.delegation_started(
            [
                {"id": "task-1", "name": "task", "args": {"description": "one"}},
                {"id": "task-2", "name": "task", "args": {"description": "two"}},
            ]
        )
        recorder.delegation_started([{"id": "task-1", "name": "task", "args": {"description": "one"}}])
        recorder.delegation_started([{"name": "task", "args": {"description": "three", "subagent_type": "general"}}])
        recorder.delegation_started([{"name": "task", "args": {"description": "three", "subagent_type": "general"}}])

        delegations = [event for event in context.normalize_events(record["events"]) if event["type"] == "delegation"]
        self.assertEqual(len(delegations), 2)
        self.assertEqual(len(delegations[0]["calls"]), 2)
        self.assertEqual(delegations[1]["calls"][0]["args"]["description"], "three")

    async def test_compaction_summary_final_text_is_not_persisted_as_assistant(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")
        result = await runner.run_turn(
            FakeAgent(
                [
                    FakeStream(
                        output={
                            "messages": [
                                StreamMessage(
                                    text=COMPACTION_SUMMARY,
                                    additional_kwargs={"lc_source": "summarization"},
                                )
                            ]
                        }
                    )
                ]
            ),
            "hello",
            RunTurnRenderer(),
            "thread-1",
        )

        recorder.ensure_assistant(result.final_text)

        self.assertEqual(context.normalize_messages(record["events"]), [])

    async def test_unmarked_compaction_summary_final_text_is_not_persisted_as_assistant(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")
        result = await runner.run_turn(
            FakeAgent([FakeStream(output={"messages": [StreamMessage(text=COMPACTION_SUMMARY)]})]),
            "hello",
            RunTurnRenderer(),
            "thread-1",
        )

        recorder.ensure_assistant(result.final_text)

        self.assertEqual(context.normalize_messages(record["events"]), [])

    async def test_ai_message_tool_call_repr_is_not_persisted_as_assistant(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")
        message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Need to write a file."},
                {"type": "text", "text": "\n\n"},
                {
                    "type": "tool_call",
                    "id": "call-write",
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                },
            ],
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                    "id": "call-write",
                }
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "unexecuted tool call"):
            result = await runner.run_turn(
                FakeAgent([FakeStream(output={"messages": [message]})]),
                "write",
                RunTurnRenderer(),
                "thread-1",
            )
            recorder.ensure_assistant(result.final_text)

        self.assertEqual(context.normalize_messages(record["events"]), [])

    async def test_post_success_state_sync_error_does_not_append_system_error(self) -> None:
        record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        store = Store()
        agent = AgentWithFailingState([FakeStream(output={"messages": [StreamMessage(text="done")]})])

        result = await run_user_turn(
            agent=agent,
            plan_agent=agent,
            renderer=RunTurnRenderer(),
            store=store,
            session=record,
            mode={"planning": False},
            text="hello",
        )

        self.assertEqual(result.final_text, "done")
        self.assertEqual(record["turns"], 1)
        self.assertEqual([event["type"] for event in context.normalize_events(record["events"])], ["user", "assistant"])

    async def test_main_turn_failure_still_records_system_error(self) -> None:
        record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        store = Store()
        agent = AgentWithFailingTurn()

        with self.assertRaisesRegex(RuntimeError, "main turn failed"):
            await run_user_turn(
                agent=agent,
                plan_agent=agent,
                renderer=RunTurnRenderer(),
                store=store,
                session=record,
                mode={"planning": False},
                text="hello",
            )

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["user", "system_error"])
        self.assertIn("main turn failed", events[-1]["text"])

    def test_resume_context_injects_once(self) -> None:
        record = {
            "resume_context_pending": True,
            "events": [
                {
                    "id": 1,
                    "type": "compaction",
                    "cutoff_index": 8,
                    "file_path": "/.mira/conversation_history/thread-1.md",
                    "summary": "Earlier work debugged session latency.",
                    "created_at": "now",
                },
                {
                    "id": 2,
                    "type": "user",
                    "mode": "action",
                    "created_at": "now",
                    "text": "recent request",
                },
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
