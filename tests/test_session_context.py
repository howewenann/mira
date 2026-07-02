"""Tests for durable session transcript helpers."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.compaction import PostTurnCompactionResult, compact_after_turn, mark_summarization_engine
from langchain_core.messages import AIMessage, HumanMessage

from session import context
from session.dashboard import apply_context_usage, apply_turn_usage, ensure_dashboard
from session.recorder import RecordingRenderer as SessionRecordingRenderer
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


class AgentWithMutableState(AgentWithState):
    def __init__(self, values: dict[str, Any], summarization: Any) -> None:
        super().__init__(values)
        self.mira_summarization = summarization
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def aupdate_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        self.updates.append((config, values))
        self.values.update(values)


class FakeSummarization:
    def __init__(self) -> None:
        self._backend = object()
        self.offloaded: list[Any] = []
        self.thread_ids: list[str] = []

    def _get_thread_id(self) -> str:
        return "unset"

    def _apply_event_to_messages(self, messages: list[Any], event: Any) -> list[Any]:
        if event is None:
            return messages
        return [event["summary_message"], *messages[int(event["cutoff_index"]) :]]

    def _determine_cutoff_index(self, messages: list[Any]) -> int:
        return 1

    def _partition_messages(self, messages: list[Any], cutoff: int) -> tuple[list[Any], list[Any]]:
        return messages[:cutoff], messages[cutoff:]

    async def _aoffload_to_backend(self, backend: Any, messages: list[Any]) -> str:
        self.thread_ids.append(self._get_thread_id())
        self.offloaded.append(messages)
        return "/.mira/conversation_history/thread-1.md"

    async def _acreate_summary(self, messages: list[Any]) -> str:
        return "Older context was summarized."

    def _build_new_messages_with_path(self, summary: str, file_path: str) -> list[Any]:
        return [
            HumanMessage(
                content=f"Summary: {summary}\nArchive: {file_path}",
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _compute_state_cutoff(self, event: Any, cutoff: int) -> int:
        return cutoff


class AgentWithFailingState(FakeAgent):
    async def aget_state(self, config: dict[str, Any]) -> Snapshot:
        raise TypeError("'MockValSer' object is not an instance of 'SchemaSerializer'")


class AgentWithFailingTurn:
    async def astream_events(self, payload: Any, config: dict[str, Any], version: str, **kwargs: Any) -> FakeStream:
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
        self.assertEqual(normalized["dashboard"]["context"]["used_tokens"], 0)
        self.assertEqual(normalized["dashboard"]["context"]["percent"], 0.0)

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

    def test_dashboard_context_uses_provider_total_when_reported(self) -> None:
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
                    "source": "usage_metadata",
                }
            },
        )()

        apply_turn_usage(record, result, model_name="lmstudio:qwen3.5-27b-mtp", context_limit_tokens=12000)

        self.assertEqual(record["dashboard"]["tokens"], {"in": 1400, "out": 67})
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 0)
        self.assertEqual(record["dashboard"]["context"]["percent"], 0.0)
        self.assertEqual(record["dashboard"]["context"]["source"], "unknown")

    def test_dashboard_context_uses_provider_pair_above_visible_estimate(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        result = type(
            "Result",
            (),
            {
                "usage": {
                    "input_tokens": 9467,
                    "output_tokens": 123,
                    "total_tokens": 9590,
                    "context_tokens": 454,
                    "context_source": "langchain_approx.count_tokens",
                    "source": "usage_metadata",
                }
            },
        )()

        apply_turn_usage(record, result, model_name="lmstudio:qwen3.5-27b-mtp", context_limit_tokens=10000)

        self.assertEqual(record["dashboard"]["tokens"], {"in": 9467, "out": 123})
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 0)
        self.assertEqual(record["dashboard"]["context"]["source"], "unknown")

    def test_dashboard_context_comes_from_deepagents_count(self) -> None:
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

        apply_turn_usage(record, provider_result, model_name="lmstudio:qwen", context_limit_tokens=10000)
        self.assertEqual(record["dashboard"]["tokens"], {"in": 8400, "out": 1400})
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 0)

        apply_context_usage(
            record,
            9800,
            model_name="lmstudio:qwen",
            context_limit_tokens=10000,
            source="deepagents.summarization._count_tokens",
        )
        self.assertEqual(record["dashboard"]["context"]["used_tokens"], 9800)
        self.assertEqual(record["dashboard"]["context"]["source"], "deepagents.summarization._count_tokens")

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

    async def test_post_turn_compaction_updates_summary_event_and_sanitizes_archive_messages(self) -> None:
        summarization = FakeSummarization()
        mark_summarization_engine(summarization)
        agent = AgentWithMutableState(
            {
                "messages": [
                    AIMessage(
                        content=[
                            {"type": "reasoning", "reasoning": "private chain of thought"},
                            {"type": "text", "text": "Visible answer."},
                        ],
                        additional_kwargs={"reasoning_content": "private chain of thought"},
                    ),
                    HumanMessage(content="recent prompt"),
                ]
            },
            summarization,
        )

        result = await compact_after_turn(agent, "thread-1")

        self.assertTrue(result.compacted)
        self.assertEqual(summarization.thread_ids, ["thread-1"])
        self.assertEqual(agent.updates[0][0], {"configurable": {"thread_id": "thread-1"}})
        event = agent.values["_summarization_event"]
        self.assertEqual(event["cutoff_index"], 1)
        self.assertEqual(event["file_path"], "/.mira/conversation_history/thread-1.md")
        self.assertIsInstance(event["summary_message"], HumanMessage)
        self.assertNotEqual(event["summary_message"].additional_kwargs.get("lc_source"), "summarization")
        rendered_archive = repr(summarization.offloaded[0][0])
        self.assertIn("Visible answer.", rendered_archive)
        self.assertNotIn("private chain", rendered_archive)
        self.assertNotIn("reasoning_content", rendered_archive)

    def test_checkpointed_summary_event_replays_as_human_message(self) -> None:
        summarization = FakeSummarization()
        mark_summarization_engine(summarization)
        event = {
            "cutoff_index": 1,
            "file_path": "/.mira/conversation_history/thread-1.md",
            "summary_message": {
                "type": "human",
                "content": (
                    "You are in the middle of a conversation that has been summarized.\n\n"
                    "<summary>\nEarlier context was summarized.\n</summary>"
                ),
                "additional_kwargs": {"lc_source": "summarization"},
                "response_metadata": {},
                "name": None,
                "id": None,
            },
        }

        effective = summarization._apply_event_to_messages(
            [HumanMessage(content="old"), HumanMessage(content="recent")],
            event,
        )

        self.assertIsInstance(effective[0], HumanMessage)
        self.assertEqual(effective[0].content, event["summary_message"]["content"])
        self.assertNotEqual(effective[0].additional_kwargs.get("lc_source"), "summarization")
        self.assertEqual([message.content for message in effective[1:]], ["recent"])

    def test_openai_style_summary_event_dict_is_normalized(self) -> None:
        summarization = FakeSummarization()
        mark_summarization_engine(summarization)
        event = {
            "cutoff_index": 0,
            "summary_message": {
                "role": "user",
                "content": "OpenAI-style summary message.",
                "additional_kwargs": {"lc_source": "summarization"},
            },
        }

        effective = summarization._apply_event_to_messages([HumanMessage(content="recent")], event)

        self.assertIsInstance(effective[0], HumanMessage)
        self.assertEqual(effective[0].content, "OpenAI-style summary message.")
        self.assertNotEqual(effective[0].additional_kwargs.get("lc_source"), "summarization")

    async def test_post_turn_compaction_accepts_checkpointed_summary_event(self) -> None:
        summarization = FakeSummarization()
        mark_summarization_engine(summarization)
        agent = AgentWithMutableState(
            {
                "messages": [HumanMessage(content="old"), HumanMessage(content="recent")],
                "_summarization_event": {
                    "cutoff_index": 1,
                    "summary_message": {
                        "type": "human",
                        "content": "Checkpointed summary.",
                        "additional_kwargs": {"lc_source": "summarization"},
                    },
                    "file_path": "/.mira/conversation_history/thread-1.md",
                },
            },
            summarization,
        )

        result = await compact_after_turn(agent, "thread-1")

        self.assertTrue(result.compacted)
        rendered_archive = repr(summarization.offloaded[0][0])
        self.assertIn("Checkpointed summary.", rendered_archive)
        self.assertNotIn("{'type': 'human'", rendered_archive)

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

    def test_recorder_separates_reasoning_around_intervening_events(self) -> None:
        cases = [
            ("assistant", lambda recorder: recorder.text_delta("mira text")),
            ("tool", lambda recorder: recorder.tool_call("read_file", {"path": "README.md"}, call_id="call-read")),
            ("delegation", lambda recorder: recorder.delegation_started([{"name": "task", "args": {"description": "judge"}}])),
            ("subagent", lambda recorder: recorder.subagent_started("general-purpose", "judge")),
            ("info", lambda recorder: recorder.info("status update")),
            ("error", lambda recorder: recorder.system_error("error update")),
            ("interrupted", lambda recorder: recorder.interrupted("turn interrupted")),
        ]

        for name, action in cases:
            with self.subTest(name=name):
                record = {"events": []}
                recorder = SessionRecorder(record, Store(), "action")

                recorder.reasoning_delta("first reasoning")
                action(recorder)
                recorder.reasoning_delta("second reasoning")
                recorder.finish_main()

                reasoning_events = [
                    event for event in context.normalize_events(record["events"]) if event["type"] == "reasoning"
                ]

                self.assertEqual([event["text"] for event in reasoning_events], ["first reasoning", "second reasoning"])

    def test_recorder_places_recovered_tool_result_before_last_assistant(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.tool_call("execute", {"command": "conda env list"}, call_id="call-execute")
        recorder.text_delta("The envs are ai_agents and base.")
        recorder.finish_main()
        recorder.recovered_tool_result("execute", "env list", call_id="call-execute")

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["tool_call", "tool_result", "assistant"])
        self.assertEqual(events[1]["call_id"], "call-execute")

    def test_recording_renderer_renders_recovered_tool_result(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.tool_call("execute", {"command": "conda env list"}, call_id="call-execute")
        recording.text_delta("The envs are ai_agents and base.")
        recording.finish_main()
        recording.recovered_tool_result("execute", "env list", call_id="call-execute")

        self.assertIn(("tool_result", "execute", "env list", "call-execute"), renderer.events)
        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["tool_call", "tool_result", "assistant"])

    def test_present_plan_tool_events_are_not_persisted_or_rendered(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "planning")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.tool_call("present_plan", {"title": "Plan"}, call_id="call-plan")
        recording.tool_result("present_plan", "interrupt", call_id="call-plan")
        recording.recovered_tool_result("present_plan", "interrupt", call_id="call-plan")

        self.assertEqual(renderer.events, [])
        self.assertEqual(context.normalize_events(record["events"]), [])

    def test_normalize_events_hides_legacy_present_plan_tool_events(self) -> None:
        record = {
            "events": [
                {"id": 1, "type": "tool_call", "mode": "planning", "name": "present_plan", "args": {}},
                {"id": 2, "type": "tool_result", "mode": "planning", "name": "present_plan", "output": "interrupt"},
                {
                    "id": 3,
                    "type": "plan",
                    "mode": "planning",
                    "status": "pending",
                    "plan": {
                        "id": "plan-1",
                        "title": "Plan",
                        "summary": ["One."],
                        "key_changes": ["Two."],
                        "test_plan": ["Three."],
                        "assumptions": ["Four."],
                    },
                },
            ]
        }

        events = context.normalize_events(record["events"])

        self.assertEqual([event["type"] for event in events], ["plan"])

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
        self.assertEqual(subagents[1]["output"], "done")
        self.assertEqual(subagents[2]["status"], "RUNNING")
        self.assertEqual(subagents[2]["output"], "")
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
        self.assertEqual(subagents[1]["output"], "done")

    def test_recorder_preserves_dynamic_subagent_origin(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.subagent_started("general-purpose [one]", "", origin="dynamic_tool_subagent")
        recorder.subagent_finished("general-purpose [one]", "done")

        subagents = [event for event in context.normalize_events(record["events"]) if event["type"] == "subagent"]
        self.assertEqual(subagents[0]["origin"], "dynamic_tool_subagent")
        self.assertEqual(subagents[1]["origin"], "dynamic_tool_subagent")

    def test_recorder_clears_dynamic_origin_when_task_request_arrives_late(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.subagent_started("general-purpose [one]", "", origin="dynamic_tool_subagent")
        recorder.subagent_request_updated("general-purpose [one]", "write scary story")
        recorder.subagent_finished("general-purpose [one]", "done")

        subagents = [event for event in context.normalize_events(record["events"]) if event["type"] == "subagent"]
        self.assertNotIn("origin", subagents[0])
        self.assertNotIn("origin", subagents[1])

    def test_eval_subagent_renderer_events_are_not_persisted_as_subagents(self) -> None:
        class EvalForwarder:
            def __init__(self) -> None:
                self.events: list[tuple[Any, ...]] = []

            def eval_subagent_started(
                self,
                name: str,
                task_input: str = "",
                *,
                eval_id: str = "",
                row_id: str = "",
                model: str = "",
            ) -> None:
                self.events.append(("eval_subagent_started", name, task_input, eval_id, row_id, model))

            def eval_subagent_finished(
                self,
                name: str,
                result: str = "",
                *,
                eval_id: str = "",
                row_id: str = "",
                duration_ms: int | None = None,
            ) -> None:
                self.events.append(("eval_subagent_finished", name, result, eval_id, row_id, duration_ms))

        record = {"events": []}
        renderer = EvalForwarder()
        recorder = SessionRecordingRenderer(renderer, SessionRecorder(record, Store(), "action"))

        recorder.eval_subagent_started(
            "general-purpose [one]",
            "judge pair",
            eval_id="eval-round-a",
            row_id="row-a",
            model="claude-haiku",
        )
        recorder.eval_subagent_finished(
            "general-purpose [one]",
            eval_id="eval-round-a",
            row_id="row-a",
            duration_ms=1200,
        )

        self.assertEqual(context.normalize_events(record["events"]), [])
        self.assertEqual(
            renderer.events,
            [
                ("eval_subagent_started", "general-purpose [one]", "judge pair", "eval-round-a", "row-a", "claude-haiku"),
                ("eval_subagent_finished", "general-purpose [one]", "", "eval-round-a", "row-a", 1200),
            ],
        )

    def test_done_subagent_output_contributes_to_resume_context(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "subagent",
                    "mode": "action",
                    "name": "general-purpose [one]",
                    "status": "RUNNING",
                    "task_input": "find dead code",
                    "output": "",
                },
                {
                    "id": 2,
                    "type": "subagent",
                    "mode": "action",
                    "name": "general-purpose [one]",
                    "status": "DONE",
                    "task_input": "find dead code",
                    "output": "No dead code found.",
                },
            ]
        }

        messages = context.normalize_messages(record["events"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "subagent")
        self.assertIn("general-purpose [one] completed", messages[0]["content"])
        self.assertIn("Request:\nfind dead code", messages[0]["content"])
        self.assertIn("Output:\nNo dead code found.", messages[0]["content"])

        resume = context.build_resume_context(record)
        self.assertIn("subagent (action):", resume)
        self.assertIn("No dead code found.", resume)

    def test_cancelled_subagent_output_contributes_to_resume_context(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "subagent",
                    "mode": "action",
                    "name": "general-purpose [one]",
                    "status": "CANCELLED",
                    "task_input": "inspect README",
                    "output": "Partial notes before cancellation.",
                },
            ]
        }

        messages = context.normalize_messages(record["events"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "subagent")
        self.assertIn("general-purpose [one] was cancelled", messages[0]["content"])
        self.assertIn("Partial notes before cancellation.", messages[0]["content"])

        resume = context.build_resume_context(record)
        self.assertIn("subagent (action):", resume)
        self.assertIn("Partial notes before cancellation.", resume)

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

    def test_recorder_does_not_temporarily_save_compaction_reasoning(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")

        recorder.reasoning_delta("The user wants me to extract context from the conversation history. ")
        recorder.reasoning_delta("Key information to extract: Session intent, Summary, Artifacts, Next Steps.")

        self.assertEqual(record["events"], [])
        self.assertEqual(store.saved, [])

    def test_recorder_drops_compaction_tail_fragment(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")

        recorder.reasoning_delta(": None - the task is complete")
        recorder.finish_main()

        self.assertEqual(record["events"], [])
        self.assertEqual(store.saved, [])

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

    async def test_high_context_turn_does_not_trigger_manual_post_turn_compaction(self) -> None:
        record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        store = Store()
        agent = AgentWithState({})
        calls = 0

        async def fake_run_turn(*args: Any, **kwargs: Any) -> runner.TurnResult:
            nonlocal calls
            calls += 1
            return runner.TurnResult(
                final_text="Why did the scarecrow win an award? Because he was outstanding in his field!",
                input_tokens=9900,
                output_tokens=89,
                total_tokens=9989,
                context_tokens=9989,
                context_source="usage_metadata",
                usage_source="usage_metadata",
            )

        with patch("ui.repl.run_turn", fake_run_turn):
            result = await run_user_turn(
                agent=agent,
                plan_agent=agent,
                renderer=RunTurnRenderer(),
                store=store,
                session=record,
                mode={"planning": False},
                text="tell me a joke",
                context_limit_tokens=10000,
            )

        self.assertEqual(calls, 1)
        self.assertIn("scarecrow", result.final_text)
        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["user", "assistant"])
        self.assertFalse(any(event["type"] == "compaction" for event in events))

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

    def test_resume_context_includes_recent_structured_plans(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "tool_call",
                    "mode": "planning",
                    "name": "present_plan",
                    "args": {"title": "hidden"},
                },
                {
                    "id": 2,
                    "type": "tool_result",
                    "mode": "planning",
                    "name": "present_plan",
                    "output": "interrupt",
                },
                {
                    "id": 3,
                    "type": "plan",
                    "mode": "planning",
                    "status": "revision requested",
                    "plan": {
                        "id": "plan-1",
                        "title": "Original Palindrome Plan",
                        "summary": ["Create palindrome.py."],
                        "key_changes": ["Add is_palindrome."],
                        "test_plan": ["Run python palindrome.py."],
                        "assumptions": ["Use Python."],
                    },
                },
                {
                    "id": 4,
                    "type": "plan",
                    "mode": "planning",
                    "status": "approved for implementation",
                    "plan": {
                        "id": "plan-2",
                        "title": "Revised Palindrome Plan",
                        "summary": ["Create palindrome.py with docs."],
                        "key_changes": ["Add type hints.", "Add a docstring."],
                        "test_plan": ["Run python palindrome.py.", "Verify Racecar is true."],
                        "assumptions": ["Use the project root."],
                    },
                },
            ],
        }

        plans = context.normalize_plans(record["events"])
        resume = context.build_resume_context(record)

        self.assertEqual([plan["id"] for plan in plans], ["plan-1", "plan-2"])
        self.assertIn("Recent structured plans:", resume)
        self.assertIn("plan-1 (revision requested): Original Palindrome Plan", resume)
        self.assertIn("plan-2 (approved for implementation): Revised Palindrome Plan", resume)
        self.assertIn("Summary:\n- Create palindrome.py with docs.", resume)
        self.assertIn("Key Changes:\n- Add type hints.\n- Add a docstring.", resume)
        self.assertIn("Test Plan:\n- Run python palindrome.py.\n- Verify Racecar is true.", resume)
        self.assertIn("Assumptions:\n- Use the project root.", resume)
        self.assertNotIn("tool_call", resume)
        self.assertNotIn("interrupt", resume)

    def test_resume_context_limits_recent_structured_plans(self) -> None:
        record = {
            "events": [
                {
                    "id": index,
                    "type": "plan",
                    "status": "discarded" if index == 2 else "pending",
                    "plan": {
                        "id": f"plan-{index}",
                        "title": f"Plan {index}",
                        "summary": [f"Summary {index}."],
                    },
                }
                for index in range(1, 5)
            ],
        }

        resume = context.build_resume_context(record)

        self.assertNotIn("plan-1 (pending): Plan 1", resume)
        self.assertIn("plan-2 (discarded): Plan 2", resume)
        self.assertIn("plan-3 (pending): Plan 3", resume)
        self.assertIn("plan-4 (pending): Plan 4", resume)

    def test_resume_context_pending_when_only_plan_events_exist(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "plan",
                    "status": "discarded",
                    "plan": {"id": "plan-1", "title": "Saved Plan", "summary": ["Do it."]},
                }
            ]
        }

        context.mark_resume_context_pending(record, resumed=True)

        self.assertTrue(record["resume_context_pending"])


if __name__ == "__main__":
    unittest.main()
