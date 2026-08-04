"""Tests for durable session transcript helpers."""

from __future__ import annotations

import asyncio
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.compaction import PostTurnCompactionResult, compact_after_turn, prepare_summarization_engine
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from session import context
from session.dashboard import apply_context_usage, apply_turn_usage, ensure_dashboard
from session.goals import (
    clear_current_goal,
    current_goal,
    finish_goal_attempt,
    goal_artifact,
    replace_current_goal,
    start_goal_attempt,
)
from session.plans import plan_artifact
from session.recorder import RecordingRenderer as SessionRecordingRenderer
from session.recorder import SessionRecorder
from session.store import SessionStore
from runtime import runner
from runtime.message_events import consume_messages
from runtime.message_metadata import MessageInvocationMetadata
from runtime.tool_events import CONTROL_TOOLS
from tests.test_runner import (
    AsyncItems,
    COMPACTION_SUMMARY,
    FakeAgent,
    FakeStream,
    GatedAsyncItems,
    Message as StreamMessage,
    RunTurnRenderer,
    values_event,
)
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


class IneligibleFakeSummarization(FakeSummarization):
    def _is_eligible_for_compaction(self, messages: list[Any]) -> bool:
        return False


class EmptyRetentionFakeSummarization(FakeSummarization):
    def _determine_cutoff_index(self, messages: list[Any]) -> int:
        return 0


class FailingSummaryFakeSummarization(FakeSummarization):
    async def _acreate_summary(self, messages: list[Any]) -> str:
        raise RuntimeError("summary failed")


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
            custom = store.new(session_id="custom-session", workspace=Path("workspace"))
            custom["title"] = "Custom Session"
            custom["created_at"] = "2026-01-01T00:00:00+00:00"
            custom["updated_at"] = "2026-01-01T00:00:00+00:00"
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
                "current_plan",
                "current_goal",
                "events",
            ],
        )
        self.assertEqual(record["title"], "Untitled session")
        self.assertEqual(record["dashboard"]["context"]["percent"], 0.0)
        self.assertEqual(record["events"], [])
        self.assertIsNone(record["current_plan"])
        self.assertIsNone(record["current_goal"])
        self.assertNotIn("llm_direct", record)

    def test_transient_resume_flag_is_not_persisted(self) -> None:
        record = SessionStore(Path(".")).new(session_id="thread-1", workspace=Path("workspace"))
        record["resume_context_pending"] = True

        normalized = context.normalize_session(record)

        self.assertNotIn("resume_context_pending", normalized)

    def test_current_goal_survives_save_load_and_old_events_are_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new(session_id="thread-1", workspace=Path("workspace"))
            record["events"] = [
                {
                    "id": 1,
                    "type": "proposal",
                    "proposal": {
                        "id": "proposal-1",
                        "objective": "Build search.",
                        "criteria": "- Search works.",
                        "plan": {"title": "Search"},
                    },
                    "status": "approved for implementation",
                }
            ]
            store.save(record)
            self.assertEqual(record["events"], [])
            self.assertIsNone(store.read(store.path("thread-1"))["current_goal"])

            replace_current_goal(
                record,
                goal_artifact(
                    goal_id="goal-1",
                    title="Search",
                    objective="Build search.",
                    success_criteria="- Search works.",
                    rubric_enabled=False,
                    rubric_iterations=3,
                ),
            )
            store.save(record)
            restored = store.read(store.path("thread-1"))

        self.assertEqual(restored["current_goal"]["objective"], "Build search.")
        self.assertEqual(restored["current_goal"]["success_criteria"], "- Search works.")
        self.assertNotIn("plan", restored["current_goal"])
        self.assertEqual(restored["current_goal"]["status"], "proposed")

    def test_current_goal_status_transitions_preserve_history(self) -> None:
        record = {
            "events": [
                {"type": "goal", "goal": {"id": "goal-1"}, "status": "proposed"}
            ]
        }
        value = goal_artifact(goal_id="goal-1", title="Report", objective="Write report.", success_criteria="- Report exists.", rubric_enabled=True, rubric_iterations=2)
        replace_current_goal(record, value)
        start_goal_attempt(record)
        capped = finish_goal_attempt(record, rubric_status="max_iterations_reached")
        self.assertEqual(capped["status"], "max_iterations_reached")
        start_goal_attempt(record)
        completed = finish_goal_attempt(record, rubric_status="satisfied")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completion_source"], "rubric-verified")
        self.assertEqual(record["events"][0]["status"], "completed")
        self.assertEqual(clear_current_goal(record)["id"], "goal-1")
        self.assertIsNone(current_goal(record))

    def test_new_current_goal_supersedes_previous_goal(self) -> None:
        record = {
            "events": [
                {"type": "goal", "goal": {"id": "goal-1"}, "status": "proposed"},
            ]
        }
        first = goal_artifact(goal_id="goal-1", title="Work", objective="Do work.", success_criteria="- Work is done.", rubric_enabled=False, rubric_iterations=3)
        replace_current_goal(record, first)
        second = goal_artifact(goal_id="goal-2", title="More work", objective="Do more work.", success_criteria="- More work is done.", rubric_enabled=False, rubric_iterations=3)
        replace_current_goal(record, second)
        self.assertEqual(record["events"][0]["status"], "superseded")
        self.assertEqual(record["current_goal"]["id"], "goal-2")

    def test_action_resume_context_can_exclude_duplicate_current_goal(self) -> None:
        goal = goal_artifact(goal_id="goal-1", title="Search", objective="Build search.", success_criteria="- Search works.", rubric_enabled=True, rubric_iterations=3)
        record = {"current_goal": goal, "events": [{"id": 1, "type": "goal", "goal": goal, "status": "proposed"}]}

        planning_context = context.build_resume_context(record)
        action_context = context.build_resume_context(record, exclude_current_goal=True)

        self.assertIn("Authoritative current Goal", planning_context)
        self.assertNotIn("Authoritative current Goal", action_context)

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

    async def test_legacy_compaction_summary_aliases_are_ignored(self) -> None:
        for alias in ("summary", "summary_text"):
            with self.subTest(alias=alias):
                record = {"events": []}
                agent = AgentWithState(
                    {
                        "_summarization_event": {
                            "cutoff_index": 4,
                            "file_path": "/.mira/conversation_history/thread-1.md",
                            alias: "Earlier messages were summarized.",
                        }
                    }
                )

                await context.sync_deepagents_compaction(record, agent, "thread-1")

                compactions = context.normalize_compactions(record["events"])
                self.assertEqual(compactions[0]["summary"], "")

    async def test_post_turn_compaction_updates_summary_event_and_sanitizes_archive_messages(self) -> None:
        summarization = FakeSummarization()
        prepare_summarization_engine(summarization)
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

    async def test_manual_compaction_does_not_apply_agent_tool_eligibility_gate(self) -> None:
        summarization = IneligibleFakeSummarization()
        agent = AgentWithMutableState(
            {"messages": [HumanMessage(content="old"), HumanMessage(content="recent")]},
            summarization,
        )

        result = await compact_after_turn(agent, "thread-1")

        self.assertTrue(result.compacted)
        self.assertIn("_summarization_event", agent.values)

    async def test_manual_compaction_returns_noop_within_retention_window(self) -> None:
        summarization = EmptyRetentionFakeSummarization()
        agent = AgentWithMutableState(
            {"messages": [HumanMessage(content="recent")]},
            summarization,
        )

        result = await compact_after_turn(agent, "thread-1")

        self.assertFalse(result.compacted)
        self.assertEqual(result.reason, "nothing_to_compact")
        self.assertEqual(summarization.offloaded, [])
        self.assertEqual(agent.updates, [])

    async def test_manual_compaction_summary_failure_has_no_side_effects(self) -> None:
        summarization = FailingSummaryFakeSummarization()
        agent = AgentWithMutableState(
            {"messages": [HumanMessage(content="old"), HumanMessage(content="recent")]},
            summarization,
        )

        with self.assertRaisesRegex(RuntimeError, "summary failed"):
            await compact_after_turn(agent, "thread-1")

        self.assertEqual(summarization.offloaded, [])
        self.assertEqual(agent.updates, [])
        self.assertNotIn("_summarization_event", agent.values)

    def test_checkpointed_summary_event_replays_as_human_message(self) -> None:
        summarization = FakeSummarization()
        prepare_summarization_engine(summarization)
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
        prepare_summarization_engine(summarization)
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
        prepare_summarization_engine(summarization)
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

    async def test_compaction_sync_does_not_guess_event_type_from_wording(self) -> None:
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

        self.assertFalse(changed)
        self.assertEqual([event["type"] for event in record["events"]], ["reasoning", "compaction", "info"])

    def test_recorder_saves_info_regardless_of_compaction_wording(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.info(
            "The user wants me to extract the most important context from this conversation history. "
            "Key information to extract: Session intent, Summary, Artifacts, Next Steps."
        )

        self.assertEqual(record["events"][0]["type"], "info")
        self.assertIn("Key information to extract", record["events"][0]["text"])

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

    def test_correction_keeps_rejected_prose_and_retry_prompt_in_session(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "planning")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.text_delta("I'll inspect next.\nRESPONSE_STATUS: NEEDS_RESEARCH")
        recording.correction(
            {
                "type": "correction",
                "protocol": "plan_response_status",
                "check_name": "Response",
                "workflow": "Plan",
                "failed_check": (
                    "RESPONSE_STATUS: NEEDS_RESEARCH was declared without a tool call."
                ),
                "retry_prompt": "Perform that research now.",
                "attempt": 1,
                "max_retries": 2,
            }
        )
        recording.text_delta("Inspection complete.")

        events = context.normalize_events(record["events"])
        self.assertEqual(
            [(event["type"], event.get("text")) for event in events],
            [
                ("assistant", "I'll inspect next.\nRESPONSE_STATUS: NEEDS_RESEARCH"),
                ("correction", None),
                ("assistant", "Inspection complete."),
            ],
        )
        self.assertEqual(events[1]["retry_prompt"], "Perform that research now.")
        self.assertEqual(
            [message["role"] for message in context.normalize_messages(record["events"])],
            ["assistant", "correction", "assistant"],
        )
        self.assertTrue(any(event[0] == "correction" for event in renderer.events))

    def test_complete_response_status_survives_session_and_resume_context(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "planning")
        recording = SessionRecordingRenderer(RunTurnRenderer(), recorder)

        exact = "Navigation explained.\nRESPONSE_STATUS: COMPLETE"
        recording.text_delta(exact)
        recording.finish_main()

        events = context.normalize_events(record["events"])
        self.assertEqual(events[-1]["text"], exact)
        resume = context.build_resume_context(
            {"events": events, "current_plan": None, "current_goal": None}
        )
        self.assertIn(exact, resume)

    def test_correction_exhaustion_keeps_last_candidate_and_appends_failure(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "planning")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.text_delta("Still researching.\nRESPONSE_STATUS: NEEDS_RESEARCH")
        recording.correction(
            {
                "type": "correction",
                "protocol": "plan_response_status",
                "check_name": "Response",
                "workflow": "Plan",
                "failed_check": (
                    "RESPONSE_STATUS: NEEDS_RESEARCH was declared without a tool call."
                ),
                "attempt": 2,
                "max_retries": 2,
                "exhausted": True,
            }
        )
        recording.text_delta("MIRA could not produce a valid response.")

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["assistant", "correction", "assistant"])
        self.assertEqual(
            events[0]["text"],
            "Still researching.\nRESPONSE_STATUS: NEEDS_RESEARCH",
        )
        self.assertTrue(events[1]["exhausted"])
        self.assertEqual(events[2]["text"], "MIRA could not produce a valid response.")
        self.assertIn(
            "correction (planning): Response check (Plan) failed",
            context.build_resume_context({"events": events, "current_plan": None, "current_goal": None}),
        )

    def test_recovered_call_and_result_persist_before_final_assistant(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.text_delta("Done.")
        recording.finish_main()
        recording.recovered_tool_call(
            "read_file",
            {"file_path": "README.md"},
            call_id="call-read",
        )
        recording.recovered_tool_result(
            "read_file",
            "contents",
            call_id="call-read",
        )

        events = context.normalize_events(record["events"])
        self.assertEqual(
            [event["type"] for event in events],
            ["tool_call", "tool_result", "assistant"],
        )
        self.assertEqual(events[0]["call_id"], events[1]["call_id"])

    def test_completed_tool_result_preserves_active_assistant_and_call_grouping(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.tool_call("read_file", {"path": "README.md"}, call_id="call-read")
        recording.text_delta("The answer")
        recording.completed_tool_result("read_file", "contents", call_id="call-read")
        recording.text_delta(" continues.")

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["tool_call", "tool_result", "assistant"])
        self.assertEqual(events[1]["call_id"], "call-read")
        self.assertEqual(events[2]["text"], "The answer continues.")
        self.assertEqual(
            [event for event in renderer.events if event[0] == "tool_result"],
            [("tool_result", "read_file", "contents", "call-read")],
        )

    def test_completed_tool_error_persists_before_turn_failure(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)

        recording.tool_call("read_file", {"path": "missing.txt"}, call_id="call-read")
        recording.text_delta("Trying the file")
        recording.completed_tool_error("read_file", "file not found", call_id="call-read")
        recorder.system_error("turn error: graph failed")

        events = context.normalize_events(record["events"])
        self.assertEqual(
            [event["type"] for event in events],
            ["tool_call", "tool_result", "assistant", "system_error"],
        )
        self.assertEqual(events[1]["status"], "error")
        self.assertEqual(events[1]["call_id"], "call-read")
        self.assertIn(("tool_error", "read_file", "file not found", "call-read"), renderer.events)

    async def test_live_values_control_error_is_saved_before_graph_stream_finishes(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "plan")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)
        call = AIMessage(
            content="",
            tool_calls=[{"name": "finalize_plan", "args": {}, "id": "call-invalid"}],
        )
        error = ToolMessage(
            content="title is required",
            name="finalize_plan",
            tool_call_id="call-invalid",
            status="error",
        )
        raw_events = GatedAsyncItems(
            [values_event([]), values_event([call, error])],
            [values_event([call, error])],
        )
        stream = FakeStream(
            output={"messages": [call, error]},
            raw_events=raw_events,
        )
        turn = asyncio.create_task(
            runner.run_turn(FakeAgent([stream]), "finish", recording, "thread-1")
        )

        await asyncio.wait_for(raw_events.reached_gate.wait(), timeout=1)
        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["tool_call", "tool_result"])
        self.assertEqual(events[1]["status"], "error")
        self.assertEqual(events[1]["call_id"], "call-invalid")
        self.assertGreaterEqual(len(store.saved), 2)
        self.assertFalse(turn.done())

        raw_events.release.set()
        await turn

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["tool_call", "tool_result"])

    def test_completed_idless_results_group_by_original_call_order(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")

        recorder.tool_call("read_file", {"path": "one"})
        recorder.tool_call("read_file", {"path": "two"})
        recorder.completed_tool_result("read_file", "one")
        recorder.completed_tool_result("read_file", "two")

        events = context.normalize_events(record["events"])
        self.assertEqual(
            [(event["type"], event.get("output")) for event in events],
            [
                ("tool_call", None),
                ("tool_result", "one"),
                ("tool_call", None),
                ("tool_result", "two"),
            ],
        )

    def test_control_tool_calls_and_results_are_persisted_and_rendered(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "planning")
        renderer = RunTurnRenderer()
        recording = SessionRecordingRenderer(renderer, recorder)
        completions = {
            "ask_user": "Use A",
            "prepare_goal": "Success Criteria are ready. Continue to finalize_goal.",
            "prepare_plan": "Success Criteria are ready. Continue to finalize_plan.",
            "finalize_goal": "Goal presented for user review.",
            "finalize_plan": "Plan presented for user review.",
            "show_goal": "Current Goal rendered.",
            "show_plan": "Current Plan rendered.",
        }

        for name in CONTROL_TOOLS:
            call_id = f"call-{name}"
            recording.tool_call(name, {"value": name}, call_id=call_id)
            recording.completed_tool_result(
                name,
                completions[name],
                call_id=call_id,
            )

        events = context.normalize_events(record["events"])
        self.assertEqual(
            [event["type"] for event in events],
            ["tool_call", "tool_result"] * len(CONTROL_TOOLS),
        )
        for call, result in zip(events[::2], events[1::2], strict=True):
            self.assertEqual(call["name"], result["name"])
            self.assertEqual(call["call_id"], result["call_id"])
        self.assertEqual(
            {event[1] for event in renderer.events if event[0] == "tool_call"},
            CONTROL_TOOLS,
        )
        self.assertEqual(
            {event[1] for event in renderer.events if event[0] == "tool_result"},
            CONTROL_TOOLS,
        )

    async def test_recording_renderer_show_plan_uses_exact_terminal_fallback(self) -> None:
        class TerminalFallback:
            def __init__(self) -> None:
                self.plans: list[dict[str, Any]] = []

            def render_current_plan(self, value: dict[str, Any]) -> None:
                self.plans.append(value)

        retained = plan_artifact(
            plan_id="plan-1",
            title="Exact Plan",
            objective="Deliver the requested result.",
            context_and_constraints="Preserve unrelated work.",
            key_changes=["Produce the deliverable."],
            test_plan=["Verify the observable result."],
            assumptions=["No additional assumptions."],
            success_criteria="- The result is complete.",
            rubric_enabled=False,
            rubric_iterations=3,
        )
        record = {"events": [], "current_plan": retained}
        renderer = TerminalFallback()
        recording = SessionRecordingRenderer(
            renderer,
            SessionRecorder(record, Store(), "action"),
        )

        result = await recording.show_plan({"type": "show_plan"})

        self.assertEqual(result, "Current Plan rendered.")
        self.assertEqual(renderer.plans, [retained])

    async def test_recording_renderer_show_mismatches_name_the_current_artifact(self) -> None:
        goal = goal_artifact(
            goal_id="goal-1",
            title="Exact Goal",
            objective="Deliver the result.",
            success_criteria="- The result is complete.",
            rubric_enabled=False,
            rubric_iterations=3,
        )
        plan = plan_artifact(
            plan_id="plan-1",
            title="Exact Plan",
            objective="Deliver the result.",
            context_and_constraints="Preserve unrelated work.",
            key_changes=["Produce the deliverable."],
            test_plan=["Verify the result."],
            assumptions=["No additional assumptions."],
            success_criteria="- The result is complete.",
            rubric_enabled=False,
            rubric_iterations=3,
        )
        goal_recording = SessionRecordingRenderer(
            object(), SessionRecorder({"events": [], "current_goal": goal}, Store(), "action")
        )
        plan_recording = SessionRecordingRenderer(
            object(), SessionRecorder({"events": [], "current_plan": plan}, Store(), "action")
        )

        self.assertIn("Use /goal-show", await goal_recording.show_plan())
        self.assertIn("Use /plan-show", await plan_recording.show_goal())

    def test_rubric_results_are_persisted_as_events_and_reconciled(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        evaluation = {
            "grading_run_id": "grade-1",
            "iteration": 0,
            "result": "needs_revision",
            "grader_model": "lmstudio:bonsai",
            "duration_ms": 65_000,
            "explanation": "Missing a test.",
            "criteria": [{"name": "Tested", "passed": False, "gap": "No test."}],
        }

        recorder.rubric_evaluation_finished(evaluation, 1)
        recorder.rubric_evaluation_status("grade-1", 1, "max_iterations_reached")

        events = context.normalize_events(record["events"])
        self.assertEqual([event["type"] for event in events], ["rubric"])
        self.assertEqual(events[0]["evaluation"]["result"], "max_iterations_reached")
        self.assertEqual(events[0]["evaluation"]["grader_model"], "lmstudio:bonsai")
        self.assertEqual(events[0]["evaluation"]["duration_ms"], 65_000)
        self.assertEqual(events[0]["max_iterations"], 1)

    def test_proposal_events_are_discarded(self) -> None:
        events = [
            {
                "id": 1,
                "type": "proposal",
                "status": "pending",
                "proposal": {
                    "id": "proposal-1",
                    "kind": "plan",
                    "original_objective": "Build search.",
                    "objective": "Build search.\n\nResolved decisions:\n- Backend: SQLite",
                    "resolved_decisions": [{"question": "Backend?", "answer": "SQLite"}],
                    "criteria": "- Search returns ranked results.",
                    "plan": {"id": "plan-1", "title": "Search plan", "summary": ["Add search."]},
                    "rubric_iterations": 3,
                },
            }
        ]

        self.assertEqual(context.normalize_events(events), [])
        self.assertEqual(context.normalize_goals(events), [])
        self.assertEqual(context.build_resume_context({"events": events}), "")

    def test_malformed_persisted_iteration_values_restore_defaults(self) -> None:
        events = [
            {
                "id": 1,
                "type": "rubric",
                "evaluation": {"grading_run_id": "grade-1", "iteration": 0},
                "max_iterations": 99,
            },
        ]

        self.assertEqual(context.normalize_events(events)[0]["max_iterations"], 1)

    def test_normalize_events_replays_control_events_in_original_order(self) -> None:
        record = {
            "events": [
                {
                    "id": 1,
                    "type": "tool_call",
                    "mode": "planning",
                    "name": "ask_user",
                    "call_id": "call-ask",
                    "args": {
                        "question": "Which path?",
                        "options": ["Use A", "Use B"],
                    },
                },
                {
                    "id": 2,
                    "type": "tool_result",
                    "mode": "planning",
                    "name": "ask_user",
                    "call_id": "call-ask",
                    "output": "Use B",
                },
                {
                    "id": 3,
                    "type": "plan",
                    "mode": "planning",
                    "status": "pending",
                    "plan": plan_artifact(
                        plan_id="plan-1",
                        title="Plan",
                        objective="Complete the requested outcome.",
                        context_and_constraints="Preserve unrelated behavior.",
                        key_changes=["Two."],
                        test_plan=["Three."],
                        assumptions=["Four."],
                        success_criteria="- The requested outcome is complete.",
                        rubric_enabled=False,
                        rubric_iterations=3,
                    ),
                },
            ]
        }

        events = context.normalize_events(record["events"])

        self.assertEqual(
            [event["type"] for event in events],
            ["tool_call", "tool_result", "plan"],
        )
        self.assertEqual(events[0]["args"]["question"], "Which path?")
        self.assertEqual(events[0]["args"]["options"], ["Use A", "Use B"])
        self.assertEqual(events[0]["call_id"], "call-ask")
        self.assertEqual(events[1]["output"], "Use B")
        self.assertEqual(events[1]["call_id"], "call-ask")

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

    def test_recorder_saves_reasoning_without_phrase_detection(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")

        recorder.reasoning_delta("The user wants me to extract context from the conversation history. ")
        recorder.reasoning_delta("Key information to extract: Session intent, Summary, Artifacts, Next Steps.")

        self.assertEqual(record["events"][0]["type"], "reasoning")
        self.assertIn("Key information to extract", record["events"][0]["text"])
        self.assertTrue(store.saved)

    async def test_metadata_marked_summary_stream_is_not_recorded(self) -> None:
        record = {"events": []}
        recorder = SessionRecorder(record, Store(), "action")
        renderer = SessionRecordingRenderer(RunTurnRenderer(), recorder)
        registry = MessageInvocationMetadata()
        registry.record("summary-1", {"lc_source": "summarization"})

        await consume_messages(
            AsyncItems(
                [
                    StreamMessage(
                        reasoning=AsyncItems(["arbitrary hidden reasoning"]),
                        text=AsyncItems(["arbitrary hidden summary"]),
                        message_id="summary-1",
                    )
                ]
            ),
            renderer,
            invocation_metadata=registry,
        )

        self.assertEqual(record["events"], [])

    def test_recorder_keeps_reasoning_tail_without_phrase_detection(self) -> None:
        record = {"events": []}
        store = Store()
        recorder = SessionRecorder(record, store, "action")

        recorder.reasoning_delta(": None - the task is complete")
        recorder.finish_main()

        self.assertEqual(record["events"][0]["text"], ": None - the task is complete")
        self.assertTrue(store.saved)

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

    async def test_unmarked_summary_shaped_text_is_persisted_as_assistant(self) -> None:
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

        self.assertEqual(context.normalize_messages(record["events"])[0]["content"], COMPACTION_SUMMARY.strip())

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

    def test_old_historical_plan_events_are_discarded(self) -> None:
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

        self.assertEqual(context.normalize_events(record["events"]), [])
        self.assertEqual(context.build_resume_context(record), "")

    def test_resume_context_not_pending_for_historical_plan_events_only(self) -> None:
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

        self.assertFalse(record["resume_context_pending"])


if __name__ == "__main__":
    unittest.main()
