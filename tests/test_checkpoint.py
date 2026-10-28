"""Tests for MIRA's stock LangGraph checkpointer wiring."""

from __future__ import annotations

import unittest

from deepagents._messages_reducer import _messages_delta_reducer
from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.types import Command
from pydantic import BaseModel
from pydantic_core import SchemaSerializer

from session.checkpoint import make_checkpointer


class Finding(BaseModel):
    """Small Pydantic model shared with SQLite and probe tests."""

    summary: str


class BindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports tool binding for agent tests."""

    def bind_tools(self, *args: object, **kwargs: object) -> "BindableFakeMessagesListChatModel":
        return self


def toy(command: str) -> str:
    """Run a toy command."""
    return f"ran {command}"


class CheckpointTests(unittest.TestCase):
    """Tests for stock checkpoint serialization and replay."""

    def test_memory_saver_uses_stock_default_serializer(self) -> None:
        saver = make_checkpointer()
        self.assertIs(type(saver), InMemorySaver)
        self.assertIs(type(saver.serde), JsonPlusSerializer)

    def test_message_instances_round_trip_as_messages(self) -> None:
        serde = make_checkpointer().serde
        message = AIMessage(content="hello")

        kind, payload = serde.dumps_typed(message)
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, AIMessage)
        self.assertEqual(value.content, "hello")
        self.assertEqual(convert_to_messages([value])[0].content, "hello")

    def test_delta_snapshot_round_trip_replays_as_message_state(self) -> None:
        serde = make_checkpointer().serde
        snapshot = _DeltaSnapshot([
            HumanMessage(content="user"),
            AIMessage(content="assistant"),
            ToolMessage(content="tool", tool_call_id="call-1"),
        ])

        kind, payload = serde.dumps_typed(snapshot)
        loaded = serde.loads_typed((kind, payload))
        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(loaded, _DeltaSnapshot)
        self.assertEqual([message.content for message in loaded.value], ["user", "assistant", "tool"])

        channel = DeltaChannel(_messages_delta_reducer, list, snapshot_frequency=50).from_checkpoint(loaded)
        channel.update([[HumanMessage(content="next")]])
        self.assertEqual([message.__class__.__name__ for message in channel.get()], [
            "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage",
        ])

    def test_memory_saver_restores_plain_state_messages_and_pending_writes(self) -> None:
        saver = make_checkpointer()
        config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
        state = {"nested": {"items": [1, "two"], "enabled": True}}
        checkpoint = {
            "v": 4,
            "ts": "2026-01-01T00:00:00+00:00",
            "id": "checkpoint-1",
            "channel_values": {"messages": [AIMessage(content="stored")], "state": state},
            "channel_versions": {"messages": "1", "state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        saved = saver.put(config, checkpoint, {}, {"messages": "1", "state": "1"})
        saver.put_writes(saved, [("state", {"finding": Finding(summary="queued")})], "task-1")

        loaded = saver.get_tuple(saved)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIsInstance(loaded.checkpoint["channel_values"]["messages"][0], AIMessage)
        self.assertEqual(loaded.checkpoint["channel_values"]["messages"][0].content, "stored")
        self.assertEqual(loaded.checkpoint["channel_values"]["state"], state)
        self.assertEqual(loaded.pending_writes, [("task-1", "state", {"finding": Finding(summary="queued")})])

    def test_pydantic_model_round_trips_as_model(self) -> None:
        serde = make_checkpointer().serde
        kind, payload = serde.dumps_typed(Finding(summary="ok"))
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, Finding)
        self.assertEqual(value.summary, "ok")

    def test_unsupported_runtime_objects_raise(self) -> None:
        serde = make_checkpointer().serde
        for value in (AIMessage, BaseModel.__pydantic_serializer__, SchemaSerializer({"type": "any"})):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    serde.dumps_typed({"value": value})


class CheckpointHitlTests(unittest.IsolatedAsyncioTestCase):
    """Integration coverage for native LangGraph HITL resume with MIRA checkpoints."""

    async def test_native_hitl_resume_preserves_ai_tool_and_final_messages(self) -> None:
        model = BindableFakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "toy",
                            "args": {"command": "conda env list"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="final with joke"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[toy],
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"toy": True})],
            checkpointer=make_checkpointer(),
        )
        config = {"configurable": {"thread_id": "checkpoint-hitl"}}

        stream = await graph.astream_events(
            {"messages": [{"role": "user", "content": "run toy then joke"}]},
            config=config,
            version="v3",
        )
        await stream.output()
        self.assertEqual(len(await stream.interrupts()), 1)

        stream = await graph.astream_events(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            version="v3",
        )
        output = await stream.output()

        messages = output["messages"]
        self.assertEqual([message.__class__.__name__ for message in messages], [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "AIMessage",
        ])
        self.assertEqual(messages[2].content, "ran conda env list")
        self.assertEqual(messages[3].content, "final with joke")


if __name__ == "__main__":
    unittest.main()
