"""Tests for MIRA's LangGraph checkpointer wiring."""

from __future__ import annotations

import unittest

from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.types import Command
from pydantic import BaseModel
from pydantic_core import SchemaSerializer, SchemaValidator

from deepagents._messages_reducer import _messages_delta_reducer

from session.checkpoint import make_checkpointer, message_snapshot_shape, sanitize_checkpoint_value


class Finding(BaseModel):
    """Small Pydantic model used to exercise schema-class checkpoint values."""

    summary: str


class BrokenDump:
    """Object shaped like Pydantic but failing during model_dump."""

    def model_dump(self) -> dict[str, object]:
        raise TypeError("'MockValSer' object is not an instance of 'SchemaSerializer'")


class BrokenAIMessage(AIMessage):
    """AIMessage test double whose model_dump path fails."""

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise TypeError("broken message dump")


class BindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports tool binding for agent tests."""

    def bind_tools(self, *args: object, **kwargs: object) -> "BindableFakeMessagesListChatModel":
        return self


def toy(command: str) -> str:
    """Run a toy command."""
    return f"ran {command}"


class CheckpointTests(unittest.TestCase):
    """Tests for checkpoint serialization fallbacks."""

    def test_message_instances_round_trip_as_messages(self) -> None:
        serde = make_checkpointer().serde
        message = AIMessage(content="hello")

        kind, payload = serde.dumps_typed(message)
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, AIMessage)
        self.assertEqual(value.content, "hello")
        self.assertEqual(convert_to_messages([value])[0].content, "hello")

    def test_raw_jsonplus_preserves_delta_snapshots(self) -> None:
        serde = JsonPlusSerializer()
        snapshot = _DeltaSnapshot([HumanMessage(content="hello")])

        kind, payload = serde.dumps_typed(snapshot)
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, _DeltaSnapshot)
        self.assertEqual(value.value[0].content, "hello")

    def test_mira_jsonplus_preserves_delta_snapshots(self) -> None:
        serde = make_checkpointer().serde
        snapshot = _DeltaSnapshot([HumanMessage(content="hello")])

        kind, payload = serde.dumps_typed(snapshot)
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, _DeltaSnapshot)
        self.assertEqual(value.value[0].content, "hello")

    def test_delta_snapshot_round_trip_replays_as_message_state(self) -> None:
        serde = make_checkpointer().serde
        snapshot = _DeltaSnapshot([
            HumanMessage(content="user"),
            AIMessage(content="assistant"),
            ToolMessage(content="tool", tool_call_id="call-1"),
        ])

        kind, payload = serde.dumps_typed(snapshot)
        loaded = serde.loads_typed((kind, payload))
        channel = DeltaChannel(_messages_delta_reducer, list, snapshot_frequency=50).from_checkpoint(loaded)

        channel.update([[HumanMessage(content="next")]])

        self.assertEqual([message.__class__.__name__ for message in channel.get()], [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "HumanMessage",
        ])

    def test_load_repairs_legacy_corrupted_message_snapshot(self) -> None:
        raw_serde = JsonPlusSerializer()
        mira_serde = make_checkpointer().serde
        corrupted = {
            "channel_values": {
                "messages": [[
                    HumanMessage(content="user"),
                    AIMessage(content="assistant"),
                    ToolMessage(content="tool", tool_call_id="call-1"),
                ]],
                "state": [[AIMessage(content="not a messages channel")]],
            },
        }

        data = raw_serde.dumps_typed(corrupted)
        value = mira_serde.loads_typed(data)

        self.assertEqual([message.__class__.__name__ for message in value["channel_values"]["messages"]], [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
        ])
        self.assertIsInstance(value["channel_values"]["state"][0], list)

    def test_message_snapshot_shape_reports_nested_corruption(self) -> None:
        shape = message_snapshot_shape(
            [[HumanMessage(content="user"), AIMessage(content="assistant")]],
            source="write",
        )

        self.assertEqual(shape["channel"], "messages")
        self.assertEqual(shape["source"], "write")
        self.assertEqual(shape["outer_len"], 1)
        self.assertTrue(shape["nested"])
        self.assertEqual(shape["inner_len"], 2)
        self.assertEqual(shape["inner_item_types"], ["HumanMessage", "AIMessage"])

    def test_checkpointer_serializes_message_type_markers(self) -> None:
        serde = make_checkpointer().serde

        kind, payload = serde.dumps_typed({"schema": AIMessage, "finding": Finding})
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertEqual(
            value,
            {
                "schema": {"__mira_type__": "langchain_core.messages.ai.AIMessage"},
                "finding": {"__mira_type__": "tests.test_checkpoint.Finding"},
            },
        )

    def test_checkpointer_round_trips_writes_with_nested_risky_values(self) -> None:
        checkpointer = make_checkpointer()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        checkpoint = {
            "v": 4,
            "ts": "2026-01-01T00:00:00+00:00",
            "id": "checkpoint-1",
            "channel_values": {
                "messages": [AIMessage(content="stored")],
                "state": {"nested": [AIMessage, Finding]},
            },
            "channel_versions": {"messages": "1", "state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        saved_config = checkpointer.put(config, checkpoint, {}, {"messages": "1", "state": "1"})

        checkpointer.put_writes(saved_config, [("state", {"nested": [AIMessage, Finding]})], "task-1")

        loaded = checkpointer.get_tuple(saved_config)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        message = convert_to_messages(loaded.checkpoint["channel_values"]["messages"])[0]
        self.assertEqual(message.content, "stored")
        self.assertEqual(
            loaded.checkpoint["channel_values"]["state"],
            {
                "nested": [
                    {"__mira_type__": "langchain_core.messages.ai.AIMessage"},
                    {"__mira_type__": "tests.test_checkpoint.Finding"},
                ]
            },
        )
        self.assertEqual(
            loaded.pending_writes,
            [
                (
                    "task-1",
                    "state",
                    {
                        "nested": [
                            {"__mira_type__": "langchain_core.messages.ai.AIMessage"},
                            {"__mira_type__": "tests.test_checkpoint.Finding"},
                        ]
                    },
                )
            ],
        )

    def test_pydantic_model_instances_serialize_as_plain_dicts(self) -> None:
        serde = make_checkpointer().serde

        kind, payload = serde.dumps_typed({"finding": Finding(summary="ok")})
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertEqual(value, {"finding": {"summary": "ok"}})

    def test_checkpointer_survives_pydantic_mock_serializer_values(self) -> None:
        checkpointer = make_checkpointer()
        config = {
            "configurable": {
                "thread_id": "thread-mock",
                "checkpoint_ns": "",
            }
        }
        checkpoint = {
            "v": 4,
            "ts": "2026-01-01T00:00:00+00:00",
            "id": "checkpoint-mock",
            "channel_values": {
                "state": {"serializer": BaseModel.__pydantic_serializer__},
            },
            "channel_versions": {"state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        saved_config = checkpointer.put(config, checkpoint, {}, {"state": "1"})
        checkpointer.put_writes(
            saved_config,
            [("state", {"serializer": BaseModel.__pydantic_serializer__})],
            "task-mock",
        )

        loaded = checkpointer.get_tuple(saved_config)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        expected = {
            "__mira_pydantic_internal__": "pydantic._internal._mock_val_ser.MockValSer",
        }
        self.assertEqual(loaded.checkpoint["channel_values"]["state"]["serializer"], expected)
        self.assertEqual(loaded.pending_writes[0][2]["serializer"], expected)

    def test_pydantic_serializer_internals_become_stable_markers(self) -> None:
        serde = make_checkpointer().serde

        kind, payload = serde.dumps_typed(
            {
                "mock": BaseModel.__pydantic_serializer__,
                "serializer": SchemaSerializer({"type": "any"}),
                "validator": SchemaValidator({"type": "any"}),
            }
        )
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertEqual(
            value,
            {
                "mock": {
                    "__mira_pydantic_internal__": "pydantic._internal._mock_val_ser.MockValSer",
                },
                "serializer": {
                    "__mira_pydantic_internal__": "pydantic_core._pydantic_core.SchemaSerializer",
                },
                "validator": {
                    "__mira_pydantic_internal__": "pydantic_core._pydantic_core.SchemaValidator",
                },
            },
        )

    def test_model_dump_schema_serializer_error_is_sanitized(self) -> None:
        serde = make_checkpointer().serde

        kind, payload = serde.dumps_typed({"broken": BrokenDump()})
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIn("BrokenDump", value["broken"])

    def test_message_sanitize_fallback_stays_structured(self) -> None:
        message = BrokenAIMessage(
            content="hello",
            tool_calls=[{"name": "toy", "args": {"command": "x"}, "id": "call-1"}],
        )

        value = sanitize_checkpoint_value(message)

        self.assertEqual(value["type"], "ai")
        self.assertEqual(value["content"], "hello")
        self.assertEqual(value["tool_calls"][0]["name"], "toy")
        self.assertNotIn("AIMessage(content=", str(value))


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
