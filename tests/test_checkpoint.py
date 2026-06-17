"""Tests for MIRA's LangGraph checkpointer wiring."""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, convert_to_messages
from pydantic import BaseModel

from session.checkpoint import make_checkpointer


class Finding(BaseModel):
    """Small Pydantic model used to exercise schema-class checkpoint values."""

    summary: str


class CheckpointTests(unittest.TestCase):
    """Tests for checkpoint serialization fallbacks."""

    def test_message_instances_serialize_as_convertible_plain_dicts(self) -> None:
        serde = make_checkpointer().serde
        message = AIMessage(content="hello")

        kind, payload = serde.dumps_typed(message)
        value = serde.loads_typed((kind, payload))

        self.assertEqual(kind, "msgpack")
        self.assertIsInstance(value, dict)
        self.assertEqual(value["type"], "ai")
        self.assertEqual(value["content"], "hello")
        self.assertEqual(convert_to_messages([value])[0].content, "hello")

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
        self.assertIn("MockValSer", loaded.checkpoint["channel_values"]["state"]["serializer"])
        self.assertIn("MockValSer", loaded.pending_writes[0][2]["serializer"])


if __name__ == "__main__":
    unittest.main()
