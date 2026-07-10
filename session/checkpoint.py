"""LangGraph checkpointer construction."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from pydantic import BaseModel
from pydantic_core import SchemaSerializer, SchemaValidator

try:
    from pydantic._internal._mock_val_ser import MockValSer
except Exception:  # pragma: no cover - pydantic private module availability
    MockValSer = None  # type: ignore[assignment]


class MiraJsonPlusSerializer(JsonPlusSerializer):
    """LangGraph serializer with MIRA checkpoint boundary normalization."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize normalized checkpoint values without pickle fallback."""
        normalized = normalize_checkpoint_value(obj)
        try:
            return super().dumps_typed(normalized)
        except Exception:
            return super().dumps_typed(sanitize_checkpoint_value(normalized))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize checkpoint values and keep risky internals plain."""
        return repair_loaded_checkpoint_value(normalize_checkpoint_value(super().loads_typed(data)))


class MiraMemorySaver(MemorySaver):
    """MemorySaver that normalizes risky values before checkpoint storage."""

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        """Save a checkpoint after normalizing MIRA-owned boundary values."""
        return super().put(
            config,
            normalize_checkpoint_value(checkpoint),
            normalize_checkpoint_value(metadata),
            new_versions,
        )

    def put_writes(self, config: Any, writes: Any, task_id: str, task_path: str = "") -> None:
        """Save writes after normalizing values that may contain schema objects."""
        normalized_writes = [(channel, normalize_checkpoint_value(value)) for channel, value in writes]
        return super().put_writes(config, normalized_writes, task_id, task_path)


def make_checkpointer() -> MemorySaver:
    """Create the in-memory LangGraph checkpointer used by both agents."""
    return MiraMemorySaver(serde=MiraJsonPlusSerializer())


def normalize_checkpoint_value(value: Any) -> Any:
    """Return checkpoint values in a safe plain-data shape where needed."""
    if value is None or isinstance(value, str | int | float | bool | bytes | bytearray):
        return value

    if is_pydantic_serializer_internal(value):
        return serializer_marker(value)

    if isinstance(value, _DeltaSnapshot):
        return _DeltaSnapshot(normalize_checkpoint_value(value.value))

    if isinstance(value, BaseMessage):
        return value

    if isinstance(value, type):
        return type_marker(value)

    if isinstance(value, BaseModel):
        try:
            return normalize_checkpoint_value(value.model_dump())
        except Exception:
            return repr(value)

    if isinstance(value, Mapping):
        return {normalize_mapping_key(key): normalize_checkpoint_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [normalize_checkpoint_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(normalize_checkpoint_value(item) for item in value)

    if isinstance(value, set | frozenset):
        return [normalize_checkpoint_value(item) for item in value]

    return value


def sanitize_checkpoint_value(value: Any) -> Any:
    """Return a conservative msgpack-safe representation of checkpoint values."""
    if value is None or isinstance(value, str | int | float | bool | bytes | bytearray):
        return value

    if is_pydantic_serializer_internal(value):
        return serializer_marker(value)

    if isinstance(value, _DeltaSnapshot):
        return _DeltaSnapshot(sanitize_checkpoint_value(value.value))

    if isinstance(value, BaseMessage):
        try:
            return sanitize_checkpoint_value(value.model_dump())
        except Exception:
            return safe_message_dict(value)

    if isinstance(value, type):
        return type_marker(value)

    if isinstance(value, Mapping):
        return {normalize_mapping_key(key): sanitize_checkpoint_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [sanitize_checkpoint_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_checkpoint_value(item) for item in value)

    if isinstance(value, set | frozenset):
        return [sanitize_checkpoint_value(item) for item in value]

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize_checkpoint_value(dataclasses.asdict(value))

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return sanitize_checkpoint_value(value.model_dump())
        except Exception:
            return repr(value)

    return repr(value)


def repair_loaded_checkpoint_value(value: Any) -> Any:
    """Repair narrow legacy checkpoint shapes produced by older MIRA builds."""
    if not isinstance(value, Mapping):
        return value

    channel_values = value.get("channel_values")
    if not isinstance(channel_values, Mapping):
        return value

    messages = channel_values.get("messages")
    if not is_legacy_corrupted_message_snapshot(messages):
        return value

    repaired = dict(value)
    repaired_channels = dict(channel_values)
    repaired_channels["messages"] = messages[0]
    repaired["channel_values"] = repaired_channels
    return repaired


def is_legacy_corrupted_message_snapshot(value: Any) -> bool:
    """Return whether value is the exact old `_DeltaSnapshot` -> nested-list bug."""
    if not isinstance(value, list) or len(value) != 1:
        return False
    inner = value[0]
    if not isinstance(inner, list) or not inner:
        return False
    return all(is_message_like(item) for item in inner)


def is_message_like(value: Any) -> bool:
    """Return whether value has a shape LangChain can convert into a message."""
    if isinstance(value, BaseMessage):
        return True
    if isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return bool({"role", "type"} & set(value.keys())) and "content" in value
    if isinstance(value, tuple) and len(value) == 2:
        return isinstance(value[0], str)
    return False


def message_snapshot_shape(value: Any, *, channel: str = "messages", source: str = "checkpoint") -> dict[str, Any]:
    """Return compact diagnostic details for a checkpoint message snapshot."""
    shape: dict[str, Any] = {
        "channel": channel,
        "kind": value.__class__.__name__,
        "source": source,
    }
    if isinstance(value, _DeltaSnapshot):
        shape["snapshot"] = True
        value = value.value
    if isinstance(value, list):
        shape["outer_len"] = len(value)
        shape["outer_item_types"] = [item.__class__.__name__ for item in value[:5]]
        if len(value) == 1 and isinstance(value[0], list):
            shape["nested"] = True
            shape["inner_len"] = len(value[0])
            shape["inner_item_types"] = [item.__class__.__name__ for item in value[0][:5]]
    return shape


def is_pydantic_serializer_internal(value: Any) -> bool:
    """Return whether value is a Pydantic serializer/validator implementation object."""
    internal_types = [SchemaSerializer, SchemaValidator]
    if MockValSer is not None:
        internal_types.append(MockValSer)
    return isinstance(value, tuple(internal_types))


def serializer_marker(value: Any) -> dict[str, str]:
    """Return a stable marker for Pydantic serializer/validator internals."""
    return {"__mira_pydantic_internal__": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}


def normalize_mapping_key(value: Any) -> Any:
    """Return a msgpack-safe mapping key."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def safe_message_dict(message: BaseMessage) -> dict[str, Any]:
    """Return a constructor-compatible message dict without using model_dump."""
    data: dict[str, Any] = {
        "type": getattr(message, "type", message.__class__.__name__.removesuffix("Message").lower()),
        "content": sanitize_checkpoint_value(getattr(message, "content", "")),
    }
    for key in (
        "additional_kwargs",
        "response_metadata",
        "name",
        "id",
        "tool_calls",
        "invalid_tool_calls",
        "usage_metadata",
        "tool_call_id",
        "artifact",
        "status",
    ):
        value = getattr(message, key, None)
        if value is not None:
            data[key] = sanitize_checkpoint_value(value)
    return data


def type_marker(value: type) -> dict[str, str]:
    """Return a stable marker for class objects in checkpoint payloads."""
    return {"__mira_type__": f"{value.__module__}.{value.__qualname__}"}
