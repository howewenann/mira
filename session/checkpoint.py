"""LangGraph checkpointer construction."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel


class MiraJsonPlusSerializer(JsonPlusSerializer):
    """LangGraph serializer with MIRA checkpoint boundary normalization."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize normalized checkpoint values without pickle fallback."""
        normalized = normalize_checkpoint_value(obj)
        try:
            return super().dumps_typed(normalized)
        except Exception:
            return super().dumps_typed(sanitize_checkpoint_value(normalized))


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

    if isinstance(value, BaseMessage):
        return normalize_checkpoint_value(value.model_dump())

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

    if isinstance(value, BaseMessage):
        return sanitize_checkpoint_value(value.model_dump())

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


def normalize_mapping_key(value: Any) -> Any:
    """Return a msgpack-safe mapping key."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def type_marker(value: type) -> dict[str, str]:
    """Return a stable marker for class objects in checkpoint payloads."""
    return {"__mira_type__": f"{value.__module__}.{value.__qualname__}"}
