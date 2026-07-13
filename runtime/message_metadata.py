"""Invocation metadata for LangGraph chat-model streams."""

from __future__ import annotations

from typing import Any

from langgraph.stream import ProtocolEvent, StreamTransformer


class MessageInvocationMetadata:
    """Record model-call metadata before message streams reach consumers."""

    def __init__(self) -> None:
        self._by_message_id: dict[str, dict[str, Any]] = {}

    def record(self, message_id: Any, metadata: Any) -> None:
        key = str(message_id or "")
        if key:
            self._by_message_id[key] = dict(metadata or {})

    def for_message(self, message: Any) -> dict[str, Any]:
        message_id = getattr(message, "message_id", None) or getattr(message, "id", None)
        return self._by_message_id.get(str(message_id or ""), {})

    def is_summarization(self, message: Any) -> bool:
        return self.for_message(message).get("lc_source") == "summarization"


class MessageInvocationMetadataTransformer(StreamTransformer):
    """Observe invocation metadata before LangGraph builds message streams."""

    before_builtins = True
    required_stream_modes = ("messages",)

    def __init__(self, scope: tuple[str, ...], registry: MessageInvocationMetadata) -> None:
        super().__init__(scope)
        self.registry = registry

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True

        params = event["params"]
        if params["namespace"] != list(self.scope):
            return True

        payload, metadata = params["data"]
        if isinstance(payload, dict):
            if payload.get("event") == "message-start":
                self.registry.record(payload.get("message_id") or payload.get("id"), metadata)
        else:
            self.registry.record(getattr(payload, "id", None), metadata)
        return True


__all__ = ["MessageInvocationMetadata", "MessageInvocationMetadataTransformer"]
