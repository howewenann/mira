"""Enrich LangSmith-created OpenTelemetry spans with OpenInference semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from openinference.instrumentation import (
    Message,
    TokenCount,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_session_attributes,
    get_span_kind_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.trace import SpanProcessor

_KIND_BY_LANGSMITH_TYPE = {
    "llm": OpenInferenceSpanKindValues.LLM,
    "tool": OpenInferenceSpanKindValues.TOOL,
    "retriever": OpenInferenceSpanKindValues.RETRIEVER,
}
_ROLE_BY_MESSAGE_TYPE = {
    "ai": "assistant",
    "aimessage": "assistant",
    "assistant": "assistant",
    "human": "user",
    "humanmessage": "user",
    "system": "system",
    "systemmessage": "system",
    "tool": "tool",
    "toolmessage": "tool",
}


class LangSmithOpenInferenceProcessor(SpanProcessor):
    """Augment the readable LangSmith span before downstream export."""

    def on_end(self, span: Any) -> None:
        attributes = dict(span.attributes or {})
        if "langsmith.span.kind" not in attributes:
            return
        attributes.update(_openinference_attributes(span.name, attributes))

        # OTel SDK 1.37+ freezes the live span's BoundedAttributes before
        # processor callbacks. OpenInference's own conversion processors use
        # this replacement so later processors receive the enriched view.
        span._attributes = attributes


def _openinference_attributes(name: str, attributes: Mapping[str, Any]) -> dict[str, Any]:
    kind = _span_kind(name, attributes)
    enriched = dict(get_span_kind_attributes(kind))
    prompt = _json_value(attributes.get("gen_ai.prompt"))
    completion = _json_value(attributes.get("gen_ai.completion"))

    if kind is OpenInferenceSpanKindValues.TOOL:
        enriched[SpanAttributes.TOOL_NAME] = str(
            attributes.get("langsmith.trace.name") or name
        )
        completion = _tool_output(completion)

    if prompt is not None:
        enriched.update(get_input_attributes(prompt))
    if completion is not None:
        enriched.update(get_output_attributes(completion))

    session_id = attributes.get("langsmith.metadata.thread_id")
    if session_id:
        enriched.update(get_session_attributes(session_id=str(session_id)))

    if kind is OpenInferenceSpanKindValues.LLM:
        enriched.update(_llm_attributes(attributes, prompt, completion))
    return enriched


def _span_kind(name: str, attributes: Mapping[str, Any]) -> OpenInferenceSpanKindValues:
    run_type = str(attributes.get("langsmith.span.kind", "")).lower()
    if mapped := _KIND_BY_LANGSMITH_TYPE.get(run_type):
        return mapped
    if "agent" in name.lower():
        return OpenInferenceSpanKindValues.AGENT
    return OpenInferenceSpanKindValues.CHAIN


def _llm_attributes(
    attributes: Mapping[str, Any],
    prompt: Any,
    completion: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    model_name = attributes.get("gen_ai.response.model") or attributes.get(
        "gen_ai.request.model"
    )
    provider = attributes.get("langsmith.metadata.ls_provider")
    system = attributes.get("langsmith.metadata.provider")
    input_messages = _input_messages(prompt)
    output_messages = _output_messages(completion)
    token_count = _token_count(attributes)
    invocation_parameters = _invocation_parameters(attributes)
    values.update(
        get_llm_attributes(
            provider=str(provider) if provider else None,
            system=str(system) if system else None,
            model_name=str(model_name) if model_name else None,
            invocation_parameters=invocation_parameters or None,
            input_messages=input_messages or None,
            output_messages=output_messages or None,
            token_count=token_count,
        )
    )
    return values


def _input_messages(payload: Any) -> list[Message]:
    if not isinstance(payload, Mapping):
        return []
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return []
    if raw_messages and all(isinstance(item, list) for item in raw_messages):
        raw_messages = [message for batch in raw_messages for message in batch]
    return [message for item in raw_messages if (message := _message(item))]


def _output_messages(payload: Any) -> list[Message]:
    if not isinstance(payload, Mapping):
        return []
    generations = payload.get("generations")
    if not isinstance(generations, list):
        return []
    while generations and isinstance(generations[0], list):
        generations = [item for group in generations for item in group]
    messages: list[Message] = []
    for generation in generations:
        raw = generation.get("message") if isinstance(generation, Mapping) else None
        if message := _message(raw):
            messages.append(message)
    return messages


def _message(raw: Any) -> Message | None:
    if not isinstance(raw, Mapping):
        return None
    data = raw.get("kwargs") if isinstance(raw.get("kwargs"), Mapping) else raw
    assert isinstance(data, Mapping)
    message_type = str(data.get("type") or _constructor_name(raw)).lower()
    role = _ROLE_BY_MESSAGE_TYPE.get(message_type)
    if role is None:
        return None

    message = Message(role=role)
    if text := _message_text(data.get("content")):
        message["content"] = text
    if tool_call_id := data.get("tool_call_id"):
        message["tool_call_id"] = str(tool_call_id)
    if tool_calls := _tool_calls(data):
        message["tool_calls"] = tool_calls
    return message


def _constructor_name(raw: Mapping[str, Any]) -> str:
    identifier = raw.get("id")
    return str(identifier[-1]) if isinstance(identifier, list) and identifier else ""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") not in {"text", "text_delta"}:
            continue
        value = item.get("text") or item.get("content")
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def _tool_calls(data: Mapping[str, Any]) -> list[ToolCall]:
    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list):
        content = data.get("content")
        raw_calls = (
            [
                item
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "tool_call"
            ]
            if isinstance(content, list)
            else []
        )
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), Mapping) else raw
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments", function.get("args", {}))
        call = ToolCall(function=ToolCallFunction(name=str(name), arguments=arguments))
        if call_id := raw.get("id"):
            call["id"] = str(call_id)
        calls.append(call)
    return calls


def _token_count(attributes: Mapping[str, Any]) -> TokenCount | None:
    keys = {
        "prompt": "gen_ai.usage.input_tokens",
        "completion": "gen_ai.usage.output_tokens",
        "total": "gen_ai.usage.total_tokens",
    }
    values = {
        target: int(attributes[source])
        for target, source in keys.items()
        if isinstance(attributes.get(source), int)
    }
    return TokenCount(**values) if values else None


def _invocation_parameters(attributes: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "gen_ai.request."
    excluded = {"gen_ai.request.model"}
    return {
        key.removeprefix(prefix): value
        for key, value in attributes.items()
        if key.startswith(prefix) and key not in excluded
    }


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _tool_output(value: Any) -> Any:
    if isinstance(value, Mapping) and "output" in value:
        return value["output"]
    return value
