"""Enrich OpenTelemetry spans before generic OTLP export."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any

from openinference.instrumentation import (
    Message,
    PromptDetails,
    ReasoningMessageContent,
    TextMessageContent,
    TokenCount,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_session_attributes,
    get_span_kind_attributes,
    get_tag_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.util.types import AttributeValue

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
    """Add OpenInference semantics and configured attributes before export."""

    def __init__(self, span_attributes: Mapping[str, AttributeValue] | None = None) -> None:
        self._profile_attributes = dict(span_attributes or {})

    def on_end(self, span: Any) -> None:
        attributes = dict(span.attributes or {})
        if "langsmith.span.kind" in attributes:
            kind = _span_kind(attributes)
            enriched = _openinference_attributes(span.name, attributes, kind)
            _remove_misleading_genai_attributes(attributes, kind)
            attributes.update(enriched)
        elif not self._profile_attributes:
            return
        attributes.update(self._profile_attributes)

        # OTel SDK 1.37+ freezes the live span's BoundedAttributes before
        # processor callbacks. OpenInference's own conversion processors use
        # this replacement so later processors receive the enriched view.
        span._attributes = attributes


def _openinference_attributes(
    name: str,
    attributes: Mapping[str, Any],
    kind: OpenInferenceSpanKindValues,
) -> dict[str, Any]:
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

    if tags := _tags(attributes.get("langsmith.span.tags")):
        enriched.update(get_tag_attributes(tags=tags))
    if kind is OpenInferenceSpanKindValues.AGENT:
        if agent_name := attributes.get("langsmith.metadata.lc_agent_name"):
            enriched[SpanAttributes.AGENT_NAME] = str(agent_name)
    if kind is OpenInferenceSpanKindValues.LLM:
        enriched.update(_llm_attributes(attributes, prompt, completion))
    return enriched


def _span_kind(attributes: Mapping[str, Any]) -> OpenInferenceSpanKindValues:
    run_type = str(attributes.get("langsmith.span.kind", "")).upper()
    # DeepAgents propagates agent metadata to descendants. Only a chain run is
    # upgraded; authoritative model/tool/etc. run types retain their semantics.
    if run_type == "CHAIN" and attributes.get("langsmith.metadata.ls_agent_type"):
        return OpenInferenceSpanKindValues.AGENT
    try:
        kind = OpenInferenceSpanKindValues[run_type]
    except KeyError:
        return OpenInferenceSpanKindValues.CHAIN
    return (
        OpenInferenceSpanKindValues.CHAIN
        if kind is OpenInferenceSpanKindValues.UNKNOWN
        else kind
    )


def _remove_misleading_genai_attributes(
    attributes: dict[str, Any],
    kind: OpenInferenceSpanKindValues,
) -> None:
    attributes.pop("gen_ai.system", None)
    if kind is OpenInferenceSpanKindValues.LLM:
        return
    for key in tuple(attributes):
        if key.startswith(("gen_ai.request.", "gen_ai.response.", "gen_ai.usage.")):
            attributes.pop(key)


def _llm_attributes(
    attributes: Mapping[str, Any],
    prompt: Any,
    completion: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    model_name = attributes.get("gen_ai.response.model") or attributes.get(
        "gen_ai.request.model"
    )
    provider = _llm_provider(attributes)
    input_messages = _input_messages(prompt)
    output_messages = _output_messages(completion)
    token_count = _token_count(attributes)
    invocation_parameters = _invocation_parameters(attributes)
    values.update(
        get_llm_attributes(
            provider=str(provider) if provider else None,
            model_name=str(model_name) if model_name else None,
            invocation_parameters=invocation_parameters or None,
            input_messages=input_messages or None,
            output_messages=output_messages or None,
            token_count=token_count,
            tools=_llm_tools(attributes) or None,
        )
    )
    if finish_reason := attributes.get("gen_ai.response.finish_reasons"):
        values[SpanAttributes.LLM_FINISH_REASON] = _joined(finish_reason)
    details = _mapping_value(attributes.get("gen_ai.usage.output_token_details"))
    if details:
        detail_keys = {
            "audio": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
            "reasoning": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
        }
        for source, target in detail_keys.items():
            if isinstance(details.get(source), int):
                values[target] = details[source]
    return values


def _llm_provider(attributes: Mapping[str, Any]) -> Any:
    if provider := attributes.get("langsmith.metadata.provider"):
        return provider
    provider = attributes.get("langsmith.metadata.ls_provider")
    if provider and str(provider).lower() not in {"anyllm", "langchain"}:
        return provider
    system = attributes.get("gen_ai.system")
    return system if system and str(system).lower() != "langchain" else None


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
    if contents := _message_contents(data):
        message["contents"] = contents
    elif isinstance(data.get("content"), str) and data["content"]:
        message["content"] = str(data["content"])
    if tool_call_id := data.get("tool_call_id"):
        message["tool_call_id"] = str(tool_call_id)
    if tool_calls := _tool_calls(data):
        message["tool_calls"] = tool_calls
    return message


def _constructor_name(raw: Mapping[str, Any]) -> str:
    identifier = raw.get("id")
    return str(identifier[-1]) if isinstance(identifier, list) and identifier else ""


def _message_contents(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = data.get("content")
    contents: list[dict[str, Any]] = []
    has_reasoning = False
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping):
                continue
            block_type = item.get("type")
            if block_type == "reasoning":
                value = item.get("reasoning") or item.get("text")
                if value is not None:
                    contents.append(ReasoningMessageContent(type="reasoning", text=str(value)))
                    has_reasoning = True
            elif block_type in {"text", "text_delta"}:
                value = item.get("text") or item.get("content")
                if value is not None:
                    contents.append(TextMessageContent(type="text", text=str(value)))
    additional = data.get("additional_kwargs")
    fallback = additional.get("reasoning_content") if isinstance(additional, Mapping) else None
    if fallback and not has_reasoning:
        contents.insert(0, ReasoningMessageContent(type="reasoning", text=str(fallback)))
        if isinstance(content, str) and content:
            contents.append(TextMessageContent(type="text", text=content))
    return contents


def _tool_calls(data: Mapping[str, Any]) -> list[ToolCall]:
    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
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
    details = _mapping_value(attributes.get("gen_ai.usage.input_token_details"))
    if details:
        prompt_details = {
            "audio": details.get("audio"),
            "cache_read": details.get("cache_read"),
            "cache_write": details.get("cache_write", details.get("cache_creation")),
        }
        supported = {key: value for key, value in prompt_details.items() if isinstance(value, int)}
        if supported:
            values["prompt_details"] = PromptDetails(**supported)
    return TokenCount(**values) if values else None


def _invocation_parameters(attributes: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(_mapping_value(attributes.get("langsmith.metadata.invocation_params")) or {})
    for key in ("model", "model_name", "tools"):
        values.pop(key, None)
    prefix = "gen_ai.request."
    excluded = {"gen_ai.request.model", "gen_ai.request.tools"}
    values.update(
        {
            key.removeprefix(prefix): value
            for key, value in attributes.items()
            if key.startswith(prefix) and key not in excluded
        }
    )
    return values


def _llm_tools(attributes: Mapping[str, Any]) -> list[Tool]:
    invocation = _mapping_value(attributes.get("langsmith.metadata.invocation_params"))
    raw_tools = invocation.get("tools") if invocation else None
    if raw_tools is None:
        raw_tools = _json_value(attributes.get("gen_ai.request.tools"))
    if not isinstance(raw_tools, list):
        return []
    return [Tool(json_schema=tool) for tool in raw_tools if isinstance(tool, (str, dict))]


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, (list, tuple)):
        return [str(tag) for tag in value if str(tag)]
    return []


def _joined(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _mapping_value(value: Any) -> Mapping[str, Any] | None:
    parsed = _json_value(value)
    if isinstance(parsed, Mapping):
        return parsed
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


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
