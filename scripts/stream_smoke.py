"""Opt-in live streaming smoke test for DeepAgents event timing.

This script intentionally avoids SessionStore/SessionRecorder so it does not
write normal .mira/_sessions files. It needs the configured model endpoint
running, for example LM Studio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.factory import build_agent  # noqa: E402
from config.loader import load_config  # noqa: E402
from config.metadata import infer_model_metadata  # noqa: E402
from runtime.message_events import consume_messages  # noqa: E402
from runtime.output_events import capture_output, final_text  # noqa: E402
from runtime.runner import SubagentRequestRenderer, run_turn  # noqa: E402
from runtime.subagent_events import consume_subagents  # noqa: E402
from runtime.tool_events import consume_tool_calls  # noqa: E402
from session.checkpoint import make_checkpointer  # noqa: E402

DEFAULT_PROMPT = "use 2 subagents to tell me 2 100 word different stories, one scary, one funny"


class SmokeRenderer:
    """Minimal renderer that records timestamped streaming callbacks."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self._subagent_labels: dict[int, str] = {}
        self._subagent_requests: dict[str, str] = {}
        self._blank_subagent_starts: list[str] = []
        self._seen_text = False
        self._seen_reasoning = False

    def startup_progress(self, state: str) -> None:
        self._record("startup", state=state)

    def waiting_started(self) -> None:
        self._record("waiting_started")

    def waiting_finished(self) -> None:
        self._record("waiting_finished")

    def reasoning_delta(self, delta: str) -> None:
        if delta and not self._seen_reasoning:
            self._seen_reasoning = True
            self._record("reasoning", chars=len(delta), sample=sample(delta))

    def text_delta(self, delta: str) -> None:
        if delta and not self._seen_text:
            self._seen_text = True
            self._record("text", chars=len(delta), sample=sample(delta))

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self._record("tool_call", name=name, call_id=call_id, args=args)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        self._record("tool_result", name=name, call_id=call_id, chars=len(result or ""))

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        self._record("delegation_delta", calls=simplify_calls(calls))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self._record("delegation_started", calls=simplify_calls(calls))

    def start_subagent_live(self) -> None:
        self._record("subagents_live_started")

    def stop_subagent_live(self) -> None:
        self._record("subagents_live_stopped")

    def subagent_label(self, subagent: Any) -> str:
        key = id(subagent)
        if key not in self._subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self._subagent_labels[key] = f"{name} [{len(self._subagent_labels) + 1}]"
        return self._subagent_labels[key]

    def subagent_started(self, subagent: str, task_input: str = "", *, origin: str = "") -> None:
        if not task_input:
            self._blank_subagent_starts.append(subagent)
        else:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_started", name=subagent, request=task_input, origin=origin)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        if task_input:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_request_updated", name=subagent, request=task_input)

    def subagent_finished(self, subagent: str, result: str = "", task_input: str = "") -> None:
        if task_input:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_finished", name=subagent, chars=len(result or ""), request=task_input)

    def subagent_cancelled(self, subagent: str, result: str = "", task_input: str = "") -> None:
        self._record("subagent_cancelled", name=subagent, chars=len(result or ""), request=task_input)

    def subagents_cancelled(self) -> None:
        self._record("subagents_cancelled")

    def tick_subagents(self) -> None:
        return None

    def finish_main(self) -> None:
        return None

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        raise RuntimeError(f"Smoke prompt unexpectedly requested approvals: {interrupts!r}")

    async def ask_user(self, interrupt: Any) -> str:
        raise RuntimeError(f"Smoke prompt unexpectedly asked the user: {interrupt!r}")

    def _record(self, kind: str, **payload: Any) -> None:
        event = {"t": round(time.monotonic() - self.started_at, 3), "kind": kind, **payload}
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=True, default=str), flush=True)

    def raw_event(self, event: dict[str, Any]) -> None:
        self._record("raw_event", **raw_event_summary(event))


def simplify_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified = []
    for call in calls:
        args = call.get("args", {}) if isinstance(call, dict) else {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        args = args if isinstance(args, dict) else {"raw": str(args)}
        simplified.append(
            {
                "id": str(call.get("id") or call.get("call_id") or ""),
                "name": str(call.get("name") or call.get("tool_name") or ""),
                "subagent_type": str(args.get("subagent_type") or ""),
                "description": str(args.get("description") or ""),
            }
        )
    return simplified


def sample(text: str, limit: int = 80) -> str:
    normalized = " ".join(str(text).split())
    return normalized[:limit]


def raw_event_summary(event: Any) -> dict[str, Any]:
    """Return a compact, privacy-conscious summary of a raw protocol event."""
    if not isinstance(event, dict):
        return {"event_type": type(event).__name__}

    params = event.get("params") if isinstance(event.get("params"), dict) else {}
    namespace = params.get("namespace", event.get("namespace", []))
    data = params.get("data", event.get("data"))
    method = str(event.get("method") or event.get("event") or "")
    summary: dict[str, Any] = {
        "method": method,
        "namespace": compact_namespace(namespace),
        "payload_kind": payload_kind(data),
    }
    summary.update(protocol_type_summary(data))
    tool = find_tool_payload(data)
    if tool:
        summary["tool_like"] = tool
    text = text_sample(data)
    if text:
        summary["sample"] = text
    protocol_sample = protocol_value_sample(data)
    if protocol_sample:
        summary["protocol_sample"] = protocol_sample
    return summary


def sse_chunk_summary(chunk: Any) -> dict[str, Any]:
    """Return compact fields from an OpenAI-compatible SSE JSON chunk."""
    if not isinstance(chunk, dict):
        return {"payload_kind": type(chunk).__name__}

    choices = chunk.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    delta = choice.get("delta") if isinstance(choice, dict) and isinstance(choice.get("delta"), dict) else {}
    summary: dict[str, Any] = {
        "payload_kind": "chat.completion.chunk",
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "delta_keys": sorted(str(key) for key in delta.keys()),
    }
    tool = find_tool_payload(delta)
    if tool:
        summary["tool_like"] = tool
    text = text_sample(delta)
    if text:
        summary["sample"] = text
    protocol_sample = protocol_value_sample({"delta": delta})
    if protocol_sample:
        summary["protocol_sample"] = protocol_sample
    return {key: value for key, value in summary.items() if value not in (None, "", [], {})}


def protocol_type_summary(value: Any) -> dict[str, str]:
    """Return compact protocol type fields from nested raw payload shapes."""
    if isinstance(value, list | tuple):
        for item in value:
            found = protocol_type_summary(item)
            if found:
                return found
        return {}
    if not isinstance(value, dict):
        return {}

    summary: dict[str, str] = {}
    event = value.get("event")
    if event:
        summary["protocol_event"] = str(event)
    delta = value.get("delta")
    if isinstance(delta, dict) and delta.get("type"):
        summary["delta_type"] = str(delta["type"])
    block = value.get("content_block")
    if isinstance(block, dict) and block.get("type"):
        summary["content_block_type"] = str(block["type"])
    return summary


def protocol_value_sample(value: Any) -> dict[str, Any]:
    """Return tiny structural samples for opaque protocol deltas/blocks."""
    if isinstance(value, list | tuple):
        for item in value:
            found = protocol_value_sample(item)
            if found:
                return found
        return {}
    if not isinstance(value, dict):
        return {}

    for key, label in (("delta", "delta"), ("content_block", "content_block")):
        nested = value.get(key)
        if isinstance(nested, dict):
            compact = compact_mapping_sample(nested)
            if compact:
                return {label: compact}
    for key in ("fields", "chunk", "message", "data", "tool_calls"):
        nested = value.get(key)
        if isinstance(nested, dict | list | tuple):
            found = protocol_value_sample(nested)
            if found:
                return found
    return {}


def compact_mapping_sample(mapping: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("type", "id", "name", "tool_name", "call_id", "tool_call_id", "index"):
        value = mapping.get(key)
        if value not in (None, ""):
            fields[key] = sample(value, limit=80)
    for key in ("args", "arguments", "input", "text", "content", "reasoning"):
        value = mapping.get(key)
        if value not in (None, ""):
            fields[key] = sample(value, limit=160)
    if not fields:
        keys = sorted(str(key) for key in mapping.keys())[:8]
        if keys:
            fields["keys"] = keys
    return fields


def compact_namespace(namespace: Any) -> str:
    if isinstance(namespace, list | tuple):
        return "/".join(str(part).split(":")[0] for part in namespace)
    return str(namespace or "")


def payload_kind(value: Any) -> str:
    if isinstance(value, list | tuple):
        if not value:
            return f"{type(value).__name__}:empty"
        return f"{type(value).__name__}:{payload_kind(value[0])}"
    if isinstance(value, dict):
        event = str(value.get("event") or value.get("type") or "")
        if event:
            return event
        return "dict"
    return type(value).__name__


def find_tool_payload(value: Any) -> dict[str, Any] | None:
    """Find compact tool-call fields in known raw provider/LangGraph shapes."""
    if isinstance(value, list | tuple):
        for item in value:
            found = find_tool_payload(item)
            if found:
                return found
        return None

    if not isinstance(value, dict):
        return None

    candidates = [value]
    for key in ("delta", "content_block", "fields", "chunk", "message", "data", "tool_calls"):
        nested = value.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
        elif isinstance(nested, list | tuple):
            found = find_tool_payload(nested)
            if found:
                return found
    for candidate in candidates:
        type_name = str(candidate.get("type") or candidate.get("event") or "")
        function = candidate.get("function") if isinstance(candidate.get("function"), dict) else {}
        name = candidate.get("name") or candidate.get("tool_name") or function.get("name") or ""
        args = first_present(candidate, "args", "arguments", "input")
        if args is None:
            args = first_present(function, "arguments", "args")
        call_id = candidate.get("id") or candidate.get("call_id") or candidate.get("tool_call_id") or ""
        if name or args is not None or "tool" in type_name or "function" in type_name:
            return {
                "type": type_name,
                "id": str(call_id),
                "name": str(name),
                "args_sample": sample(args, limit=120) if args is not None else "",
            }
    return None


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def text_sample(value: Any) -> str:
    """Return small text/reasoning samples without dumping full payloads."""
    if isinstance(value, list | tuple):
        for item in value:
            found = text_sample(item)
            if found:
                return found
        return ""
    if not isinstance(value, dict):
        return ""
    candidates = [value]
    for key in ("delta", "content_block", "fields", "chunk", "message", "data", "tool_calls"):
        nested = value.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
        elif isinstance(nested, list | tuple):
            for item in nested:
                if isinstance(item, dict):
                    candidates.append(item)
    for candidate in candidates:
        for key in ("text", "reasoning", "content"):
            raw = candidate.get(key)
            if isinstance(raw, str) and raw.strip():
                return sample(raw)
    return ""


def assert_streaming_shape(renderer: SmokeRenderer, strict_start_requests: bool) -> None:
    kinds = [event["kind"] for event in renderer.events]
    activity_indices = [
        index
        for index, kind in enumerate(kinds)
        if kind in {"delegation_started", "subagent_started", "subagent_request_updated", "subagent_finished"}
    ]
    if not activity_indices:
        raise AssertionError("No task/subagent activity was observed.")

    text_indices = [index for index, kind in enumerate(kinds) if kind == "text"]
    if text_indices and min(activity_indices) > min(text_indices):
        raise AssertionError("Task/subagent activity started only after assistant text.")

    started = [event for event in renderer.events if event["kind"] == "subagent_started"]
    if not started:
        raise AssertionError("No subagent_started events were observed.")
    missing_final_requests = [event["name"] for event in started if not renderer._subagent_requests.get(event["name"])]
    if missing_final_requests:
        raise AssertionError(f"Subagents still have blank requests: {missing_final_requests}")
    if strict_start_requests:
        blank_starts = [event["name"] for event in started if not event.get("request")]
        if blank_starts:
            raise AssertionError(f"subagent_started events had blank initial requests: {blank_starts}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live DeepAgents streaming smoke test.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--direct", action="store_true", help="Use direct local HTTP client settings from MIRA.")
    parser.add_argument(
        "--raw-diagnostic",
        action="store_true",
        help="Consume raw v3 protocol events and projections concurrently for timing diagnosis.",
    )
    parser.add_argument(
        "--sse-probe",
        action="store_true",
        help="Bypass LangChain and inspect the raw OpenAI-compatible LM Studio SSE stream.",
    )
    parser.add_argument(
        "--strict-start-requests",
        action="store_true",
        help="Fail if a subagent start is initially blank, even if it is patched later.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    print(f"[smoke] workspace={workspace}", flush=True)
    config = load_config(workspace)
    config["llm_direct"] = bool(args.direct)
    print(
        f"[smoke] provider={config.get('llm_provider')} model={config.get('llm_model')} direct={config.get('llm_direct')}",
        flush=True,
    )
    renderer = SmokeRenderer()
    if args.sse_probe:
        await asyncio.wait_for(run_openai_sse_probe(config, args.prompt, renderer), timeout=args.timeout)
        return 0

    print("[smoke] inferring model metadata...", flush=True)
    metadata = await infer_model_metadata(config)
    config["llm_inferred_context_tokens"] = metadata.context_tokens
    config["llm_context_source"] = metadata.context_source
    print(
        f"[smoke] metadata context={metadata.context_tokens} source={metadata.context_source}",
        flush=True,
    )
    print("[smoke] building agent...", flush=True)
    agent = build_agent(config=config, workspace=workspace, checkpointer=make_checkpointer(), metadata=metadata)
    print(f"[smoke] running prompt={args.prompt!r}", flush=True)
    if args.raw_diagnostic:
        await asyncio.wait_for(run_raw_diagnostic(agent, args.prompt, renderer), timeout=args.timeout)
        assert_streaming_shape(renderer, strict_start_requests=args.strict_start_requests)
    else:
        result = await asyncio.wait_for(
            run_turn(
                agent=agent,
                text=args.prompt,
                renderer=renderer,
                thread_id=f"stream-smoke-{uuid.uuid4()}",
            ),
            timeout=args.timeout,
        )
        renderer._record("done", final_chars=len(result.final_text or ""))
        assert_streaming_shape(renderer, strict_start_requests=args.strict_start_requests)
    return 0


async def run_raw_diagnostic(agent: Any, prompt: str, renderer: SmokeRenderer) -> None:
    """Run one prompt while recording raw v3 events and projection callbacks."""
    stream = await agent.astream_events(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": f"raw-stream-smoke-{uuid.uuid4()}"}},
        version="v3",
    )
    event_renderer = SubagentRequestRenderer(renderer)
    output: dict[str, Any] = {}
    await asyncio.gather(
        consume_raw_events(stream, renderer),
        consume_messages(stream.messages, event_renderer, render_normal_tools=False),
        consume_tool_calls(stream.tool_calls, event_renderer),
        consume_subagents(stream.subagents, event_renderer),
        capture_output(stream.output(), output),
    )
    renderer._record("done", final_chars=len(final_text(output.get("value"))))


async def run_openai_sse_probe(config: dict[str, Any], prompt: str, renderer: SmokeRenderer) -> None:
    """Inspect provider SSE chunks before LangChain parses them."""
    import httpx

    base_url = str(config.get("llm_base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("MIRA_LLM_BASE_URL / llm_base_url is required for --sse-probe.")
    url = urljoin(f"{base_url}/", "chat/completions")
    headers = {"Accept": "text/event-stream"}
    api_key = str(config.get("llm_api_key") or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": config["llm_model"],
        "messages": [{"role": "user", "content": prompt}],
        "tools": [task_tool_schema()],
        "tool_choice": "auto",
        "stream": True,
    }
    for source_key, target_key in (
        ("llm_temperature", "temperature"),
        ("llm_max_tokens", "max_tokens"),
        ("llm_top_p", "top_p"),
    ):
        value = config.get(source_key)
        if value is not None:
            payload[target_key] = value

    renderer._record("sse_probe_started", url=url, model=str(config["llm_model"]))
    saw_tool_args = False
    saw_tool_call = False
    async with httpx.AsyncClient(timeout=None, trust_env=not bool(config.get("llm_direct")), verify=False) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            renderer._record("sse_response", status=response.status_code)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    renderer._record("sse_done", saw_tool_call=saw_tool_call, saw_tool_args=saw_tool_args)
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    renderer._record("sse_chunk", payload_kind="invalid-json", sample=sample(data, limit=160))
                    continue
                summary = sse_chunk_summary(chunk)
                tool = summary.get("tool_like")
                if isinstance(tool, dict):
                    saw_tool_call = True
                    if tool.get("args_sample"):
                        saw_tool_args = True
                renderer._record("sse_chunk", **summary)


def task_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Delegate an independent task to a subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The complete request the subagent should perform.",
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "The subagent type to use.",
                    },
                },
                "required": ["description", "subagent_type"],
            },
        },
    }


async def consume_raw_events(stream: Any, renderer: SmokeRenderer) -> None:
    """Consume raw protocol events and record compact summaries."""
    async for event in stream:
        renderer.raw_event(event)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
