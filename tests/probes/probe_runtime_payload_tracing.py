"""Probe large runtime-tool payload visibility in MIRA's tracing path.

This probe answers four questions:

1. Does LangChain's tool callback lifecycle receive the full runtime payload?
2. Does MIRA's current LangSmith -> OTEL -> OpenInference export contain it?
3. Can an export-time processor redact the payload without removing the tool span?
4. Does export-time redaction leave the in-process callback payload unchanged?

No network traffic is allowed. LangSmith REST calls are patched out and OTEL
spans are exported only to an in-memory exporter.

Run from repository root:
    python tests/probes/probe_runtime_payload_tracing.py

Run from tests/probes:
    python probe_runtime_payload_tracing.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import langsmith
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import _get_all_injected_args
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tracing.semantic_processor import LangSmithOpenInferenceProcessor


BEGIN_MARKER = "MIRA_RUNTIME_PAYLOAD_BEGIN_8f6dd919"
END_MARKER = "MIRA_RUNTIME_PAYLOAD_END_b47ab012"
LARGE_PAYLOAD = (
    BEGIN_MARKER
    + "\n"
    + ("runtime-payload-line-0123456789abcdef\n" * 14000)
    + END_MARKER
)
assert len(LARGE_PAYLOAD) > 500_000

REDACT_TAG = "mira:runtime-payload-redact"
REDACT_METADATA_KEY = "mira_runtime_payload_policy"


class ProbeChatModel(BaseChatModel):
    """Minimal model required only to construct a DeepAgents graph."""

    @property
    def _llm_type(self) -> str:
        return "runtime-payload-probe"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="PROBE_MODEL_RESPONSE"))
            ]
        )

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "ProbeChatModel":
        del tools, tool_choice, kwargs
        return self


class CaptureToolInputs(AsyncCallbackHandler):
    """Record whether LangChain's normal tool callbacks see the large payload."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, parent_run_id, kwargs
        self.starts.append(
            {
                "name": serialized.get("name"),
                "input_str_chars": len(input_str),
                "input_str_has_begin": BEGIN_MARKER in input_str,
                "input_str_has_end": END_MARKER in input_str,
                "inputs_has_payload": _contains_markers(inputs),
            }
        )


class ProbeRuntimePayloadRedactor(SpanProcessor):
    """Probe-only exporter guard.

    This is NOT proposed production code. It proves that MIRA can mutate the
    finished LangSmith/OTEL span after OpenInference enrichment but before export
    while preserving the span itself.

    Only spans tagged by this probe are touched.
    """

    def on_end(self, span: Any) -> None:
        attributes = dict(span.attributes or {})
        tags = str(attributes.get("langsmith.span.tags", ""))
        metadata_policy = attributes.get(
            f"langsmith.metadata.{REDACT_METADATA_KEY}"
        )

        if REDACT_TAG not in tags and metadata_policy != "redact":
            return

        changed = False
        for key, value in tuple(attributes.items()):
            if _contains_markers(value):
                attributes[key] = _redacted_value(value)
                changed = True

        if changed:
            # Same frozen-span replacement technique MIRA's existing
            # LangSmithOpenInferenceProcessor already uses.
            span._attributes = attributes


def _redacted_value(value: Any) -> str:
    text = str(value)
    return f"<MIRA runtime payload redacted; exported_chars={len(text)}>"


def _contains_markers(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_contains_markers(k) or _contains_markers(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_markers(item) for item in value)
    text = str(value)
    return BEGIN_MARKER in text or END_MARKER in text


def _contains_full_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_contains_full_payload(k) or _contains_full_payload(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_full_payload(item) for item in value)
    text = str(value)
    return (
        BEGIN_MARKER in text
        and END_MARKER in text
        and len(text) >= len(LARGE_PAYLOAD)
    )


def _construct_runtime(runtime_cls: type[Any], values: dict[str, Any]) -> Any:
    signature = inspect.signature(runtime_cls)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            kwargs[name] = None
    return runtime_cls(**kwargs)


def make_runtime(
    *,
    config: dict[str, Any],
    tools: list[BaseTool],
    tool_call_id: str,
) -> Any:
    values = {
        "state": {"messages": []},
        "tool_call_id": tool_call_id,
        "config": config,
        "context": None,
        "store": None,
        "stream_writer": lambda event: None,
        "tools": tools,
        "execution_info": None,
        "server_info": None,
    }
    return _construct_runtime(ToolRuntime, values)


def inject_tool_args(
    tool: BaseTool,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    enriched = dict(payload)
    injected = _get_all_injected_args(tool)
    if not injected:
        return enriched

    derived = _construct_runtime(
        type(outer_runtime),
        {
            "state": outer_runtime.state,
            "tool_call_id": tool_call_id,
            "config": outer_runtime.config,
            "context": getattr(outer_runtime, "context", None),
            "store": getattr(outer_runtime, "store", None),
            "stream_writer": outer_runtime.stream_writer,
            "tools": outer_runtime.tools,
            "execution_info": getattr(outer_runtime, "execution_info", None),
            "server_info": getattr(outer_runtime, "server_info", None),
        },
    )

    if injected.runtime:
        enriched[injected.runtime] = derived

    if injected.state:
        for arg_name, state_field in injected.state.items():
            if state_field:
                state = outer_runtime.state
                enriched[arg_name] = (
                    state.get(state_field)
                    if isinstance(state, dict)
                    else getattr(state, state_field, None)
                )
            else:
                enriched[arg_name] = outer_runtime.state

    if injected.store and outer_runtime.store is not None:
        enriched[injected.store] = outer_runtime.store

    return enriched


async def call_actual_tool(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    runtime: Any,
) -> Any:
    call_id = f"runtime_payload_probe_{uuid.uuid4().hex[:8]}"
    injected_args = inject_tool_args(tool, args, runtime, call_id)

    callbacks = None
    tags = None
    metadata = None
    run_name = None
    if isinstance(runtime.config, dict):
        callbacks = runtime.config.get("callbacks")
        tags = runtime.config.get("tags")
        metadata = runtime.config.get("metadata")
        run_name = runtime.config.get("run_name")

    # BaseTool.arun() accepts callbacks/tags/metadata/run_name separately.
    # Passing only config= is not enough to attach tags/metadata to the tool's
    # own callback/LangSmith run. A production MIRA runtime executor should
    # preserve the complete tracing envelope.
    return await tool.arun(
        injected_args,
        callbacks=callbacks,
        tags=tags,
        metadata=metadata,
        run_name=run_name,
        config=runtime.config,
        tool_call_id=call_id,
    )


def eager_tool_registry(agent: Any) -> dict[str, BaseTool]:
    """Use the construction-time path proven by probe_eager_tool_registry.py."""
    try:
        tool_node = agent.nodes["tools"].bound
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "Expected eager tool registry at agent.nodes['tools'].bound"
        ) from exc

    if not isinstance(tool_node, ToolNode):
        raise RuntimeError(
            f"Expected ToolNode, found {type(tool_node).__name__}"
        )
    return dict(tool_node.tools_by_name)


def span_payload_hits(spans: list[Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for span in spans:
        for key, value in dict(span.attributes or {}).items():
            if not _contains_markers(value):
                continue
            hits.append(
                {
                    "span": span.name,
                    "kind": span.attributes.get("openinference.span.kind")
                    or span.attributes.get("langsmith.span.kind"),
                    "attribute": key,
                    "chars": len(str(value)),
                    "full_payload": _contains_full_payload(value),
                    "tags": span.attributes.get("langsmith.span.tags"),
                }
            )
    return hits


def span_summary(spans: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": span.name,
            "kind": span.attributes.get("openinference.span.kind")
            or span.attributes.get("langsmith.span.kind"),
            "parent": span.parent.span_id if span.parent else None,
            "tags": span.attributes.get("langsmith.span.tags"),
        }
        for span in spans
    ]


async def run_trace_case(
    *,
    write_tool: BaseTool,
    all_tools: list[BaseTool],
    workspace: Path,
    filename: str,
    redact_export: bool,
) -> tuple[list[Any], CaptureToolInputs, bool]:
    """Run one real write_file call through MIRA's tracing processor."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LangSmithOpenInferenceProcessor())
    if redact_export:
        provider.add_span_processor(ProbeRuntimePayloadRedactor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    capture = CaptureToolInputs()
    metadata = {
        "probe_case": "redacted" if redact_export else "control",
    }
    tags = ["runtime-payload-probe"]
    if redact_export:
        metadata[REDACT_METADATA_KEY] = "redact"
        tags.append(REDACT_TAG)

    config: dict[str, Any] = {
        "callbacks": [capture],
        "metadata": metadata,
        "tags": tags,
        "configurable": {
            "thread_id": f"runtime-payload-{metadata['probe_case']}",
        },
    }
    runtime = make_runtime(
        config=config,
        tools=all_tools,
        tool_call_id=f"outer_{metadata['probe_case']}",
    )

    client = None
    request_patch = patch.object(
        langsmith.Client,
        "request_with_retries",
        autospec=True,
    )
    rest_request = request_patch.start()

    try:
        client = langsmith.Client(
            tracing_mode="otel",
            otel_tracer_provider=provider,
        )
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_TRACING_MODE": "otel",
            },
        ):
            langsmith.configure(client=client, enabled=True)

            # Create a parent chain span so we also prove nested topology survives.
            async with langsmith.trace(
                "MIRA Runtime Composite Probe",
                run_type="chain",
                inputs={"input": f"write {filename}"},
                client=client,
            ) as parent:
                await call_actual_tool(
                    write_tool,
                    {
                        "file_path": f"/{filename}",
                        "content": LARGE_PAYLOAD,
                    },
                    runtime=runtime,
                )
                parent.set(outputs={"output": filename})
                parent.end()
    finally:
        langsmith.configure(client=None, enabled=None)
        if client is not None:
            client.close(timeout=1.0)
        request_patch.stop()
        provider.force_flush(timeout_millis=1_000)
        provider.shutdown()

    rest_request.assert_not_called()

    output_path = workspace / filename
    exact_file = (
        output_path.exists()
        and output_path.read_text(encoding="utf-8") == LARGE_PAYLOAD
    )

    return list(exporter.get_finished_spans()), capture, exact_file


def report(label: str, passed: bool, detail: Any = None) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")
    if detail is not None:
        print(detail)


async def main() -> int:
    print("=" * 88)
    print("MIRA runtime payload tracing probe")
    print("=" * 88)
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Probe cwd:       {Path.cwd()}")
    print(f"Payload chars:   {len(LARGE_PAYLOAD)}")

    with tempfile.TemporaryDirectory(prefix="mira-runtime-trace-") as temp_dir:
        workspace = Path(temp_dir)
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )

        agent = create_deep_agent(
            model=ProbeChatModel(),
            backend=backend,
        )
        registry = eager_tool_registry(agent)
        write_tool = registry["write_file"]
        all_tools = list(registry.values())

        print("\n" + "-" * 88)
        print("CASE 1: CURRENT MIRA TRACE PIPELINE")
        print("-" * 88)

        control_spans, control_callbacks, control_file_ok = await run_trace_case(
            write_tool=write_tool,
            all_tools=all_tools,
            workspace=workspace,
            filename="control.md",
            redact_export=False,
        )
        control_hits = span_payload_hits(control_spans)
        callback_control_leak = any(
            item["input_str_has_begin"]
            and item["input_str_has_end"]
            and item["inputs_has_payload"]
            for item in control_callbacks.starts
        )

        report(
            "Real DeepAgents write_file receives the full payload and writes it exactly",
            control_file_ok,
            f"path={workspace / 'control.md'}",
        )
        report(
            "LangChain on_tool_start currently receives the full runtime payload",
            callback_control_leak,
            control_callbacks.starts,
        )
        report(
            "Current MIRA-exported OTEL spans contain the runtime payload",
            bool(control_hits),
            control_hits if control_hits else span_summary(control_spans),
        )

        print("\n" + "-" * 88)
        print("CASE 2: PROBE-ONLY EXPORT REDACTION")
        print("-" * 88)

        redacted_spans, redacted_callbacks, redacted_file_ok = await run_trace_case(
            write_tool=write_tool,
            all_tools=all_tools,
            workspace=workspace,
            filename="redacted.md",
            redact_export=True,
        )
        redacted_hits = span_payload_hits(redacted_spans)
        callback_redacted_leak = any(
            item["input_str_has_begin"]
            and item["input_str_has_end"]
            and item["inputs_has_payload"]
            for item in redacted_callbacks.starts
        )
        redacted_tool_spans = [
            span
            for span in redacted_spans
            if span.name == "write_file"
            or span.attributes.get("tool.name") == "write_file"
        ]

        report(
            "Export redaction does not alter actual tool execution",
            redacted_file_ok,
            f"path={workspace / 'redacted.md'}",
        )
        report(
            "Probe-only processor removes the payload before OTEL export",
            not redacted_hits,
            redacted_hits if redacted_hits else "no payload markers in exported span attributes",
        )
        report(
            "Tool span survives export redaction",
            bool(redacted_tool_spans),
            span_summary(redacted_spans),
        )
        report(
            "Export-time redaction does NOT hide payload from in-process tool callbacks",
            callback_redacted_leak,
            redacted_callbacks.starts,
        )

        print("\n" + "=" * 88)
        print("INTERPRETATION")
        print("=" * 88)

        if control_hits:
            print(
                "CURRENT BEHAVIOR: the large runtime payload is exported in one or more "
                "LangSmith/OTEL span attributes."
            )
        else:
            print(
                "CURRENT BEHAVIOR: no payload marker reached the exported OTEL span "
                "attributes in this environment."
            )

        if callback_control_leak:
            print(
                "CALLBACK BEHAVIOR: normal LangChain tool-start callbacks receive the "
                "full tool arguments, including the large content value."
            )
        else:
            print(
                "CALLBACK BEHAVIOR: this environment did not expose the full content "
                "through the captured tool-start callback."
            )

        if not redacted_hits and redacted_tool_spans:
            print(
                "EXPORT REDACTION: feasible without deleting the tool span or changing "
                "the actual write result."
            )
        else:
            print(
                "EXPORT REDACTION: probe did not successfully preserve a clean tool span."
            )

        print(
            "IMPORTANT: export-time redaction and callback-input redaction are separate "
            "problems. If callbacks themselves must never observe large runtime payloads, "
            "the production runtime executor needs a source-level callback policy rather "
            "than only an OTEL span processor."
        )

    # This is a diagnostic probe, not a policy gate. Return failure only if the
    # actual write path or the redaction feasibility check broke.
    hard_failure = (
        not control_file_ok
        or not redacted_file_ok
        or bool(redacted_hits)
        or not redacted_tool_spans
    )

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    if hard_failure:
        print("FAILED: runtime write or export-redaction feasibility check failed")
        return 1

    print("PASS: payload visibility measured and export-redaction feasibility proven")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
