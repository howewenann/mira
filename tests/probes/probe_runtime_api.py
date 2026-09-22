"""
Proof-of-concept for a broader MIRA in-process runtime API.

Run from either the repository root:

    python tests/probes/probe_runtime_api.py

or directly from the probe directory:

    cd tests/probes
    python probe_runtime_api.py

This probe does not add a production runtime API. It proves the execution
mechanics MIRA would need before stabilizing one:

1. Capture DeepAgents' live model-request tool registry after built-in tools
   (filesystem/task/etc.) have been injected.
2. Build a probe runtime registry from those exact BaseTool instances.
3. Pass a large MCP-shaped result directly into the real DeepAgents write_file
   tool without placing that payload in parent messages.
4. Invoke a configured MIRA-compiled subagent through the real DeepAgents task
   tool programmatically.
5. Preserve LangChain tool callbacks and LangGraph custom-stream events.
6. Make the outer composite authorization scope explicit while confirming that
   direct runtime execution bypasses parent ToolNode HITL.
7. Confirm filesystem deny rules still apply to direct runtime calls.

Important:
- The MCP converter is intentionally local and deterministic. It has an MCP-like
  resolved name but does not require a real MCP server; this probe is about the
  runtime hand-off, not MCP transport correctness.
- The fake chat model is deterministic and performs no network/model call.
- This probe imports a private LangGraph injection helper and QuickJS subagent
  adapter on purpose to compare against mechanisms already proven by PTC. A
  production MiraRuntime should own a stable adapter rather than expose these
  private dependencies as public API.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langchain_quickjs._format import coerce_tool_output_for_ptc
from langchain_quickjs._subagent import call_subagent_task_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import _get_all_injected_args

# Load the compilation module by file path instead of importing
# agent.subagents.compilation through the package. Importing the package executes
# agent/subagents/__init__.py first, which currently imports discovery -> resources
# -> builder -> discovery and creates a circular import when this probe is run as a
# standalone script.
import importlib.util

_COMPILATION_PATH = REPOSITORY_ROOT / "agent" / "subagents" / "compilation.py"
_COMPILATION_SPEC = importlib.util.spec_from_file_location(
    "_mira_probe_subagent_compilation",
    _COMPILATION_PATH,
)
if _COMPILATION_SPEC is None or _COMPILATION_SPEC.loader is None:
    raise RuntimeError(f"Could not load MIRA subagent compilation module: {_COMPILATION_PATH}")

_compilation_module = importlib.util.module_from_spec(_COMPILATION_SPEC)
_COMPILATION_SPEC.loader.exec_module(_compilation_module)
compile_dynamic_subagents = _compilation_module.compile_dynamic_subagents


LARGE_MARKDOWN = "# Probe document\n\n" + ("MIRA_RUNTIME_PAYLOAD_7A9D\n" * 20_000)
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: Any = None) -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"\n[{marker}] {name}")
    if detail is not None:
        print(detail)
    if not condition:
        FAILURES.append(name)


class ProbeChatModel(BaseChatModel):
    """Tiny deterministic model that is sufficient for DeepAgents plumbing."""

    response: str = "MIRA_RUNTIME_SUBAGENT_RESPONSE"

    @property
    def _llm_type(self) -> str:
        return "mira-runtime-probe"

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
                ChatGeneration(
                    message=AIMessage(content=self.response),
                )
            ]
        )

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ProbeChatModel":
        del tools, tool_choice, kwargs
        return self


class CaptureLiveToolsMiddleware(AgentMiddleware):
    """Capture the exact tool objects visible at the model boundary."""

    def __init__(self) -> None:
        self.tools: list[BaseTool] = []

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        self.tools = list(request.tools)
        return await handler(request)


class ToolTraceRecorder(AsyncCallbackHandler):
    """Record only tool lifecycle names, never the large tool payloads."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []
        self.chains_started: list[dict[str, Any]] = []
        self.chat_models_started: list[dict[str, Any]] = []

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del input_str, run_id, parent_run_id, tags, metadata, inputs, kwargs
        self.started.append(str(serialized.get("name") or "<unknown>"))

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        del output, run_id, parent_run_id, kwargs
        # LangChain does not provide serialized tool metadata on tool end.
        self.ended.append("end")


    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del inputs, run_id, parent_run_id, tags
        self.chains_started.append(
            {
                "name": kwargs.get("name"),
                "serialized_name": (
                    serialized.get("name")
                    if isinstance(serialized, dict)
                    else None
                ),
                "metadata": dict(metadata or {}),
            }
        )


    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del messages, run_id, parent_run_id, tags
        self.chat_models_started.append(
            {
                "name": kwargs.get("name"),
                "serialized_name": serialized.get("name"),
                "metadata": dict(metadata or {}),
            }
        )


@tool("mcp__converter__pdf_to_markdown")
async def mcp_pdf_to_markdown(path: str, runtime: ToolRuntime) -> str:
    """Return deterministic Markdown as if produced by a resolved MCP tool."""
    runtime.stream_writer(
        {
            "type": "runtime_probe",
            "phase": "mcp_result_ready",
            "path": path,
            "chars": len(LARGE_MARKDOWN),
        }
    )
    return LARGE_MARKDOWN


def _construct_runtime(runtime_cls: type[Any], values: dict[str, Any]) -> Any:
    """Construct ToolRuntime across small LangGraph signature changes."""
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
    state: dict[str, Any],
    tool_call_id: str,
    config: dict[str, Any],
    stream_writer: Any,
    tools: list[BaseTool],
    template: Any | None = None,
) -> Any:
    """Create an execution runtime, preserving any newer optional fields."""
    runtime_cls = type(template) if template is not None else ToolRuntime
    values = {
        "state": state,
        "tool_call_id": tool_call_id,
        "config": config,
        "context": getattr(template, "context", None),
        "store": getattr(template, "store", None),
        "stream_writer": stream_writer,
        "tools": tools,
        "execution_info": getattr(template, "execution_info", None),
        "server_info": getattr(template, "server_info", None),
    }
    return _construct_runtime(runtime_cls, values)


def inject_tool_args(
    tool: BaseTool,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Mirror the ToolRuntime/state/store injection used by current QuickJS PTC."""
    enriched = dict(payload)
    injected = _get_all_injected_args(tool)
    if not injected or outer_runtime is None:
        return enriched

    derived = make_runtime(
        state=outer_runtime.state,
        tool_call_id=tool_call_id,
        config=outer_runtime.config,
        stream_writer=outer_runtime.stream_writer,
        tools=outer_runtime.tools,
        template=outer_runtime,
    )

    if injected.runtime:
        enriched[injected.runtime] = derived

    if injected.state:
        for arg_name, state_field in injected.state.items():
            if state_field:
                if isinstance(outer_runtime.state, dict):
                    enriched[arg_name] = outer_runtime.state.get(state_field)
                else:
                    enriched[arg_name] = getattr(
                        outer_runtime.state,
                        state_field,
                        None,
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
    """Invoke the actual resolved BaseTool using the active execution context."""
    call_id = f"runtime_probe_{tool.name}_{uuid.uuid4().hex[:8]}"
    injected_args = inject_tool_args(tool, args, runtime, call_id)

    callbacks = None
    if isinstance(runtime.config, dict):
        callbacks = runtime.config.get("callbacks")

    result = await tool.arun(
        injected_args,
        callbacks=callbacks,
        config=runtime.config,
        tool_call_id=call_id,
    )
    # BaseTool.arun() may return LangChain transport envelopes such as
    # ToolMessage or Command rather than the tool's native value. Current
    # QuickJS PTC already normalizes those envelopes before returning values
    # to programmatic callers, so the runtime proof must do the same.
    return coerce_tool_output_for_ptc(result)


@dataclass(frozen=True)
class ProbeAuthorizationScope:
    """Probe-only stand-in for an approved outer composite operation."""

    allowed_children: frozenset[str]

    def require(self, name: str) -> None:
        if name not in self.allowed_children:
            raise PermissionError(
                f"runtime child {name!r} is outside the approved composite scope"
            )


@dataclass
class ProbeExecution:
    """Minimal shape of the proposed bound MiraExecution."""

    tools: dict[str, BaseTool]
    runtime: Any
    authorization: ProbeAuthorizationScope

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.authorization.require(name)
        try:
            selected = self.tools[name]
        except KeyError as exc:
            raise KeyError(f"runtime tool not found: {name}") from exc
        return await call_actual_tool(selected, args, runtime=self.runtime)


async def main() -> int:
    print("=" * 88)
    print("MIRA runtime API proof")
    print("=" * 88)
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Probe cwd:       {Path.cwd()}")

    with tempfile.TemporaryDirectory(prefix="mira-runtime-probe-") as directory:
        workspace = Path(directory)
        (workspace / "input.pdf").write_bytes(b"%PDF-1.4\n% runtime probe\n")

        backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
        permissions = [
            FilesystemPermission(
                operations=["write"],
                paths=["/blocked/**"],
                mode="deny",
            ),
            FilesystemPermission(
                operations=["read", "write"],
                paths=["/**"],
                mode="allow",
            ),
        ]

        model = ProbeChatModel()
        compiled_subagents = compile_dynamic_subagents(
            [
                {
                    "name": "researcher",
                    "description": "Deterministic runtime probe subagent.",
                    "system_prompt": "Return the probe model response.",
                }
            ],
            model=model,
            tools=[mcp_pdf_to_markdown],
            backend=backend,
            skills=None,
            permissions=permissions,
            interrupt_on=None,
        )

        capture = CaptureLiveToolsMiddleware()
        agent = create_deep_agent(
            model=model,
            backend=backend,
            tools=[mcp_pdf_to_markdown],
            subagents=compiled_subagents,
            permissions=permissions,
            middleware=[capture],
            # Deliberately enabled. The direct runtime calls below should not
            # re-enter ToolNode/HITL for each child call.
            interrupt_on={"write_file": True},
            checkpointer=MemorySaver(),
        )

        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Return a short response. Do not call any tools. "
                            "This invocation only captures the live tool registry."
                        )
                    )
                ]
            },
            config={"configurable": {"thread_id": "runtime-probe-capture"}},
        )

        live_tools = list(capture.tools)
        registry = {tool.name: tool for tool in live_tools}

        check(
            "DeepAgents live model boundary includes injected write_file",
            "write_file" in registry,
            sorted(registry),
        )
        check(
            "DeepAgents live model boundary includes injected task",
            "task" in registry,
            sorted(registry),
        )
        check(
            "MCP-shaped tool is the same object passed into the agent",
            registry.get("mcp__converter__pdf_to_markdown")
            is mcp_pdf_to_markdown,
        )

        required = {
            "mcp__converter__pdf_to_markdown",
            "write_file",
            "task",
        }
        if not required.issubset(registry):
            print("\nCannot continue because required live tools were not captured.")
            return 1

        trace = ToolTraceRecorder()
        stream_events: list[Any] = []
        parent_messages = [
            HumanMessage(content="PARENT_CONTEXT_SENTINEL_DO_NOT_MUTATE")
        ]
        state = {"messages": parent_messages}
        config = {
            "callbacks": [trace],
            "configurable": {"thread_id": "runtime-probe-execution"},
        }
        outer_runtime = make_runtime(
            state=state,
            tool_call_id="runtime_probe_outer_composite",
            config=config,
            stream_writer=stream_events.append,
            tools=live_tools,
        )

        execution = ProbeExecution(
            tools=registry,
            runtime=outer_runtime,
            authorization=ProbeAuthorizationScope(
                allowed_children=frozenset(
                    {
                        "mcp__converter__pdf_to_markdown",
                        "write_file",
                    }
                )
            ),
        )

        markdown = await execution.call_tool(
            "mcp__converter__pdf_to_markdown",
            {"path": "/input.pdf"},
        )
        check(
            "MCP-shaped tool returns the deliberately large Markdown payload",
            markdown == LARGE_MARKDOWN,
            f"chars={len(markdown)}",
        )

        await execution.call_tool(
            "write_file",
            {
                "file_path": "/output.md",
                "content": markdown,
            },
        )

        output_path = workspace / "output.md"
        check(
            "Large MCP result passes directly into real DeepAgents write_file",
            output_path.exists()
            and output_path.read_text(encoding="utf-8") == LARGE_MARKDOWN,
            f"path={output_path}",
        )

        parent_text = "\n".join(str(message.content) for message in state["messages"])
        check(
            "Large intermediate payload never enters parent messages",
            LARGE_MARKDOWN not in parent_text
            and state["messages"] is parent_messages
            and len(state["messages"]) == 1,
            f"parent_messages={len(state['messages'])}",
        )

        check(
            "Direct runtime write bypasses parent ToolNode HITL intentionally",
            output_path.exists(),
            "agent was configured with interrupt_on={'write_file': True}",
        )

        blocked_path = workspace / "blocked" / "secret.md"
        blocked_error: str | None = None
        try:
            await call_actual_tool(
                registry["write_file"],
                {
                    "file_path": "/blocked/secret.md",
                    "content": "THIS MUST NOT BE WRITTEN",
                },
                runtime=outer_runtime,
            )
        except BaseException as exc:
            blocked_error = f"{type(exc).__name__}: {exc}"

        check(
            "Filesystem deny policy still applies to direct runtime calls",
            not blocked_path.exists(),
            blocked_error or "blocked write returned without creating a file",
        )

        scope_error: str | None = None
        try:
            await execution.call_tool(
                "read_file",
                {"file_path": "/output.md"},
            )
        except PermissionError as exc:
            scope_error = str(exc)

        check(
            "Outer composite authorization scope blocks undeclared child tools",
            scope_error is not None,
            scope_error,
        )

        task_tool = registry["task"]
        subagent_result = await call_subagent_task_tool(
            task_tool,
            description="Return the deterministic runtime probe response.",
            subagent_type="researcher",
            response_schema=None,
            runtime=outer_runtime,
            label="runtime probe",
        )

        check(
            "Configured MIRA-compiled subagent is callable through real task dispatch",
            "MIRA_RUNTIME_SUBAGENT_RESPONSE" in str(subagent_result),
            repr(subagent_result),
        )

        check(
            "Programmatic runtime tool calls preserve LangChain tool lifecycle callbacks",
            "mcp__converter__pdf_to_markdown" in trace.started
            and "write_file" in trace.started,
            f"tool_starts={trace.started}",
        )

        # The current QuickJS task adapter intentionally supplies `config=`
        # but does not explicitly forward `callbacks=` to BaseTool.arun().
        # For the proposed MIRA runtime API, invoke the exact same captured
        # DeepAgents task tool through the normal runtime bridge, which does
        # forward callbacks explicitly. This tests whether MIRA can preserve
        # tracing without constructing or reimplementing the subagent.
        model_events_before_runtime_task = len(trace.chat_models_started)
        runtime_task_result = await call_actual_tool(
            task_tool,
            {
                "description": "Return the deterministic runtime probe response.",
                "subagent_type": "researcher",
            },
            runtime=outer_runtime,
        )

        check(
            "Same captured task tool is callable through the generic runtime bridge",
            "MIRA_RUNTIME_SUBAGENT_RESPONSE" in str(runtime_task_result),
            repr(runtime_task_result),
        )

        check(
            "Runtime subagent dispatch preserves callback propagation into the subagent model",
            len(trace.chat_models_started) > model_events_before_runtime_task,
            (
                f"tool_starts={trace.started}\n"
                f"chain_starts={trace.chains_started}\n"
                f"chat_model_starts={trace.chat_models_started}"
            ),
        )

        check(
            "Runtime subagent dispatch also preserves task tool lifecycle callbacks",
            "task" in trace.started,
            f"tool_starts={trace.started}",
        )

        has_mcp_stream = any(
            isinstance(event, dict)
            and event.get("type") == "runtime_probe"
            and event.get("phase") == "mcp_result_ready"
            for event in stream_events
        )
        has_subagent_start = any(
            isinstance(event, dict)
            and event.get("type") == "subagent"
            and event.get("phase") == "start"
            for event in stream_events
        )
        has_subagent_complete = any(
            isinstance(event, dict)
            and event.get("type") == "subagent"
            and event.get("phase") == "complete"
            for event in stream_events
        )

        check(
            "Tool streaming and existing QuickJS subagent lifecycle stream events are preserved",
            has_mcp_stream and has_subagent_start and has_subagent_complete,
            stream_events,
        )

        check(
            "Runtime registry uses the exact live tool objects seen by the agent",
            execution.tools["write_file"] is registry["write_file"]
            and execution.tools["task"] is registry["task"]
            and execution.tools["mcp__converter__pdf_to_markdown"]
            is registry["mcp__converter__pdf_to_markdown"],
        )

        parent_text_after_subagent = "\n".join(
            str(message.content) for message in state["messages"]
        )
        check(
            "Programmatic subagent dispatch does not mutate parent messages in-place",
            parent_text_after_subagent == "PARENT_CONTEXT_SENTINEL_DO_NOT_MUTATE"
            and len(state["messages"]) == 1,
            f"parent_messages={state['messages']!r}",
        )

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("PASS: all runtime API proof checks succeeded")
    print()
    print("This proves the mechanics, not the final public API.")
    print("Production code still needs a stable MIRA-owned adapter for runtime")
    print("injection, authorization scopes, observability payload policy, and")
    print("construction-time/live-registry hand-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
