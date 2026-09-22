"""Bound invocation of MIRA's resolved LangChain tools."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, TypeAlias

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_quickjs._format import coerce_tool_output_for_ptc
from langgraph.config import get_config
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import _get_all_injected_args
from langgraph.runtime import get_runtime

from agent.execution.mappings import ImmutableDict

RUNTIME_PAYLOAD_TAG = "mira:execution-context-payload"


class BoundMiraTool(Runnable[dict[str, Any] | str, Any]):
    """Runnable view of one executable tool bound to the active graph runtime."""

    def __init__(self, tool: BaseTool, registry: Mapping[str, BaseTool]) -> None:
        self.tool = tool
        self._registry = registry

    @property
    def name(self) -> str:
        return self.tool.name

    def invoke(
        self,
        input: dict[str, Any] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        runtime = self._runtime(config)
        payload = _inject(self.tool, input, runtime)
        execution_config = _payload_safe_config(runtime.config)
        call_kwargs = {
            "callbacks": execution_config.get("callbacks"),
            "tags": execution_config.get("tags"),
            "metadata": execution_config.get("metadata"),
            "run_name": execution_config.get("run_name"),
            "config": execution_config,
            "tool_call_id": runtime.tool_call_id,
            **kwargs,
        }
        try:
            with _ignore_unparameterized_runtime_warning():
                result = self.tool.run(payload, **call_kwargs)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"MIRA tool {self.name!r} does not support synchronous execution; "
                "use ainvoke() instead."
            ) from exc
        return coerce_tool_output_for_ptc(result)

    async def ainvoke(
        self,
        input: dict[str, Any] | str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        runtime = self._runtime(config)
        payload = _inject(self.tool, input, runtime)
        execution_config = _payload_safe_config(runtime.config)
        call_kwargs = {
            "callbacks": execution_config.get("callbacks"),
            "tags": execution_config.get("tags"),
            "metadata": execution_config.get("metadata"),
            "run_name": execution_config.get("run_name"),
            "config": execution_config,
            "tool_call_id": runtime.tool_call_id,
            **kwargs,
        }
        with _ignore_unparameterized_runtime_warning():
            result = await self.tool.arun(payload, **call_kwargs)
        return coerce_tool_output_for_ptc(result)

    def _runtime(self, config: RunnableConfig | None) -> ToolRuntime[Any, Any]:
        from agent.execution.context import active_tool_runtime

        call_id = f"mira_context_{self.name}_{uuid.uuid4().hex[:12]}"
        current = active_tool_runtime()
        if current is not None:
            runtime = replace(
                current,
                config=config or current.config,
                tool_call_id=call_id,
                tools=list(self._registry.values()),
            )
        else:
            try:
                graph_runtime = get_runtime()
                ambient_config = get_config()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"MIRA tool {self.name!r} must be invoked from an active "
                    "LangGraph node or LangChain tool runtime."
                ) from exc
            runtime = ToolRuntime(
                state={},
                context=graph_runtime.context,
                config=config or ambient_config,
                stream_writer=graph_runtime.stream_writer,
                tool_call_id=call_id,
                store=graph_runtime.store,
                tools=list(self._registry.values()),
                execution_info=graph_runtime.execution_info,
                server_info=graph_runtime.server_info,
            )
        return _with_task_step_budget(runtime) if self.name == "task" else runtime


BoundToolMap: TypeAlias = Mapping[str, BoundMiraTool]


def bind_tools(registry: Mapping[str, BaseTool]) -> BoundToolMap:
    """Return immutable bound views over the exact executable registry."""
    return ImmutableDict(
        {name: BoundMiraTool(tool, registry) for name, tool in registry.items()}
    )


def _inject(
    tool: BaseTool,
    input: dict[str, Any] | str,
    runtime: ToolRuntime[Any, Any],
) -> dict[str, Any] | str:
    if not isinstance(input, dict):
        return input
    payload = dict(input)
    injected = _get_all_injected_args(tool)
    if injected.runtime:
        payload[injected.runtime] = runtime
    for argument, state_field in injected.state.items():
        if state_field:
            if isinstance(runtime.state, dict):
                payload[argument] = runtime.state.get(state_field)
            else:
                payload[argument] = getattr(runtime.state, state_field, None)
        else:
            payload[argument] = runtime.state
    if injected.store and runtime.store is not None:
        payload[injected.store] = runtime.store
    return payload


def _payload_safe_config(config: RunnableConfig) -> RunnableConfig:
    copied = dict(config or {})
    tags = list(copied.get("tags") or [])
    if RUNTIME_PAYLOAD_TAG not in tags:
        tags.append(RUNTIME_PAYLOAD_TAG)
    copied["tags"] = tags
    return copied


def _with_task_step_budget(runtime: ToolRuntime[Any, Any]) -> ToolRuntime[Any, Any]:
    """Supply DeepAgents' managed step value for domain-shaped callers."""
    if not isinstance(runtime.state, dict) or "remaining_steps" in runtime.state:
        return runtime
    return replace(
        runtime,
        state={
            **runtime.state,
            "remaining_steps": int(runtime.config.get("recursion_limit", 25)),
        },
    )


@contextmanager
def _ignore_unparameterized_runtime_warning():
    """Hide Pydantic's false warning for native, unparameterized ToolRuntime.

    LangChain supports ``runtime: ToolRuntime`` annotations, but Pydantic's
    callback serializer compares their populated context with a ``None`` schema
    and warns. Keep the compatibility shim confined to native tool boundaries.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)Pydantic serializer warnings:.*field_name='context'",
            category=UserWarning,
        )
        yield


__all__ = ["BoundMiraTool", "BoundToolMap", "RUNTIME_PAYLOAD_TAG", "bind_tools"]
