"""Bound access to MIRA's configured DeepAgents subagents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, TypeAlias

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_quickjs._subagent import call_subagent_task_tool

from agent.execution.mappings import ImmutableDict
from agent.execution.tools import BoundMiraTool, _payload_safe_config


class BoundMiraAgent(Runnable[str, Any]):
    """Thin runnable that delegates through MIRA's real DeepAgents task tool."""

    def __init__(self, name: str, task: BoundMiraTool) -> None:
        self.name = name
        self._task = task

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._task.invoke(
            {"description": str(input), "subagent_type": self.name},
            config=config,
            **kwargs,
        )

    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        runtime = self._task._runtime(config)
        execution_config = _payload_safe_config(runtime.config)
        runtime = replace(runtime, config=execution_config)
        task = self._task.tool.model_copy(
            update={
                "callbacks": execution_config.get("callbacks"),
                "tags": execution_config.get("tags"),
                "metadata": execution_config.get("metadata"),
            }
        )
        return await call_subagent_task_tool(
            task,
            description=str(input),
            subagent_type=self.name,
            response_schema=None,
            runtime=runtime,
            label=self.name,
        )


BoundAgentMap: TypeAlias = Mapping[str, BoundMiraAgent]


def bind_agents(task: BoundMiraTool | None, names: Iterable[str]) -> BoundAgentMap:
    """Expose only subagents available to the current task registry."""
    normalized = tuple(dict.fromkeys(str(name) for name in names if str(name)))
    if not normalized:
        return ImmutableDict()
    if task is None:
        raise RuntimeError(
            "MIRA has configured subagents but its compiled DeepAgents task tool "
            "is unavailable."
        )
    return ImmutableDict({name: BoundMiraAgent(name, task) for name in normalized})


__all__ = ["BoundAgentMap", "BoundMiraAgent", "bind_agents"]
