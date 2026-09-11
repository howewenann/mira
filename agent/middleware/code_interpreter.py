"""Narrow QuickJS compatibility bridge for inspectable Eval children."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

from langchain_quickjs import CodeInterpreterMiddleware


class _TaskToolWithCallbacks:
    """Delegate a QuickJS task call with its parent callbacks attached."""

    def __init__(self, tool: Any, callbacks: Any) -> None:
        self._tool = tool
        self._callbacks = callbacks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def arun(self, *args: Any, callbacks: Any = None, **kwargs: Any) -> Any:
        inherited = self._callbacks if callbacks is None else callbacks
        return await self._tool.arun(*args, callbacks=inherited, **kwargs)


def runtime_with_task_callbacks(runtime: Any) -> Any:
    """Carry the eval tool callback manager across QuickJS's worker loop."""
    config = getattr(runtime, "config", None)
    callbacks = config.get("callbacks") if isinstance(config, dict) else None
    tools = getattr(runtime, "tools", None)
    if callbacks is None or not tools:
        return runtime
    changed = False
    wrapped = []
    for tool in tools:
        if getattr(tool, "name", None) == "task":
            wrapped.append(_TaskToolWithCallbacks(tool, callbacks))
            changed = True
        else:
            wrapped.append(tool)
    return replace(runtime, tools=type(tools)(wrapped)) if changed else runtime


class InspectableCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """QuickJS middleware retaining native child streams for Eval tasks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        eval_tool = self.tools[0]
        original_coroutine = eval_tool.coroutine
        if original_coroutine is not None:

            @wraps(original_coroutine)
            async def async_eval(runtime: Any, code: str) -> Any:
                return await original_coroutine(
                    runtime=runtime_with_task_callbacks(runtime),
                    code=code,
                )

            eval_tool.coroutine = async_eval

        original_func = eval_tool.func
        if original_func is not None:

            @wraps(original_func)
            def sync_eval(runtime: Any, code: str) -> Any:
                return original_func(
                    runtime=runtime_with_task_callbacks(runtime),
                    code=code,
                )

            eval_tool.func = sync_eval


__all__ = ["InspectableCodeInterpreterMiddleware", "runtime_with_task_callbacks"]
