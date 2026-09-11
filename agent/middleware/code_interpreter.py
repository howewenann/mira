"""QuickJS wiring that preserves native nested-task streaming."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

from langchain_quickjs import CodeInterpreterMiddleware


class _TaskToolWithCallbacks:
    """Delegate a QuickJS task call with its parent callback stream attached."""

    def __init__(self, tool: Any, callbacks: Any) -> None:
        self._tool = tool
        self._callbacks = callbacks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def arun(self, *args: Any, callbacks: Any = None, **kwargs: Any) -> Any:
        inherited = self._callbacks if callbacks is None else callbacks
        return await self._tool.arun(*args, callbacks=inherited, **kwargs)


def _runtime_with_task_callbacks(runtime: Any) -> Any:
    """Carry the eval tool's callback manager across QuickJS's worker loop."""
    config = getattr(runtime, "config", None)
    callbacks = config.get("callbacks") if isinstance(config, dict) else None
    tools = getattr(runtime, "tools", None)
    if callbacks is None or not tools:
        return runtime

    changed = False
    wrapped_tools = []
    for tool in tools:
        if getattr(tool, "name", None) == "task":
            wrapped_tools.append(_TaskToolWithCallbacks(tool, callbacks))
            changed = True
        else:
            wrapped_tools.append(tool)
    if not changed:
        return runtime
    return replace(runtime, tools=type(tools)(wrapped_tools))


class MiraCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Use QuickJS while retaining messages/reasoning from eval-created children.

    QuickJS executes its JavaScript host bridge on a worker event loop. Its
    generic PTC bridge explicitly forwards callbacks, but its dedicated
    ``task()`` bridge currently does not. Wrapping only the task tool in the
    eval runtime restores DeepAgents' normal callback inheritance without
    replacing its task implementation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        eval_tool = self.tools[0]

        original_coroutine = eval_tool.coroutine
        if original_coroutine is not None:

            @wraps(original_coroutine)
            async def async_eval(runtime: Any, code: str) -> Any:
                return await original_coroutine(
                    runtime=_runtime_with_task_callbacks(runtime),
                    code=code,
                )

            eval_tool.coroutine = async_eval

        original_func = eval_tool.func
        if original_func is not None:

            @wraps(original_func)
            def sync_eval(runtime: Any, code: str) -> Any:
                return original_func(
                    runtime=_runtime_with_task_callbacks(runtime),
                    code=code,
                )

            eval_tool.func = sync_eval


__all__ = ["MiraCodeInterpreterMiddleware"]
