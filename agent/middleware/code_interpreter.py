"""QuickJS wiring that exposes Eval child streams to the Inspector."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from functools import wraps
from typing import Any

from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.errors import GraphInterrupt


EVAL_SUBAGENT_ROW_METADATA = "mira_eval_subagent_row_id"


class _EvalTaskIdentities:
    """Map one replayed Eval body's task calls onto stable logical rows."""

    def __init__(self) -> None:
        self._occurrences: dict[str, int] = defaultdict(int)
        self._stable_by_raw: dict[str, str] = {}

    def rewrite_event(self, event: Any) -> Any:
        if not isinstance(event, dict) or event.get("type") != "subagent":
            return event
        raw_id = str(event.get("id") or "")
        if not raw_id:
            return event
        phase = str(event.get("phase") or "")
        if phase == "start":
            fingerprint = json.dumps(
                {
                    "eval_id": str(event.get("eval_id") or ""),
                    "subagent_type": str(event.get("subagent_type") or ""),
                    "description": str(event.get("description") or ""),
                    "label": str(event.get("label") or ""),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            occurrence = self._occurrences[fingerprint]
            self._occurrences[fingerprint] += 1
            digest = hashlib.sha256(
                f"{fingerprint}\0{occurrence}".encode("utf-8")
            ).hexdigest()[:16]
            self._stable_by_raw[raw_id] = f"ptc_task_{digest}"
        stable_id = self._stable_by_raw.get(raw_id)
        if not stable_id:
            return event
        return {**event, "id": stable_id}

    def stable_id(self, raw_id: str) -> str:
        return self._stable_by_raw.get(raw_id, raw_id)


class _TaskToolWithCallbacks:
    """Delegate QuickJS ``task()`` with the Eval call's callback manager."""

    def __init__(
        self,
        tool: Any,
        callbacks: Any,
        identities: _EvalTaskIdentities,
    ) -> None:
        self._tool = tool
        self._callbacks = callbacks
        self._identities = identities

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def arun(self, *args: Any, callbacks: Any = None, **kwargs: Any) -> Any:
        inherited = self._callbacks if callbacks is None else callbacks
        raw_id = str(kwargs.get("tool_call_id") or "")
        row_id = self._identities.stable_id(raw_id)
        if row_id:
            kwargs["tool_call_id"] = row_id
            args = _runtime_args_with_row_id(args, row_id)
        config = kwargs.get("config")
        if row_id and isinstance(config, dict):
            config = dict(config)
            metadata = dict(config.get("metadata") or {})
            metadata[EVAL_SUBAGENT_ROW_METADATA] = row_id
            config["metadata"] = metadata
            kwargs["config"] = config
        return await self._tool.arun(*args, callbacks=inherited, **kwargs)


def _runtime_args_with_row_id(args: tuple[Any, ...], row_id: str) -> tuple[Any, ...]:
    """Replace the task bridge's injected runtime with its logical identity."""
    if not args or not isinstance(args[0], dict):
        return args
    payload = dict(args[0])
    runtime = payload.get("runtime")
    if runtime is None:
        return args
    payload["runtime"] = replace(runtime, tool_call_id=row_id)
    return (payload, *args[1:])


def runtime_with_task_callbacks(runtime: Any) -> Any:
    """Carry callbacks across QuickJS's dedicated task bridge.

    The generic QuickJS tool bridge forwards callbacks explicitly. Its
    dedicated ``task()`` bridge currently does not, so namespaced model deltas
    from Eval-created children otherwise never reach LangGraph's v3 stream.
    """
    config = getattr(runtime, "config", None)
    callbacks = config.get("callbacks") if isinstance(config, dict) else None
    tools = getattr(runtime, "tools", None)
    if not tools:
        return runtime

    identities = _EvalTaskIdentities()
    changed = False
    wrapped = []
    for tool in tools:
        if getattr(tool, "name", None) == "task":
            wrapped.append(_TaskToolWithCallbacks(tool, callbacks, identities))
            changed = True
        else:
            wrapped.append(tool)
    if not changed:
        return runtime

    updates: dict[str, Any] = {"tools": type(tools)(wrapped)}
    stream_writer = getattr(runtime, "stream_writer", None)
    if callable(stream_writer):

        def write(event: Any) -> Any:
            return stream_writer(identities.rewrite_event(event))

        updates["stream_writer"] = write
    return replace(runtime, **updates)


class InspectableCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Expose Eval child callbacks without replacing QuickJS task execution."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        eval_tool = self.tools[0]

        original_coroutine = eval_tool.coroutine
        if original_coroutine is not None:

            @wraps(original_coroutine)
            async def async_eval(runtime: Any, code: str) -> Any:
                snapshot: bytes | None = None
                repl = None
                if self._mode == "thread":
                    repl = self._repl_for_eval(self._slot_id(runtime.state))
                    snapshot = await repl.acreate_snapshot()
                try:
                    return await original_coroutine(
                        runtime=runtime_with_task_callbacks(runtime),
                        code=code,
                    )
                except GraphInterrupt:
                    if repl is not None and snapshot is not None:
                        try:
                            await repl.arestore_snapshot(snapshot, inject_globals=True)
                        except Exception as exc:
                            raise RuntimeError(
                                "QuickJS state rollback failed after an interrupted Eval"
                            ) from exc
                    raise

            eval_tool.coroutine = async_eval

        original_func = eval_tool.func
        if original_func is not None:

            @wraps(original_func)
            def sync_eval(runtime: Any, code: str) -> Any:
                snapshot: bytes | None = None
                repl = None
                if self._mode == "thread":
                    repl = self._repl_for_eval(self._slot_id(runtime.state))
                    snapshot = repl.create_snapshot()
                try:
                    return original_func(
                        runtime=runtime_with_task_callbacks(runtime),
                        code=code,
                    )
                except GraphInterrupt:
                    if repl is not None and snapshot is not None:
                        try:
                            repl.restore_snapshot(snapshot, inject_globals=True)
                        except Exception as exc:
                            raise RuntimeError(
                                "QuickJS state rollback failed after an interrupted Eval"
                            ) from exc
                    raise

            eval_tool.func = sync_eval


__all__ = [
    "EVAL_SUBAGENT_ROW_METADATA",
    "InspectableCodeInterpreterMiddleware",
    "runtime_with_task_callbacks",
]
