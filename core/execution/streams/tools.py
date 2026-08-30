"""Tool-call stream consumption and output normalization helpers."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict, deque
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from core.execution.streams.output import output_tool_lifecycle
from core.execution.streams.output import call_renderer
from core.execution.streams.tool_args import normalized_call
from core.context.usage import field

CONTROL_TOOLS = {
    "ask_user",
    "prepare_goal",
    "prepare_plan",
    "finalize_goal",
    "finalize_plan",
    "show_goal",
    "show_plan",
}
WATCHER_SHUTDOWN_SECONDS = 0.1
PREPARE_TOOL_COMPLETIONS = {
    "prepare_goal": "Success Criteria ready; finalizing Goal.",
    "prepare_plan": "Success Criteria ready; finalizing Plan.",
}


async def consume_tool_calls(tool_calls: Any, renderer: Any, result: Any | None = None) -> None:
    """Consume DeepAgents tool-call projections and render starts promptly."""
    watchers: set[asyncio.Task[None]] = set()
    try:
        async for call in tool_calls:
            normalized = normalized_call(call)
            identity = native_event_identity(call)
            name = str(normalized["name"])
            call_id = str(normalized.get("id") or "")
            is_new_call = True
            if result is not None:
                is_new_call = result.record_tool_call(name, call_id)

            if name == "task":
                if is_new_call:
                    call_renderer(renderer, "delegation_started", [normalized], **identity)
                continue

            if is_new_call:
                call_renderer(
                    renderer,
                    "tool_call",
                    name,
                    normalized.get("args", {}),
                    call_id=call_id,
                    **identity,
                )

            if not supports_completion_watch(call):
                continue
            watchers.add(
                asyncio.create_task(
                    watch_tool_result(
                        call,
                        name,
                        call_id,
                        renderer,
                        result,
                        identity=identity,
                    ),
                    name=f"mira-tool-result-{call_id or name}",
                )
            )
    except asyncio.CancelledError:
        await cancel_watchers(watchers)
        raise
    except BaseException:
        await finish_watchers(watchers)
        raise
    else:
        await finish_watchers(watchers)


def supports_completion_watch(call: Any) -> bool:
    """Return whether a call is terminal or exposes a supported completion stream."""
    if field(call, "completed") is not False:
        return True
    return field(call, "output_deltas") is not None or hasattr(call, "__aiter__")


async def watch_tool_result(
    call: Any,
    name: str,
    call_id: str,
    renderer: Any,
    result: Any | None,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    """Follow one call to completion and deliver a visible non-control result."""
    output, is_error, native_error = await tool_call_completion(call)
    if isinstance(output, Command):
        completion = PREPARE_TOOL_COMPLETIONS.get(name)
        if completion:
            render_tool_completion(
                renderer,
                result,
                name=name,
                text=completion,
                call_id=call_id,
                identity=identity,
            )
        return
    if name in CONTROL_TOOLS and not native_error:
        # Pinned LangGraph projects successful control-flow interrupts through
        # ToolCallStream.error, while some projections expose interrupt-shaped
        # success payloads. A native error ToolMessage is the only ordinary
        # completion a control tool renders through this path.
        return

    text = tool_output_text(output) or ("tool failed" if is_error else "")
    render_tool_completion(
        renderer,
        result,
        name=name,
        text=text,
        call_id=call_id,
        is_error=is_error,
        identity=identity,
    )


async def consume_live_tool_errors(events: Any, renderer: Any, result: Any | None = None) -> None:
    """Render newly added native error ToolMessages from root graph values."""
    if not hasattr(events, "__aiter__"):
        return

    seen: Counter[tuple[str, ...]] | None = None
    async for event in events:
        if not isinstance(event, dict) or event.get("method") != "values":
            continue
        params = event.get("params")
        if not isinstance(params, dict) or tuple(params.get("namespace") or ()):
            continue
        values = params.get("data")
        pairs = error_lifecycle_pairs(values)
        current = Counter(error_message_identity(error) for _, error in pairs)
        if seen is None:
            seen = current
            observe_baseline_errors(pairs, result)
            continue

        observed: Counter[tuple[str, ...]] = Counter()
        for call, error in pairs:
            identity = error_message_identity(error)
            observed[identity] += 1
            if observed[identity] <= seen[identity]:
                continue
            render_live_error_pair(
                call,
                error,
                renderer,
                result,
                occurrence=observed[identity],
            )

        for identity, count in current.items():
            seen[identity] = max(seen[identity], count)


def observe_baseline_errors(
    pairs: list[tuple[Any | None, dict[str, Any]]],
    result: Any | None,
) -> None:
    """Keep invocation-baseline errors out of both live and fallback rendering."""
    if result is None:
        return
    occurrences: Counter[tuple[str, ...]] = Counter()
    for call, error in pairs:
        identity = error_message_identity(error)
        occurrences[identity] += 1
        call_id = str(error.get("call_id") or "")
        observe_call = getattr(result, "observe_tool_call", None)
        if call is not None and callable(observe_call):
            observe_call(call_id)
        observe_result = getattr(result, "observe_tool_result", None)
        if callable(observe_result):
            observe_result(
                str(error.get("output") or "tool failed"),
                call_id,
                str(error.get("name") or "tool"),
                occurrence=occurrences[identity],
            )


def error_lifecycle_pairs(values: Any) -> list[tuple[Any | None, dict[str, Any]]]:
    """Pair error results with their preceding calls in one graph-state snapshot."""
    calls_by_id: dict[str, Any] = {}
    idless_calls: dict[str, deque[Any]] = defaultdict(deque)
    pairs: list[tuple[Any | None, dict[str, Any]]] = []
    for item in output_tool_lifecycle(values):
        if item.get("type") == "tool_call":
            call = item.get("call")
            normalized = normalized_call(call or {})
            call_id = str(normalized.get("id") or "")
            if call_id:
                calls_by_id[call_id] = call
            else:
                idless_calls[str(normalized.get("name") or "tool")].append(call)
            continue

        if item.get("status") != "error":
            continue
        call_id = str(item.get("call_id") or "")
        name = str(item.get("name") or "tool")
        call = calls_by_id.get(call_id) if call_id else None
        if call is None and not call_id and idless_calls[name]:
            call = idless_calls[name].popleft()
        pairs.append((call, item))
    return pairs


def error_message_identity(error: dict[str, Any]) -> tuple[str, ...]:
    """Return an occurrence-aware identity for one native error ToolMessage."""
    call_id = str(error.get("call_id") or "")
    if call_id:
        return ("id", call_id)
    return (
        "value",
        str(error.get("name") or "tool"),
        str(error.get("output") or ""),
    )


def render_live_error_pair(
    call: Any | None,
    error: dict[str, Any],
    renderer: Any,
    result: Any | None,
    *,
    occurrence: int,
) -> None:
    """Render a missing call start followed by its native live error."""
    name = str(error.get("name") or "tool")
    call_id = str(error.get("call_id") or "")
    identity = native_event_identity(call if call is not None else error)
    if call is not None:
        normalized = normalized_call(call)
        if result is None or result.record_tool_call(name, call_id):
            call_renderer(
                renderer,
                "tool_call",
                name,
                normalized.get("args", {}),
                call_id=call_id,
                **identity,
            )
    render_tool_completion(
        renderer,
        result,
        name=name,
        text=str(error.get("output") or "tool failed"),
        call_id=call_id,
        is_error=True,
        occurrence=occurrence if not call_id else None,
        identity=identity,
    )


def render_tool_completion(
    renderer: Any,
    result: Any | None,
    *,
    name: str,
    text: str,
    call_id: str = "",
    is_error: bool = False,
    recovered: bool = False,
    occurrence: int | None = None,
    identity: dict[str, Any] | None = None,
) -> bool:
    """Record and render one deduplicated live or recovered tool completion."""
    if result is not None and not result.record_tool_result(
        text,
        call_id,
        name,
        occurrence=occurrence,
    ):
        return False

    prefix = "recovered_" if recovered else "completed_"
    method = f"{prefix}tool_error" if is_error else f"{prefix}tool_result"
    callback = getattr(renderer, method, None)
    if not callable(callback) and is_error:
        callback = getattr(renderer, f"{prefix}tool_result", None)
    if callable(callback):
        call_renderer(
            renderer,
            method,
            name,
            text,
            call_id=call_id,
            **(identity or {}),
        )
    else:
        call_renderer(
            renderer,
            "tool_result",
            name,
            text,
            call_id=call_id,
            **(identity or {}),
        )
    return True


def native_event_identity(value: Any) -> dict[str, Any]:
    """Preserve native tool namespace and metadata when available."""
    if isinstance(value, dict):
        namespace = value.get("namespace") or value.get("path") or ()
        metadata = value.get("metadata") or {}
    else:
        namespace = getattr(value, "namespace", None) or getattr(value, "path", None) or ()
        metadata = getattr(value, "metadata", None) or {}
    return {
        "namespace": tuple(str(item) for item in namespace)
        if isinstance(namespace, (list, tuple))
        else (),
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }


async def finish_watchers(watchers: set[asyncio.Task[None]]) -> None:
    """Collect owned watchers, bounding cleanup if a provider never terminates one."""
    if not watchers:
        return
    done, pending = await asyncio.wait(watchers, timeout=WATCHER_SHUTDOWN_SECONDS)
    await asyncio.gather(*done, return_exceptions=True)
    await cancel_watchers(pending)


async def cancel_watchers(watchers: set[asyncio.Task[None]]) -> None:
    """Cancel and collect owned watchers without leaking task exceptions."""
    if not watchers:
        return
    for task in watchers:
        if not task.done():
            task.cancel()
    await asyncio.gather(*watchers, return_exceptions=True)


async def tool_call_completion(call: Any) -> tuple[Any, bool, bool]:
    """Return a tool call's final payload and whether it represents an error."""
    deltas: list[str] = []
    delta_error = False

    output_deltas = field(call, "output_deltas")
    if output_deltas is not None:
        async for delta in async_items(output_deltas):
            delta_error = delta_error or is_error_tool_message(delta)
            text = tool_output_text(delta)
            if text:
                deltas.append(text)
    elif hasattr(call, "__aiter__"):
        async for delta in call:
            delta_error = delta_error or is_error_tool_message(delta)
            text = tool_output_text(delta)
            if text:
                deltas.append(text)

    if isinstance(call, dict):
        if call.get("error") is not None:
            error = await maybe_await(call["error"])
            return error, True, is_error_tool_message(error)
        if call.get("output") is not None:
            output = await maybe_await(call["output"])
            native_error = is_error_tool_message(output)
            return output, native_error, native_error
        return "".join(deltas), delta_error, delta_error

    error = field(call, "error")
    if error is not None:
        error = await maybe_await(error)
        return error, True, is_error_tool_message(error)

    output = field(call, "output")
    if output is not None:
        output = await maybe_await(output)
        native_error = is_error_tool_message(output)
        return output, native_error, native_error

    return "".join(deltas), delta_error, delta_error


def is_error_tool_message(value: Any) -> bool:
    """Return whether a native ToolMessage carries error status."""
    is_tool = isinstance(value, ToolMessage) or field(value, "type") == "tool"
    return is_tool and str(field(value, "status") or "") == "error"


async def async_items(value: Any) -> Any:
    """Yield items from sync or async iterables, ignoring plain strings."""
    if hasattr(value, "__aiter__"):
        async for item in value:
            yield item
        return

    if isinstance(value, str):
        yield value
        return

    try:
        iterator = iter(value)
    except TypeError:
        if value is not None:
            yield value
        return

    for item in iterator:
        yield item


async def maybe_await(value: Any) -> Any:
    """Resolve awaitables while leaving plain values untouched."""
    return await value if hasattr(value, "__await__") else value


def tool_output_text(output: Any) -> str:
    """Convert a LangChain tool output object into displayable text."""
    if output is None:
        return ""

    content = field(output, "content")
    if content is not None:
        return str(content)

    return str(output)
