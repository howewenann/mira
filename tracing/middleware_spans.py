"""LangChain/LangGraph compatibility policy for AgentMiddleware spans.

This module is an intentional quarantine boundary around private and
semi-private framework tracing internals. Knowledge of LangChain's middleware
wrappers, LangGraph graph compilation, Pregel nodes, and RunnableSeq execution
must remain here so framework upgrades have one explicit compatibility surface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import RLock
from typing import Any, Literal

from langchain.agents.middleware.types import AgentMiddleware, TracePolicy, omit_payload
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langgraph._internal._runnable import (
    ASYNCIO_ACCEPTS_CONTEXT,
    RunnableCallable,
    RunnableSeq,
    set_config_context,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel._read import PregelNode

MiddlewareSpanMode = Literal["hidden", "full"]

# DeepAgents 0.7.9 omits middleware hook inputs from traces. MIRA keeps those
# inputs visible by default for developer diagnostics. Flip this internal
# compatibility toggle to False to retain the upstream omission behavior.
TRACE_MIDDLEWARE_INPUTS = True

_HOOK_NAMES = ("before_agent", "before_model", "after_model", "after_agent")
_CONSTRUCTION_LOCK = RLock()


class MiddlewareSpanCompatibilityError(RuntimeError):
    """Report a framework shape that MIRA cannot safely suppress."""


class _NonTracingMiddlewareNodeSeq(RunnableSeq):
    """Execute one middleware node sequence without its outer callback run."""

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        config = ensure_config() if config is None else config
        for index, step in enumerate(self.steps):
            if index == 0:
                with set_config_context(config) as context:
                    input = context.run(step.invoke, input, config, **kwargs)
            else:
                input = step.invoke(input, config)
        return input

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        config = ensure_config() if config is None else config
        for index, step in enumerate(self.steps):
            if index == 0:
                if ASYNCIO_ACCEPTS_CONTEXT:
                    with set_config_context(config) as context:
                        input = await asyncio.create_task(
                            step.ainvoke(input, config, **kwargs),
                            context=context,
                        )
                else:
                    input = await step.ainvoke(input, config, **kwargs)
            else:
                input = await step.ainvoke(input, config)
        return input


def _identity_traceable(*args: Any, **kwargs: Any) -> Any:
    """Mirror traceable's decorator forms while leaving callables unchanged."""
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        return function

    return decorate


def _trace_policy_with_inputs(policy: TracePolicy | None) -> TracePolicy | None:
    """Remove only DeepAgents' input-omission transform from a trace policy."""
    if policy is None or policy.process_inputs is not omit_payload:
        return policy
    return TracePolicy(process_outputs=policy.process_outputs)


def _middleware_hook(node_name: object, action: object) -> str | None:
    """Structurally identify a compiled AgentMiddleware before/after hook."""
    if not isinstance(node_name, str) or not isinstance(action, RunnableCallable):
        return None
    for function in (action.func, action.afunc):
        middleware = getattr(function, "__self__", None)
        method_name = getattr(function, "__name__", None)
        if not isinstance(middleware, AgentMiddleware):
            continue
        for hook_name in _HOOK_NAMES:
            expected_name = f"{middleware.name}.{hook_name}"
            if method_name not in {hook_name, f"a{hook_name}"}:
                continue
            if node_name != expected_name:
                raise MiddlewareSpanCompatibilityError(
                    f"MIRA tracing compatibility expected middleware hook {method_name!r} "
                    f"to compile as {expected_name!r}, got {node_name!r}"
                )
            if action.trace is not False:
                raise MiddlewareSpanCompatibilityError(
                    f"MIRA tracing compatibility expected {node_name!r} to use trace=False"
                )
            return node_name
    return None


def _replace_node_sequences(nodes: list[tuple[PregelNode, str]]) -> None:
    """Replace cached outer sequences after compilation attaches all writers."""
    for pregel_node, node_name in nodes:
        pregel_node.__dict__.pop("node", None)
        sequence = pregel_node.node
        if not isinstance(sequence, RunnableSeq) or len(sequence.steps) < 2:
            raise MiddlewareSpanCompatibilityError(
                f"MIRA tracing compatibility expected middleware node {node_name!r} "
                "to compile to RunnableSeq(bound, writers...)"
            )
        if sequence.steps[0] is not pregel_node.bound:
            raise MiddlewareSpanCompatibilityError(
                f"MIRA tracing compatibility found an unexpected first step for {node_name!r}"
            )
        if not hasattr(sequence, "name") or not hasattr(sequence, "trace_inputs"):
            raise MiddlewareSpanCompatibilityError(
                f"MIRA tracing compatibility cannot preserve RunnableSeq metadata for {node_name!r}"
            )
        pregel_node.__dict__["node"] = _NonTracingMiddlewareNodeSeq(
            *sequence.steps,
            name=sequence.name,
            trace_inputs=sequence.trace_inputs,
        )


@contextmanager
def middleware_span_policy(mode: MiddlewareSpanMode) -> Generator[None, None, None]:
    """Apply the selected AgentMiddleware tracing policy during graph construction."""
    if mode not in {"hidden", "full"}:
        raise ValueError(f"unsupported middleware span mode: {mode!r}")
    if mode == "full" and not TRACE_MIDDLEWARE_INPUTS:
        yield
        return

    import langchain.agents.factory as langchain_factory

    with _CONSTRUCTION_LOCK:
        original_traceable = getattr(langchain_factory, "traceable", None)
        original_attach_node = getattr(CompiledStateGraph, "attach_node", None)
        original_resolved_transform = getattr(langchain_factory, "_resolved_transform", None)
        original_node_trace_policy = getattr(langchain_factory, "_node_trace_policy", None)
        if mode == "hidden" and not callable(original_traceable):
            raise MiddlewareSpanCompatibilityError(
                "MIRA tracing compatibility requires langchain.agents.factory.traceable"
            )
        if mode == "hidden" and not callable(original_attach_node):
            raise MiddlewareSpanCompatibilityError(
                "MIRA tracing compatibility requires CompiledStateGraph.attach_node"
            )
        if TRACE_MIDDLEWARE_INPUTS and not callable(original_resolved_transform):
            raise MiddlewareSpanCompatibilityError(
                "MIRA tracing compatibility requires langchain.agents.factory._resolved_transform"
            )
        if TRACE_MIDDLEWARE_INPUTS and not callable(original_node_trace_policy):
            raise MiddlewareSpanCompatibilityError(
                "MIRA tracing compatibility requires langchain.agents.factory._node_trace_policy"
            )

        middleware_nodes: list[tuple[PregelNode, str]] = []

        def preserve_input_transform(
            policy: TracePolicy | None,
            field: Literal["process_inputs", "process_outputs"],
        ) -> Callable[[Any], Any]:
            effective = _trace_policy_with_inputs(policy) if field == "process_inputs" else policy
            return original_resolved_transform(effective, field)

        def preserve_node_inputs(policy: TracePolicy | None) -> TracePolicy:
            return original_node_trace_policy(_trace_policy_with_inputs(policy))

        def capture_attach_node(graph: CompiledStateGraph, key: str, node: Any) -> None:
            action = getattr(node, "runnable", None)
            hook_name = _middleware_hook(key, action)
            original_attach_node(graph, key, node)
            if hook_name is None:
                return
            compiled_node = graph.nodes.get(key)
            if not isinstance(compiled_node, PregelNode):
                raise MiddlewareSpanCompatibilityError(
                    f"MIRA tracing compatibility expected PregelNode for {hook_name!r}"
                )
            middleware_nodes.append((compiled_node, hook_name))

        if TRACE_MIDDLEWARE_INPUTS:
            langchain_factory._resolved_transform = preserve_input_transform
            langchain_factory._node_trace_policy = preserve_node_inputs
        if mode == "hidden":
            langchain_factory.traceable = _identity_traceable
            CompiledStateGraph.attach_node = capture_attach_node
        try:
            yield
            if mode == "hidden":
                _replace_node_sequences(middleware_nodes)
        finally:
            if mode == "hidden":
                CompiledStateGraph.attach_node = original_attach_node
                langchain_factory.traceable = original_traceable
            if TRACE_MIDDLEWARE_INPUTS:
                langchain_factory._node_trace_policy = original_node_trace_policy
                langchain_factory._resolved_transform = original_resolved_transform
