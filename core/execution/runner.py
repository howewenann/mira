"""Top-level orchestration for one streamed native agent turn."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any

from langgraph.types import Command
from langgraph.stream.transformers import CustomTransformer

from agent.middleware.correction import CORRECTION_EVENT
from agent.planning.policy import PLANNING_STAGES
from core.execution.streams.corrections import normalize_correction_event
from core.execution.streams.messages import consume_messages
from core.execution.streams.message_metadata import MessageInvocationMetadata, MessageInvocationMetadataTransformer
from core.execution.streams.output import (
    capture_output,
    collect_interrupts,
    final_text,
    output_has_tool_call_repr,
    output_tool_calls,
    output_tool_lifecycle,
)
from core.execution.streams.rubric import RubricEventRenderer
from core.execution.streams.subagents import DYNAMIC_TOOL_SUBAGENT, EVAL_SUBAGENT, consume_subagents
from core.execution.streams.tool_args import normalized_call, tool_call_args
from core.execution.streams.tools import (
    CONTROL_TOOLS,
    consume_live_tool_errors,
    consume_tool_calls,
    render_tool_completion,
)

from core.context.usage import (
    empty_usage,
    has_context_usage,
    has_usage,
    item_context_source,
    merge_usage,
    positive_int,
    select_context_usage,
    usage_from_output,
)
from core.interface.requests import APPROVAL_CONSEQUENCE


@dataclass
class TurnResult:
    """Summary of one agent turn used by REPL planning logic."""

    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0
    context_source: str = "unknown"
    usage_source: str = "unknown"
    rubric_status: str = ""
    rubric_evaluations: list[dict[str, Any]] = field(default_factory=list)
    formal_review: dict[str, Any] | None = None
    _stream_usage: dict[str, Any] = field(default_factory=empty_usage, repr=False)
    _tool_call_drafts: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _tool_call_ids: list[str] = field(default_factory=list, repr=False)
    _seen_tool_call_ids: set[str] = field(default_factory=set, repr=False)
    _seen_tool_result_ids: set[str] = field(default_factory=set, repr=False)
    _seen_tool_result_values: set[tuple[str, str]] = field(default_factory=set, repr=False)
    _seen_tool_result_occurrences: Counter[tuple[str, str]] = field(
        default_factory=Counter,
        repr=False,
    )

    @property
    def usage(self) -> dict[str, Any]:
        """Return normalized token usage for this turn."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_tokens": self.context_tokens,
            "context_source": self.context_source,
            "source": self.usage_source,
        }

    def add_usage(self, usage: dict[str, Any]) -> None:
        """Add one usage object to the persisted turn totals."""
        input_tokens = positive_int(usage.get("input_tokens"))
        output_tokens = positive_int(usage.get("output_tokens"))
        total_tokens = positive_int(usage.get("total_tokens")) or input_tokens + output_tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.set_context_usage(select_context_usage(usage))
        if self.usage_source == "unknown" and usage.get("source"):
            self.usage_source = str(usage["source"])

    def add_stream_usage(self, usage: dict[str, Any]) -> None:
        """Capture streamed usage as a fallback for providers that omit final usage."""
        self._stream_usage = merge_usage(self._stream_usage, usage)

    def set_context_usage(self, usage: dict[str, Any]) -> None:
        """Set current context usage without changing cumulative In/Out totals."""
        if not has_context_usage(usage):
            return
        selected = select_context_usage(usage)
        self.context_tokens = positive_int(selected.get("context_tokens"))
        self.context_source = item_context_source(selected)
        if self.usage_source == "unknown" and self.context_source != "unknown":
            self.usage_source = self.context_source

    def commit_loop_usage(self, output: Any) -> dict[str, Any]:
        """Commit LangChain token usage and return the per-loop usage delta."""
        committed = empty_usage()
        output_usage = usage_from_output(output)
        if has_usage(output_usage):
            self.add_usage(output_usage)
            committed = select_context_usage(output_usage)
        elif has_usage(self._stream_usage):
            self.add_usage(self._stream_usage)
            committed = select_context_usage(self._stream_usage)
        self._stream_usage = empty_usage()
        return committed

    def record_tool_call(self, name: str, call_id: str = "") -> bool:
        """Record one tool call while avoiding duplicate id-based reports."""
        if call_id:
            if call_id in self._seen_tool_call_ids:
                return False
            self._seen_tool_call_ids.add(call_id)
        self.tool_calls.append(name)
        self._tool_call_ids.append(call_id)
        return True

    def update_tool_call_name(self, index: int, name: str) -> None:
        """Keep the turn summary aligned with an edited approval action."""
        if 0 <= index < len(self.tool_calls):
            self.tool_calls[index] = name

    def observe_tool_call(self, call_id: str = "") -> None:
        """Mark a baseline call as historical without adding it to this turn."""
        if call_id:
            self._seen_tool_call_ids.add(call_id)

    def record_tool_call_draft(self, name: str, call_id: str) -> None:
        """Remember a provider draft so fallback promotion keeps its identifier."""
        self._tool_call_drafts.append((name, call_id))

    def tool_call_since(self, name: str, start: int) -> tuple[bool, str]:
        """Return whether this loop recorded a named call and its identifier."""
        for index in range(len(self.tool_calls) - 1, start - 1, -1):
            if self.tool_calls[index] == name:
                return True, self._tool_call_ids[index]
        return False, ""

    def tool_call_draft_since(self, name: str, start: int) -> str:
        """Return the newest identifier for a named draft in the current loop."""
        for draft_name, call_id in reversed(self._tool_call_drafts[start:]):
            if draft_name == name:
                return call_id
        return ""

    def record_tool_result(
        self,
        text: str,
        call_id: str = "",
        name: str = "",
        *,
        occurrence: int | None = None,
    ) -> bool:
        """Record one tool result while avoiding duplicate id-based reports."""
        value_key = (name, text)
        if call_id:
            if call_id in self._seen_tool_result_ids:
                return False
            self._seen_tool_result_ids.add(call_id)
        elif occurrence is not None:
            if occurrence <= self._seen_tool_result_occurrences[value_key]:
                return False
            self._seen_tool_result_occurrences[value_key] = occurrence
        elif value_key in self._seen_tool_result_values:
            return False
        else:
            self._seen_tool_result_occurrences[value_key] = max(
                self._seen_tool_result_occurrences[value_key],
                1,
            )
        self._seen_tool_result_values.add(value_key)
        self.tool_results.append(text)
        return True

    def observe_tool_result(
        self,
        text: str,
        call_id: str = "",
        name: str = "",
        *,
        occurrence: int = 1,
    ) -> None:
        """Mark a baseline result as historical without adding it to this turn."""
        value_key = (name, text)
        if call_id:
            self._seen_tool_result_ids.add(call_id)
        else:
            self._seen_tool_result_values.add(value_key)
            self._seen_tool_result_occurrences[value_key] = max(
                self._seen_tool_result_occurrences[value_key],
                occurrence,
            )


class SubagentRequestRenderer:
    """Fill empty subagent request text from preceding task delegations."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self._pending_requests: deque[str] = deque()
        self._pending_subagents: deque[str] = deque()
        self._hidden_subagents: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        for description in task_descriptions(calls):
            if self._pending_subagents:
                callback = getattr(self.renderer, "subagent_request_updated", None)
                if callable(callback):
                    callback(self._pending_subagents.popleft(), description)
            else:
                self._pending_requests.append(description)
        self.renderer.delegation_started(calls)

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
    ) -> None:
        queued_request = self._pending_requests.popleft() if self._pending_requests else ""
        request = task_input or queued_request
        if origin == DYNAMIC_TOOL_SUBAGENT and not request:
            self._hidden_subagents.add(subagent)
            return
        display_origin = "" if queued_request else origin
        call_renderer_with_supported_kwargs(
            self.renderer.subagent_started,
            subagent,
            request,
            origin=display_origin,
            eval_id=eval_id,
            row_id=row_id,
            model=model,
        )
        if not request:
            self._pending_subagents.append(subagent)

    def subagent_finished(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        if subagent in self._hidden_subagents:
            self._hidden_subagents.remove(subagent)
            return
        call_renderer_with_supported_kwargs(
            self.renderer.subagent_finished,
            subagent,
            result,
            eval_id=eval_id,
            row_id=row_id,
            duration_ms=duration_ms,
        )

    def subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        if subagent in self._hidden_subagents:
            self._hidden_subagents.remove(subagent)
            return
        callback = getattr(self.renderer, "subagent_cancelled", None)
        if callable(callback):
            call_renderer_with_supported_kwargs(
                callback,
                subagent,
                result,
                eval_id=eval_id,
                row_id=row_id,
                duration_ms=duration_ms,
            )


class EvalSubagentRenderer:
    """Render QuickJS eval-internal subagent lifecycle events."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self._labels: dict[str, str] = {}

    def handle(self, event: dict[str, Any]) -> None:
        phase = str(event.get("phase") or "")
        subagent_id = str(event.get("id") or "")
        if not subagent_id:
            return
        if phase == "start":
            name = eval_subagent_name(event)
            self._labels[subagent_id] = name
            callback = getattr(self.renderer, "eval_subagent_started", None)
            if callable(callback):
                call_renderer_with_supported_kwargs(
                    callback,
                    name,
                    str(event.get("description") or ""),
                    eval_id=str(event.get("eval_id") or ""),
                    row_id=subagent_id,
                    model=str(event.get("model") or ""),
                    label=str(event.get("label") or ""),
                )
            else:
                self.renderer.subagent_started(
                    name,
                    str(event.get("description") or ""),
                    origin=EVAL_SUBAGENT,
                )
        elif phase == "complete":
            name = self._labels.pop(subagent_id, eval_subagent_name(event))
            callback = getattr(self.renderer, "eval_subagent_finished", None)
            if callable(callback):
                call_renderer_with_supported_kwargs(
                    callback,
                    name,
                    eval_id=str(event.get("eval_id") or ""),
                    row_id=subagent_id,
                    duration_ms=event_duration_ms(event),
                )
            else:
                self.renderer.subagent_finished(name)
        elif phase == "error":
            name = self._labels.pop(subagent_id, eval_subagent_name(event))
            error = str(event.get("error") or "error")
            callback = getattr(self.renderer, "eval_subagent_cancelled", None)
            if callable(callback):
                call_renderer_with_supported_kwargs(
                    callback,
                    name,
                    error,
                    eval_id=str(event.get("eval_id") or ""),
                    row_id=subagent_id,
                    duration_ms=event_duration_ms(event),
                )
            else:
                self.renderer.subagent_cancelled(name, error)


async def consume_custom_events(stream: Any, renderer: Any, rubric: RubricEventRenderer) -> None:
    """Dispatch custom events without competing stream consumers."""
    eval_renderer = EvalSubagentRenderer(renderer)
    async for event in stream:
        if isinstance(event, dict) and event.get("type") == CORRECTION_EVENT:
            render_correction_event(event, renderer)
            continue
        if isinstance(event, dict) and rubric.handle(event):
            continue
        if custom_event_data(event) is not None:
            eval_renderer.handle(event)


def render_correction_event(event: dict[str, Any], renderer: Any) -> None:
    """Render one durable correction after its rejected assistant prose."""
    value = normalize_correction_event(event)
    callback = getattr(renderer, "correction", None)
    if callable(callback):
        callback(value)
    terminal_text = value["terminal_text"]
    if terminal_text:
        renderer.text_delta(terminal_text)


def custom_event_data(event: Any) -> dict[str, Any] | None:
    """Return a QuickJS eval subagent custom payload, if this is one."""
    if isinstance(event, dict) and event.get("type") == "subagent":
        return event
    return None


def eval_subagent_name(event: dict[str, Any]) -> str:
    """Build a stable visible label for one eval-internal subagent."""
    subagent_type = str(event.get("subagent_type") or "subagent")
    label = str(event.get("label") or "").strip()
    if not label:
        label = str(event.get("id") or "")[-8:] or "eval"
    return f"{subagent_type} [{label}]"


def event_duration_ms(event: dict[str, Any]) -> int | None:
    """Return optional event duration in milliseconds."""
    value = event.get("duration_ms")
    if isinstance(value, int | float):
        return int(value)
    return None


def call_renderer_with_supported_kwargs(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a renderer method without requiring every renderer to accept new metadata."""
    return callback(*args, **supported_kwargs(callback, kwargs))


def supported_kwargs(callback: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs accepted by callback, preserving all kwargs for **kwargs renderers."""
    try:
        parameters = signature(callback).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    allowed = {
        name
        for name, parameter in parameters.items()
        if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


async def run_turn(
    agent: Any,
    text: str,
    renderer: Any,
    thread_id: str,
    usage_callback: Callable[[dict[str, Any]], None] | None = None,
    rubric: str | None | object = None,
    rubric_max_iterations: int = 3,
    rubric_model_name: str = "",
    include_rubric_state: bool = False,
    planning_stage: str | None = None,
    planning_state: dict[str, str] | None = None,
    planning_context: Any | None = None,
    messages: list[Any] | None = None,
) -> TurnResult:
    """Stream one top-level agent turn and handle HITL approval loops.

    DeepAgents exposes separate async event streams for messages, tool calls,
    subagents, and final output. MIRA consumes them concurrently so the terminal
    can update as soon as each event arrives. If LangGraph interrupts for a
    write approval, ask_user prompt, or structured planning prompt, this
    function asks the renderer for the needed input and resumes the same thread
    with a ``Command`` payload.
    """
    payload: dict[str, Any] | Command = {
        "messages": list(messages) if messages is not None else [{"role": "user", "content": text}]
    }
    rubric_model_name = rubric_model_name or str(getattr(agent, "mira_rubric_model_name", "") or "")
    if include_rubric_state:
        payload["rubric"] = rubric
    if planning_stage in PLANNING_STAGES:
        payload["planning_stage"] = planning_stage
    if planning_state:
        payload.update(planning_state)
    config = {"configurable": {"thread_id": thread_id}}
    result = TurnResult()
    historical_tool_lifecycle = await checkpoint_tool_lifecycle(agent, config)

    while True:
        message_metadata = MessageInvocationMetadata()
        stream = await agent.astream_events(
            payload,
            config=config,
            version="v3",
            **({"context": planning_context} if planning_context is not None else {}),
            transformers=[
                CustomTransformer,
                lambda scope: MessageInvocationMetadataTransformer(scope, message_metadata),
            ],
        )
        event_renderer = SubagentRequestRenderer(renderer)
        rubric_renderer = RubricEventRenderer(
            event_renderer,
            rubric_max_iterations,
            grader_model=rubric_model_name,
        )
        output: dict[str, Any] = {}
        tool_call_start = len(result.tool_calls)
        tool_draft_start = len(result._tool_call_drafts)
        waiting_started = getattr(renderer, "waiting_started", None)
        if callable(waiting_started):
            waiting_started()

        try:
            await asyncio.gather(
                consume_live_tool_errors(stream, event_renderer, result),
                consume_custom_events(stream.custom, event_renderer, rubric_renderer),
                consume_messages(
                    stream.messages,
                    event_renderer,
                    result,
                    render_normal_tools=False,
                    invocation_metadata=message_metadata,
                ),
                consume_tool_calls(stream.tool_calls, event_renderer, result),
                consume_subagents(stream.subagents, event_renderer, rubric_renderer),
                capture_output(stream.output(), output),
            )
        except BaseException:
            rubric_renderer.cancel()
            raise

        if historical_tool_lifecycle is not None:
            render_output_tool_results(
                output.get("value"),
                event_renderer,
                result,
                historical=historical_tool_lifecycle,
            )
        result.rubric_evaluations.extend(rubric_renderer.evaluations)
        result.final_text = final_text(output.get("value")) or result.final_text
        usage_delta = result.commit_loop_usage(output.get("value"))
        if usage_callback is not None and (has_usage(usage_delta) or has_context_usage(usage_delta)):
            usage_callback(usage_delta)
        waiting_finished = getattr(renderer, "waiting_finished", None)
        if callable(waiting_finished):
            waiting_finished()
        renderer.finish_main()
        rubric_renderer.cancel()
        interrupts = await collect_interrupts(stream, output.get("value"))

        if not interrupts:
            pending_calls = output_tool_calls(output.get("value"))
            leaked_tool_repr = output_has_tool_call_repr(output.get("value"))
            stream_tool_calls_observed = len(result.tool_calls) > tool_call_start

            if pending_calls:
                raise RuntimeError(
                    await unexecuted_tool_call_error(
                        stream,
                        output.get("value"),
                        pending_calls=pending_calls,
                        leaked_tool_repr=False,
                        stream_tool_calls_observed=stream_tool_calls_observed,
                    )
                )
            if leaked_tool_repr:
                raise RuntimeError(
                    await unexecuted_tool_call_error(
                        stream,
                        output.get("value"),
                        pending_calls=[],
                        leaked_tool_repr=True,
                        stream_tool_calls_observed=stream_tool_calls_observed,
                    )
                )
            if include_rubric_state and isinstance(rubric, str) and rubric.strip():
                state = await completed_agent_state(agent, config)
                result.rubric_status = str(state.get("_rubric_status") or "")
                if not result.rubric_status and result.rubric_evaluations:
                    result.rubric_status = str(result.rubric_evaluations[-1].get("result") or "")
                rubric_renderer.finalize(result.rubric_status)
            if result.formal_review is not None:
                result.final_text = ""
            return result

        approval_bindings = render_interrupt_tool_calls(
            interrupts,
            output.get("value"),
            event_renderer,
            result,
            tool_call_start,
        )
        finalize_goal_interrupt = first_typed_interrupt(interrupts, "finalize_goal")
        finalize_plan_interrupt = first_typed_interrupt(interrupts, "finalize_plan")
        show_goal_interrupt = first_typed_interrupt(interrupts, "show_goal")
        show_plan_interrupt = first_typed_interrupt(interrupts, "show_plan")
        ask_user_interrupt = first_typed_interrupt(interrupts, "ask_user")
        if finalize_goal_interrupt is not None:
            call_id = ensure_control_tool_call(
                "finalize_goal",
                finalize_goal_interrupt,
                output.get("value"),
                event_renderer,
                result,
                tool_call_start,
                tool_draft_start,
            )
            decision = await resolve_control_surface(
                "finalize_goal",
                renderer.finalize_goal(finalize_goal_interrupt),
                event_renderer,
                result,
                call_id,
            )
            result.formal_review = decision if isinstance(decision, dict) else {"action": str(decision)}
            payload = Command(resume=decision)
        elif finalize_plan_interrupt is not None:
            call_id = ensure_control_tool_call(
                "finalize_plan",
                finalize_plan_interrupt,
                output.get("value"),
                event_renderer,
                result,
                tool_call_start,
                tool_draft_start,
            )
            decision = await resolve_control_surface(
                "finalize_plan",
                renderer.finalize_plan(finalize_plan_interrupt),
                event_renderer,
                result,
                call_id,
            )
            result.formal_review = decision if isinstance(decision, dict) else {"action": str(decision)}
            payload = Command(resume=decision)
        elif show_goal_interrupt is not None:
            call_id = ensure_control_tool_call(
                "show_goal",
                show_goal_interrupt,
                output.get("value"),
                event_renderer,
                result,
                tool_call_start,
                tool_draft_start,
            )
            await resolve_control_surface(
                "show_goal",
                renderer.show_goal(show_goal_interrupt),
                event_renderer,
                result,
                call_id,
            )
            result.final_text = ""
            return result
        elif show_plan_interrupt is not None:
            call_id = ensure_control_tool_call(
                "show_plan",
                show_plan_interrupt,
                output.get("value"),
                event_renderer,
                result,
                tool_call_start,
                tool_draft_start,
            )
            await resolve_control_surface(
                "show_plan",
                renderer.show_plan(show_plan_interrupt),
                event_renderer,
                result,
                call_id,
            )
            result.final_text = ""
            return result
        elif ask_user_interrupt is not None:
            call_id = ensure_control_tool_call(
                "ask_user",
                ask_user_interrupt,
                output.get("value"),
                event_renderer,
                result,
                tool_call_start,
                tool_draft_start,
            )
            answer = await resolve_control_surface(
                "ask_user",
                renderer.ask_user(ask_user_interrupt),
                event_renderer,
                result,
                call_id,
            )
            payload = Command(resume=answer)
        else:
            annotate_filesystem_approvals(interrupts, getattr(agent, "mira_backend", None))
            decisions = await renderer.ask_approvals(interrupts)
            render_edited_tool_calls(
                decisions,
                approval_bindings,
                event_renderer,
                result,
            )
            payload = Command(resume={"decisions": decisions})


async def completed_agent_state(agent: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Return checkpoint values used to reconcile rubric terminal status."""
    getter = getattr(agent, "aget_state", None)
    if not callable(getter):
        return {}
    try:
        snapshot = await getter(config)
    except Exception:
        return {}
    values = getattr(snapshot, "values", None)
    return values if isinstance(values, dict) else {}


async def checkpoint_tool_lifecycle(
    agent: Any,
    config: dict[str, Any],
) -> Counter[tuple[str, ...]] | None:
    """Return the executed lifecycle already present before a top-level turn.

    Agents without checkpoint access are treated as fresh test/provider surfaces.
    A checkpoint read failure disables final-output recovery for safety so historical
    calls cannot be mistaken for current work.
    """
    getter = getattr(agent, "aget_state", None)
    if not callable(getter):
        return Counter()
    try:
        snapshot = await getter(config)
    except Exception:
        return None
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return Counter()
    return Counter(lifecycle_identity(item) for item in output_tool_lifecycle(values))


def first_ask_user_interrupt(interrupts: list[Any]) -> Any | None:
    """Return the first ask_user interrupt payload, if present."""
    return first_typed_interrupt(interrupts, "ask_user")


def first_typed_interrupt(interrupts: list[Any], interrupt_type: str) -> Any | None:
    """Return the first interrupt payload with the requested type."""
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        if isinstance(value, dict) and value.get("type") == interrupt_type:
            return interrupt
    return None


def interrupt_value(interrupt: Any) -> Any:
    """Extract the LangGraph interrupt value from common payload shapes."""
    return getattr(interrupt, "value", interrupt)


def annotate_filesystem_approvals(interrupts: list[Any], backend: Any) -> None:
    """Add accurate filesystem consequences for approval renderers."""
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        actions = value.get("action_requests") if isinstance(value, dict) else None
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = str(action.get("name") or "")
            args = action.get("args")
            if not isinstance(args, dict):
                continue
            if name == "write_file":
                action[APPROVAL_CONSEQUENCE] = write_file_consequence(
                    backend,
                    str(args.get("file_path") or ""),
                )
            elif name == "edit_file":
                action[APPROVAL_CONSEQUENCE] = (
                    "Makes targeted replacements while preserving the rest of the existing file."
                )
            elif name == "delete":
                action[APPROVAL_CONSEQUENCE] = (
                    "Recursively deletes this path and every descendant. "
                    "This is destructive and cannot be undone."
                )


def write_file_consequence(backend: Any, file_path: str) -> str:
    """Describe whether DeepAgents write_file will create or replace."""
    reader = getattr(backend, "read", None)
    if not file_path or not callable(reader):
        return "Writes the complete file; any existing content will be replaced."
    try:
        result = reader(file_path, offset=0, limit=1)
    except Exception:
        return "Writes the complete file; any existing content will be replaced."
    error = str(getattr(result, "error", "") or "")
    if not error:
        return "Replaces the entire existing file."
    if "not found" in error.lower():
        return "Creates a new file."
    return "Writes the complete file; any existing content will be replaced."


def render_interrupt_tool_calls(
    interrupts: list[Any],
    output: Any,
    renderer: Any,
    result: TurnResult,
    tool_call_start: int,
) -> list[dict[str, Any]]:
    """Recover interrupting calls before opening their dedicated surfaces."""
    for raw_call in output_tool_calls(output):
        call = normalized_call(raw_call)
        name = str(call["name"])
        call_id = str(call.get("id") or "")
        if result.record_tool_call(name, call_id):
            renderer.tool_call(name, call.get("args", {}), call_id=call_id)

    observed = Counter(result.tool_calls[tool_call_start:])
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        if not isinstance(value, dict):
            continue
        actions = value.get("action_requests")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = str(action.get("name") or "tool")
            if observed[name] > 0:
                observed[name] -= 1
                continue
            call_id = str(
                action.get("id")
                or action.get("call_id")
                or action.get("tool_call_id")
                or ""
            )
            if not result.record_tool_call(name, call_id):
                continue
            renderer.tool_call(
                name,
                action.get("args", {}),
                call_id=call_id,
            )

    return approval_call_bindings(interrupts, result, tool_call_start)


def approval_call_bindings(
    interrupts: list[Any],
    result: TurnResult,
    tool_call_start: int,
) -> list[dict[str, Any]]:
    """Pair approval actions with current-loop calls in prompt order."""
    available = list(range(tool_call_start, len(result.tool_calls)))
    used: set[int] = set()
    bindings: list[dict[str, Any]] = []
    for interrupt in interrupts:
        value = interrupt_value(interrupt)
        actions = value.get("action_requests") if isinstance(value, dict) else None
        if not isinstance(actions, list):
            continue
        for action in actions:
            name = str(action.get("name") or "tool") if isinstance(action, dict) else "tool"
            match = next(
                (
                    index
                    for index in available
                    if index not in used and result.tool_calls[index] == name
                ),
                None,
            )
            if match is not None:
                used.add(match)
            bindings.append(
                {
                    "name": name,
                    "args": action.get("args", {}) if isinstance(action, dict) else {},
                    "call_id": result._tool_call_ids[match] if match is not None else "",
                    "result_index": match,
                }
            )
    return bindings


def render_edited_tool_calls(
    decisions: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    renderer: Any,
    result: TurnResult,
) -> None:
    """Replace proposed call args with the action selected for execution."""
    callback = getattr(renderer, "tool_call_updated", None)
    for decision, binding in zip(decisions, bindings, strict=False):
        if not isinstance(decision, dict) or decision.get("type") != "edit":
            resolve_unedited_tool_call(renderer, binding)
            continue
        edited = decision.get("edited_action")
        if not isinstance(edited, dict) or not isinstance(edited.get("args"), dict):
            resolve_unedited_tool_call(renderer, binding)
            continue
        name = str(edited.get("name") or binding["name"])
        args = edited["args"]
        if name == binding["name"] and args == binding["args"]:
            resolve_unedited_tool_call(renderer, binding)
            continue
        result_index = binding.get("result_index")
        if isinstance(result_index, int):
            result.update_tool_call_name(result_index, name)
        if callable(callback):
            callback(name, args, call_id=str(binding.get("call_id") or ""))
        else:
            renderer.tool_call(name, args, call_id=str(binding.get("call_id") or ""))


def resolve_unedited_tool_call(renderer: Any, binding: dict[str, Any]) -> None:
    """Advance renderer idless matching past an approval without an amendment."""
    callback = getattr(renderer, "tool_call_approval_resolved", None)
    if callable(callback):
        callback(
            str(binding.get("name") or "tool"),
            call_id=str(binding.get("call_id") or ""),
        )


def ensure_control_tool_call(
    name: str,
    interrupt: Any,
    output: Any,
    renderer: Any,
    result: TurnResult,
    tool_call_start: int,
    tool_draft_start: int,
) -> str:
    """Promote or reconstruct the current control call before its surface."""
    observed, call_id = result.tool_call_since(name, tool_call_start)
    if observed:
        return call_id

    fallback_call = next(
        (
            normalized
            for normalized in (
                normalized_call(call)
                for call in reversed(output_tool_calls(output))
            )
            if str(normalized.get("name") or "tool") == name
        ),
        None,
    )
    value = interrupt_value(interrupt)
    args = (
        fallback_call.get("args", {})
        if isinstance(fallback_call, dict)
        else {
            key: item
            for key, item in value.items()
            if key != "type"
        }
        if isinstance(value, dict)
        else {}
    )
    call_id = (
        str(
            fallback_call.get("id")
            or fallback_call.get("call_id")
            or fallback_call.get("tool_call_id")
            or ""
        )
        if isinstance(fallback_call, dict)
        else result.tool_call_draft_since(name, tool_draft_start)
    )
    result.record_tool_call(name, call_id)
    renderer.tool_call(name, args, call_id=call_id)
    return call_id


async def resolve_control_surface(
    name: str,
    operation: Any,
    renderer: Any,
    result: TurnResult,
    call_id: str,
) -> Any:
    """Resolve one dedicated surface and complete its existing tool block."""
    try:
        value = await operation
    except Exception as exc:
        complete_control_tool(
            name,
            str(exc) or type(exc).__name__,
            renderer,
            result,
            call_id,
            is_error=True,
        )
        raise

    complete_control_tool(name, str(value), renderer, result, call_id)
    return value


def complete_control_tool(
    name: str,
    text: str,
    renderer: Any,
    result: TurnResult,
    call_id: str,
    *,
    is_error: bool = False,
) -> None:
    """Attach one dedicated-surface outcome to its original control call."""
    if not result.record_tool_result(text, call_id, name):
        return
    if is_error:
        renderer.completed_tool_error(name, text, call_id=call_id)
    else:
        renderer.completed_tool_result(name, text, call_id=call_id)


def render_output_tool_results(
    output: Any,
    renderer: Any,
    result: TurnResult,
    *,
    historical: Counter[tuple[str, ...]] | None = None,
) -> None:
    """Recover executed calls/results that only appear in final graph state."""
    recovered_tool_call = getattr(renderer, "recovered_tool_call", None)
    lifecycle = current_tool_lifecycle(output, historical or Counter())
    control_error_ids = {
        str(item.get("call_id") or "")
        for item in lifecycle
        if item["type"] == "tool_result"
        and item["name"] in CONTROL_TOOLS
        and item.get("status") == "error"
        and item.get("call_id")
    }
    control_error_names = Counter(
        str(item["name"])
        for item in lifecycle
        if item["type"] == "tool_result"
        and item["name"] in CONTROL_TOOLS
        and item.get("status") == "error"
        and not item.get("call_id")
    )
    result_occurrences: Counter[tuple[str, str]] = Counter()
    for item in lifecycle:
        if item["type"] == "tool_call":
            call = normalized_call(item["call"])
            name = str(call["name"])
            call_id = str(call.get("id") or "")
            if name in CONTROL_TOOLS:
                if call_id and call_id not in control_error_ids:
                    continue
                if not call_id:
                    if control_error_names[name] < 1:
                        continue
                    control_error_names[name] -= 1
            if not result.record_tool_call(name, call_id):
                continue
            callback = recovered_tool_call if callable(recovered_tool_call) else renderer.tool_call
            callback(name, call.get("args", {}), call_id=call_id)
            continue

        if item["name"] in CONTROL_TOOLS:
            if item.get("status") != "error":
                continue
        text = item["output"]
        call_id = item["call_id"]
        occurrence = None
        if not call_id:
            value_key = (str(item["name"]), str(text))
            result_occurrences[value_key] += 1
            occurrence = result_occurrences[value_key]
        render_tool_completion(
            renderer,
            result,
            name=item["name"],
            text=text,
            call_id=call_id,
            is_error=item.get("status") == "error",
            recovered=True,
            occurrence=occurrence,
        )


def current_tool_lifecycle(
    output: Any,
    historical: Counter[tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Remove the pre-turn lifecycle from a completed graph-state projection."""
    remaining = historical.copy()
    current = []
    for item in output_tool_lifecycle(output):
        identity = lifecycle_identity(item)
        if remaining[identity] > 0:
            remaining[identity] -= 1
            continue
        current.append(item)
    return current


def lifecycle_identity(item: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable identity for occurrence-aware lifecycle subtraction."""
    if item.get("type") == "tool_call":
        call = normalized_call(item.get("call") or {})
        call_id = str(call.get("id") or "")
        if call_id:
            return ("tool_call", "id", call_id)
        args = json.dumps(call.get("args", {}), sort_keys=True, default=str, ensure_ascii=False)
        return ("tool_call", "value", str(call.get("name") or "tool"), args)

    call_id = str(item.get("call_id") or "")
    if call_id:
        return ("tool_result", "id", call_id)
    return (
        "tool_result",
        "value",
        str(item.get("name") or "tool"),
        str(item.get("status") or "success"),
        str(item.get("output") or ""),
    )


async def unexecuted_tool_call_error(
    stream: Any,
    output: Any,
    *,
    pending_calls: list[Any],
    leaked_tool_repr: bool,
    stream_tool_calls_observed: bool,
) -> str:
    """Return a compact diagnostic for a terminal output with pending tool calls."""
    names = [
        str(call.get("name") or "tool")
        for call in pending_calls
        if isinstance(call, dict)
    ]
    name_text = ", ".join(names) or "tool"
    interrupted = await stream_interrupted(stream)
    diagnostic = (
        f"interrupted={interrupted}; "
        f"stream_tool_calls_observed={stream_tool_calls_observed}; "
        f"leaked_tool_repr={leaked_tool_repr}; "
        f"final_messages={output_message_shapes(output)}"
    )
    return f"native HITL resume returned unexecuted tool call(s): {name_text}; diagnostic: {diagnostic}"


async def stream_interrupted(stream: Any) -> bool:
    """Return whether a LangGraph run stream reported an interrupt."""
    callback = getattr(stream, "interrupted", None)
    if not callable(callback):
        return False
    value = callback()
    if hasattr(value, "__await__"):
        value = await value
    return bool(value)


def output_message_shapes(output: Any) -> list[dict[str, Any]] | str:
    """Return compact final-output message shapes for failure diagnostics."""
    if not isinstance(output, dict):
        return type(output).__name__
    messages = output.get("messages")
    if not isinstance(messages, list):
        return []
    return [message_shape(message) for message in messages[-3:]]


def message_shape(message: Any) -> dict[str, Any]:
    """Return compact class, content, text, and tool-call details for one message."""
    content = message_value(message, "content")
    tool_calls = message_value(message, "tool_calls") or []
    text = message_value(message, "text")
    shape: dict[str, Any] = {
        "class": message.__class__.__name__,
        "content_type": type(content).__name__,
        "tool_calls": compact_tool_calls(tool_calls),
    }
    if isinstance(text, str) and text.strip():
        shape["text_sample"] = compact_sample(text)
    elif isinstance(content, str) and content.strip():
        shape["content_sample"] = compact_sample(content)
    return shape


def compact_tool_calls(tool_calls: Any) -> list[dict[str, str]]:
    """Return names and ids for compact tool-call diagnostics."""
    if not isinstance(tool_calls, list):
        return []
    compact = []
    for call in tool_calls[:5]:
        if isinstance(call, dict):
            compact.append({
                "name": str(call.get("name") or "tool"),
                "id": str(call.get("id") or call.get("call_id") or call.get("tool_call_id") or ""),
            })
    return compact


def message_value(message: Any, key: str) -> Any:
    """Read a field from dict-like and object-like message shapes."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def compact_sample(value: Any, limit: int = 180) -> str:
    """Return a single-line diagnostic sample."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def task_descriptions(calls: list[Any]) -> list[str]:
    """Extract request descriptions from task tool-call payloads."""
    descriptions = []
    for call in calls:
        args = call_args(call)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, json.JSONDecodeError):
                args = {}
        if isinstance(args, dict) and args.get("description"):
            descriptions.append(str(args["description"]))
    return descriptions


def call_args(call: Any) -> Any:
    """Extract tool-call args from a dict or object."""
    return tool_call_args(call)
