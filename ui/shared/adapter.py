"""Shared projection from the MIRA Core API to owned UI presentation callbacks."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable

from core.api.events import (
    ArtifactEvent,
    CompactionEvent,
    FrontendEvent,
    InformationEvent,
    MessageEvent,
    RubricEvent,
    RuntimeEvent,
    SubagentEvent,
    ToolEvent,
    UsageEvent,
)
from core.api.requests import (
    ApprovalRequest,
    ArtifactDisplayRequest,
    ArtifactReviewRequest,
    AskUserRequest,
    ConfirmationRequest,
    FrontendRequest,
    MCPApprovalRequest,
)


class RendererAdapter:
    """Project typed Core events onto the callback surface shared by MIRA UIs."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer

    @property
    def manages_subagent_animation(self) -> bool:
        return bool(getattr(self.renderer, "manages_subagent_animation", False))

    def subagent_label(self, subagent: Any) -> str:
        callback = getattr(self.renderer, "subagent_label", None)
        if callable(callback):
            return str(callback(subagent))
        return str(getattr(subagent, "name", "subagent") or "subagent")

    def emit(self, event: FrontendEvent) -> None:
        """Project one semantic event onto the unchanged renderer surface."""
        if isinstance(event, MessageEvent):
            self._message(event)
        elif isinstance(event, ToolEvent):
            self._tool(event)
        elif isinstance(event, SubagentEvent):
            self._subagent(event)
        elif isinstance(event, RubricEvent):
            self._rubric(event)
        elif isinstance(event, CompactionEvent):
            self._call(f"compaction_{'started' if event.phase == 'start' else 'finished'}")
        elif isinstance(event, UsageEvent):
            self._call("usage_updated")
        elif isinstance(event, InformationEvent):
            if event.correction is not None:
                self._call("correction", dict(event.correction), created_at=event.created_at)
            else:
                callback = getattr(self.renderer, "system_message", None)
                if callable(callback):
                    self._call("system_message", event.text, kind=event.kind, created_at=event.created_at)
                elif hasattr(self.renderer, "console"):
                    self.renderer.console.print(event.text)
        elif isinstance(event, ArtifactEvent):
            self._call("artifact_state_changed", event)
        elif isinstance(event, RuntimeEvent):
            self._runtime(event)

    async def request(self, request: FrontendRequest) -> Any:
        """Use the renderer's existing in-process interaction UI."""
        if isinstance(request, ApprovalRequest):
            return await self._await_call("ask_approvals", list(request.interrupts))
        if isinstance(request, AskUserRequest):
            return await self._await_call("ask_user", request.interrupt)
        if isinstance(request, ArtifactReviewRequest):
            if callable(getattr(self.renderer, "review_artifact", None)):
                return await self._await_call(
                    "review_artifact",
                    request.artifact_type,
                    request.interrupt,
                    dict(request.artifact or {}),
                )
            return await self._await_call(
                f"finalize_{request.artifact_type}",
                request.interrupt,
            )
        if isinstance(request, ArtifactDisplayRequest):
            return await self._await_call(f"show_{request.artifact_type}", request.interrupt)
        if isinstance(request, MCPApprovalRequest):
            callback = getattr(self.renderer, "approve_mcp_server", None)
            if not callable(callback):
                raise RuntimeError("this frontend cannot approve MCP servers")
            server = request.server
            return await self._await(callback, server, request.preview)
        if isinstance(request, ConfirmationRequest):
            method = {
                "create_git_repo": "ask_create_git_repo",
                "continue_without_git": "ask_continue_without_git",
            }.get(request.kind)
            if method is not None:
                return await self._await_call(method, request.message)
        raise RuntimeError(f"unsupported frontend request: {type(request).__name__}")

    def _message(self, event: MessageEvent) -> None:
        if event.phase == "user":
            self._call(
                "user_message",
                event.text,
                planning=event.mode == "planning",
                created_at=event.created_at,
            )
        elif event.phase == "content":
            self._call("text_delta", event.text, created_at=event.created_at)
        elif event.phase == "reasoning":
            self._call("reasoning_delta", event.text, created_at=event.created_at)
        elif event.phase == "discard_reasoning":
            self._call("discard_reasoning")
        elif event.phase == "end":
            self._call("model_stream_finished")

    def _tool(self, event: ToolEvent) -> None:
        common = {
            "call_id": event.tool_call_id,
            "created_at": event.created_at,
            "duration_ms": event.duration_ms,
        }
        if event.phase == "delegation":
            self._call("delegation_started", list(event.calls), created_at=event.created_at)
        elif event.phase == "arguments_delta":
            self._call("tool_call_delta", event.name, event.arguments, **common)
        elif event.phase in {"start", "recovered_start"}:
            method = "recovered_tool_call" if event.phase == "recovered_start" else "tool_call"
            self._call(method, event.name, event.arguments, **common)
        elif event.phase == "update":
            self._call("tool_call_updated", event.name, event.arguments, **common)
        elif event.phase == "approval_resolved":
            self._call("tool_call_approval_resolved", event.name, call_id=event.tool_call_id)
        elif event.phase in {"result", "completed_result", "recovered_result"}:
            method = {
                "result": "tool_result",
                "completed_result": "completed_tool_result",
                "recovered_result": "recovered_tool_result",
            }[event.phase]
            self._call(method, event.name, event.result, **common)
        elif event.phase in {"error", "completed_error", "recovered_error"}:
            method = {
                "error": "tool_result",
                "completed_error": "completed_tool_error",
                "recovered_error": "recovered_tool_error",
            }[event.phase]
            self._call(method, event.name, event.result, **common)
        elif event.phase == "stop":
            self._call("stop_active_tools", event.status)

    def _subagent(self, event: SubagentEvent) -> None:
        lifecycle = {
            "live_start": "start_subagent_live",
            "live_tick": "tick_subagents",
            "live_stop": "stop_subagent_live",
            "cancel_all": "subagents_cancelled",
        }.get(event.phase)
        if lifecycle:
            self._call(lifecycle)
            return
        if event.phase == "delegation_update":
            self._call("delegation_delta", list(event.metadata.get("calls", ())))
            return
        method = {
            "start": "subagent_started",
            "request_update": "subagent_request_updated",
            "finish": "subagent_finished",
            "cancel": "subagent_cancelled",
            "eval_start": "eval_subagent_started",
            "eval_finish": "eval_subagent_finished",
            "eval_cancel": "eval_subagent_cancelled",
        }.get(event.phase)
        if method is None:
            return
        if event.phase == "request_update":
            self._call(method, event.name, event.task_input)
            return
        value = event.task_input if event.phase in {"start", "eval_start"} else event.result
        self._call(
            method,
            event.name,
            value,
            origin=event.origin,
            eval_id=event.eval_id,
            row_id=event.row_id,
            model=event.model,
            label=event.label,
            duration_ms=event.metadata.get("duration_ms"),
            created_at=event.created_at,
        )

    def _rubric(self, event: RubricEvent) -> None:
        if event.phase == "lifecycle":
            self._call("rubric_lifecycle_event", dict(event.lifecycle or {}))
        elif event.phase == "finish":
            self._call(
                "rubric_evaluation_finished",
                dict(event.evaluation or {}),
                event.max_iterations,
                created_at=event.created_at,
            )
        elif event.phase == "cancel":
            self._call("rubric_evaluations_cancelled")
        elif event.phase == "status":
            self._call(
                "rubric_evaluation_status",
                event.run_id,
                event.pass_number,
                event.status,
                event.max_iterations,
            )
        else:
            self._call(
                "rubric_evaluation_started",
                event.run_id,
                event.pass_number,
                event.max_iterations,
                grader_model=event.grader_model,
                phase=event.phase,
                created_at=event.created_at,
            )

    def _runtime(self, event: RuntimeEvent) -> None:
        if event.kind == "startup":
            self._call("startup_progress", event.state)
        elif event.kind == "waiting":
            if event.state == "start":
                detail = event.detail if isinstance(event.detail, dict) else {}
                self._call(
                    "waiting_started",
                    detail.get("label"),
                    **{key: value for key, value in detail.items() if key != "label"},
                )
            else:
                self._call("waiting_finished")
        elif event.kind == "message_group" and event.state == "finish":
            self._call("finish_main")

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        callback = getattr(self.renderer, method, None)
        if not callable(callback):
            return None
        return callback(*args, **_supported_kwargs(callback, kwargs))

    async def _await_call(self, method: str, *args: Any) -> Any:
        callback = getattr(self.renderer, method, None)
        if not callable(callback):
            raise RuntimeError(f"this frontend does not support {method}")
        return await self._await(callback, *args)

    @staticmethod
    async def _await(callback: Callable[..., Any], *args: Any) -> Any:
        result = callback(*args)
        return await result if hasattr(result, "__await__") else result


def _supported_kwargs(callback: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass optional event metadata only when a renderer accepts it."""
    try:
        parameters = signature(callback).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters and value is not None
    }


__all__ = ["RendererAdapter"]
