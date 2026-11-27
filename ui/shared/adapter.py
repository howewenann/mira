"""Shared projection from the public MIRA API to owned UI callbacks."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable

from mira.api import (
    ApprovalRequest,
    ArtifactEvent,
    ArtifactDisplayRequest,
    ArtifactReviewRequest,
    AskUserRequest,
    CompactionEvent,
    FrontendEvent,
    FrontendRequest,
    InformationEvent,
    MCPApprovalRequest,
    MCPEvent,
    MCPElicitationRequest,
    MessageEvent,
    RubricEvent,
    RuntimeEvent,
    SubagentEvent,
    ToolEvent,
    UsageEvent,
    WorkflowEvent,
)


class RendererAdapter:
    """Project typed Core events onto the callback surface shared by MIRA UIs."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer

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
            self._call(f"compaction_{'started' if event.phase == 'start' else 'finished'}", frontend_event=event)
        elif isinstance(event, UsageEvent):
            self._call("usage_updated", frontend_event=event)
        elif isinstance(event, InformationEvent):
            if event.correction is not None:
                self._call("correction", dict(event.correction), created_at=event.created_at, frontend_event=event)
            else:
                callback = getattr(self.renderer, "system_message", None)
                if callable(callback):
                    self._call("system_message", event.text, kind=event.kind, created_at=event.created_at, frontend_event=event)
                elif hasattr(self.renderer, "console"):
                    self.renderer.console.print(event.text)
        elif isinstance(event, ArtifactEvent):
            self._call("artifact_state_changed", event, frontend_event=event)
        elif isinstance(event, RuntimeEvent):
            self._runtime(event)
        elif isinstance(event, MCPEvent):
            self._call("mcp_activity", event.phase, event.detail, frontend_event=event)
        elif isinstance(event, WorkflowEvent):
            self._workflow(event)

    async def request(self, request: FrontendRequest) -> Any:
        """Use the renderer's existing in-process interaction UI."""
        if isinstance(request, ApprovalRequest):
            return await self._await_call("ask_approvals", list(request.interrupts))
        if isinstance(request, AskUserRequest):
            return await self._await_call("ask_user", request.interrupt)
        if isinstance(request, MCPElicitationRequest):
            return await self._await_call("answer_mcp_elicitation", request.interrupt)
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
        raise RuntimeError(f"unsupported frontend request: {type(request).__name__}")

    def _message(self, event: MessageEvent) -> None:
        if event.phase == "user":
            self._call(
                "user_message",
                event.text,
                planning=event.mode == "planning",
                created_at=event.created_at,
                frontend_event=event,
            )
        elif event.phase == "content":
            self._call("text_delta", event.text, created_at=event.created_at, frontend_event=event)
        elif event.phase == "reasoning":
            self._call("reasoning_delta", event.text, created_at=event.created_at, frontend_event=event)
        elif event.phase == "discard_reasoning":
            self._call("discard_reasoning", frontend_event=event)
        elif event.phase == "end":
            self._call("model_stream_finished", frontend_event=event)

    def _tool(self, event: ToolEvent) -> None:
        common = {
            "call_id": event.tool_call_id,
            "created_at": event.created_at,
            "duration_ms": event.duration_ms,
        }
        if event.phase == "delegation":
            self._call("delegation_started", list(event.calls), created_at=event.created_at, frontend_event=event)
        elif event.phase == "arguments_delta":
            self._call("tool_call_delta", event.name, event.arguments, **common, frontend_event=event)
        elif event.phase in {"start", "recovered_start"}:
            method = "tool_call"
            if event.phase == "recovered_start" and callable(getattr(self.renderer, "recovered_tool_call", None)):
                method = "recovered_tool_call"
            self._call(method, event.name, event.arguments, **common, frontend_event=event)
        elif event.phase == "update":
            self._call("tool_call_updated", event.name, event.arguments, **common, frontend_event=event)
        elif event.phase == "approval_resolved":
            self._call("tool_call_approval_resolved", event.name, call_id=event.tool_call_id, frontend_event=event)
        elif event.phase in {"result", "completed_result", "recovered_result"}:
            method = {
                "result": "tool_result",
                "completed_result": "completed_tool_result",
                "recovered_result": "recovered_tool_result",
            }[event.phase]
            if method == "recovered_tool_result" and not callable(getattr(self.renderer, method, None)):
                method = "tool_result"
            self._call(method, event.name, event.result, **common, frontend_event=event)
        elif event.phase in {"error", "completed_error", "recovered_error"}:
            method = {
                "error": "tool_error",
                "completed_error": "completed_tool_error",
                "recovered_error": "recovered_tool_error",
            }[event.phase]
            self._call(method, event.name, event.result, **common, frontend_event=event)
        elif event.phase == "stop":
            if event.name or event.tool_call_id:
                self._call("tool_call_stopped", event.name, status=event.status, **common, frontend_event=event)
            else:
                self._call("stop_active_tools", event.status, frontend_event=event)

    def _subagent(self, event: SubagentEvent) -> None:
        lifecycle = {
            "live_start": "start_subagent_live",
            "live_tick": "tick_subagents",
            "live_stop": "stop_subagent_live",
            "cancel_all": "subagents_cancelled",
        }.get(event.phase)
        if lifecycle:
            self._call(lifecycle, frontend_event=event)
            return
        if event.phase == "delegation_update":
            self._call("delegation_delta", list(event.metadata.get("calls", ())), frontend_event=event)
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
            self._call(method, event.name, event.task_input, frontend_event=event)
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
            inspection_id=str(event.metadata.get("inspection_id") or ""),
            created_at=event.created_at,
            frontend_event=event,
        )

    def _rubric(self, event: RubricEvent) -> None:
        if event.phase == "lifecycle":
            self._call("rubric_lifecycle_event", dict(event.lifecycle or {}), frontend_event=event)
        elif event.phase == "finish":
            self._call(
                "rubric_evaluation_finished",
                dict(event.evaluation or {}),
                event.max_iterations,
                created_at=event.created_at,
                frontend_event=event,
            )
        elif event.phase == "cancel":
            self._call("rubric_evaluations_cancelled", frontend_event=event)
        elif event.phase == "status":
            self._call(
                "rubric_evaluation_status",
                event.run_id,
                event.pass_number,
                event.status,
                event.max_iterations,
                frontend_event=event,
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
                frontend_event=event,
            )

    def _runtime(self, event: RuntimeEvent) -> None:
        if event.kind == "startup":
            self._call("startup_progress", event.state, frontend_event=event)
        elif event.kind == "waiting":
            if event.state == "start":
                detail = event.detail if isinstance(event.detail, dict) else {}
                self._call(
                    "waiting_started",
                    detail.get("label"),
                    **{key: value for key, value in detail.items() if key != "label"},
                    frontend_event=event,
                )
            else:
                self._call("waiting_finished", frontend_event=event)
        elif event.kind == "message_group" and event.state == "finish":
            self._call("finish_main", frontend_event=event)
        elif event.kind == "session":
            self._call("session_state_changed", event, frontend_event=event)

    def _workflow(self, event: WorkflowEvent) -> None:
        if event.phase == "run_start":
            self._call(
                "workflow_started",
                event.workflow_id,
                event.workflow_name,
                frontend_event=event,
            )
        elif event.phase == "task_start":
            self._call(
                "workflow_task_started",
                event.task_id,
                event.name,
                event.step,
                input_state=event.input_state,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "task_waiting":
            self._call(
                "workflow_task_waiting",
                event.task_id,
                event.name,
                event.step,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "task_resume":
            self._call(
                "workflow_task_resumed",
                event.task_id,
                event.name,
                event.step,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase in {"agent_start", "agent_resume"}:
            self._call(
                "workflow_agent_started",
                event.task_id,
                event.name,
                event.inspection_id,
                task_input=event.task_input,
                resumed=event.phase == "agent_resume",
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "agent_waiting":
            self._call(
                "workflow_agent_waiting",
                event.task_id,
                event.name,
                event.inspection_id,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "agent_finish":
            self._call(
                "workflow_agent_finished",
                event.task_id,
                event.name,
                event.inspection_id,
                status=event.status,
                result=event.result,
                error=event.error,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "task_finish":
            self._call(
                "workflow_task_finished",
                event.task_id,
                event.name,
                event.step,
                status=event.status,
                error=event.error,
                result=event.result,
                result_available=event.result_available,
                workflow_id=event.workflow_id,
                frontend_event=event,
            )
        elif event.phase == "run_finish":
            self._call(
                "workflow_finished",
                event.workflow_id,
                final_state=event.final_state,
                final_state_available=event.final_state_available,
                frontend_event=event,
            )
        elif event.phase == "run_cancel":
            self._call(
                "workflow_cancelled",
                event.workflow_id,
                error=event.error,
                frontend_event=event,
            )

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
