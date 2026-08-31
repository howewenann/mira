"""Small convenience projection from Core actions to consumer events."""

from __future__ import annotations

from typing import Any, Mapping

from core.interface.events import (
    ArtifactEvent,
    ArtifactPhase,
    ArtifactType,
    CompactionEvent,
    InformationEvent,
    MCPEvent,
    MCPPhase,
    MessageEvent,
    RubricEvent,
    RubricPhase,
    RuntimeEvent,
    SubagentEvent,
    ToolEvent,
    UsageEvent,
)
from core.interface.protocol import Frontend
from core.interface.requests import (
    ApprovalRequest,
    ArtifactDisplayRequest,
    ArtifactReviewRequest,
    AskUserRequest,
)


class FrontendEmitter:
    """Emit typed frontend events while runtime code follows native streams.

    The narrow convenience methods keep stream consumers readable. They are not
    a second transport: every method immediately becomes one event or request
    on the Core Interface's ``Frontend`` contract.
    """

    def __init__(
        self,
        frontend: Frontend,
        *,
        session_id: str = "",
        turn_id: str = "",
    ) -> None:
        self.frontend = frontend
        self.session_id = session_id
        self.turn_id = turn_id
        self._context_notice_rendered = False

    @property
    def manages_subagent_animation(self) -> bool:
        return bool(getattr(self.frontend, "manages_subagent_animation", False))

    def _identity(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "session_id": str(kwargs.pop("session_id", self.session_id) or ""),
            "turn_id": str(kwargs.pop("turn_id", self.turn_id) or ""),
            "message_id": str(kwargs.pop("message_id", "") or ""),
            "namespace": tuple(str(item) for item in (kwargs.pop("namespace", ()) or ())),
            "metadata": dict(kwargs.pop("metadata", {}) or {}),
            "created_at": str(kwargs.pop("created_at", "") or ""),
        }

    def user_message(
        self,
        text: str,
        *,
        planning: bool = False,
        attachments: list[dict[str, str]] | None = None,
        **identity: Any,
    ) -> None:
        self.frontend.emit(
            MessageEvent(
                phase="user",
                text=text,
                content_blocks=({"type": "text", "text": text},),
                mode="planning" if planning else "action",
                attachments=tuple(attachments or ()),
                **self._identity(**identity),
            )
        )

    def text_delta(
        self,
        delta: str,
        *,
        content_blocks: tuple[Any, ...] | list[Any] | None = None,
        **identity: Any,
    ) -> None:
        blocks = tuple(content_blocks or ({"type": "text", "text": delta},))
        self.frontend.emit(
            MessageEvent(
                phase="content",
                text=delta,
                content_blocks=blocks,
                **self._identity(**identity),
            )
        )

    def reasoning_delta(
        self,
        delta: str,
        *,
        content_blocks: tuple[Any, ...] | list[Any] | None = None,
        **identity: Any,
    ) -> None:
        blocks = tuple(content_blocks or ({"type": "reasoning", "reasoning": delta},))
        self.frontend.emit(
            MessageEvent(
                phase="reasoning",
                text=delta,
                content_blocks=blocks,
                **self._identity(**identity),
            )
        )

    def discard_reasoning(self) -> None:
        self.frontend.emit(MessageEvent(phase="discard_reasoning", **self._identity()))

    def model_stream_finished(self, **identity: Any) -> None:
        self.frontend.emit(MessageEvent(phase="end", **self._identity(**identity)))

    def tool_call(self, name: str, args: Any, call_id: str = "", **identity: Any) -> None:
        self._tool("start", name, args=args, call_id=call_id, **identity)

    def tool_call_delta(self, name: str, args: Any, call_id: str = "", **identity: Any) -> None:
        self._tool("arguments_delta", name, args=args, call_id=call_id, **identity)

    def tool_call_updated(self, name: str, args: Any, call_id: str = "", **identity: Any) -> None:
        self._tool("update", name, args=args, call_id=call_id, **identity)

    def tool_call_approval_resolved(self, name: str, call_id: str = "") -> None:
        self._tool("approval_resolved", name, call_id=call_id)

    def tool_result(
        self,
        name: str,
        result: Any,
        call_id: str = "",
        *,
        duration_ms: int | None = None,
        **identity: Any,
    ) -> None:
        self._tool(
            "result",
            name,
            result=result,
            call_id=call_id,
            duration_ms=duration_ms,
            **identity,
        )

    def completed_tool_result(self, name: str, result: Any, call_id: str = "", **kwargs: Any) -> None:
        self._tool("completed_result", name, result=result, call_id=call_id, **kwargs)

    def completed_tool_error(self, name: str, error: Any, call_id: str = "", **kwargs: Any) -> None:
        self._tool("completed_error", name, result=error, call_id=call_id, **kwargs)

    def recovered_tool_call(self, name: str, args: Any, call_id: str = "", **kwargs: Any) -> None:
        self._tool("recovered_start", name, args=args, call_id=call_id, **kwargs)

    def recovered_tool_result(self, name: str, result: Any, call_id: str = "", **kwargs: Any) -> None:
        self._tool("recovered_result", name, result=result, call_id=call_id, **kwargs)

    def recovered_tool_error(self, name: str, error: Any, call_id: str = "", **kwargs: Any) -> None:
        self._tool("recovered_error", name, result=error, call_id=call_id, **kwargs)

    def delegation_started(self, calls: list[Any], **identity: Any) -> None:
        self.frontend.emit(
            ToolEvent(phase="delegation", calls=tuple(calls), **self._identity(**identity))
        )

    def delegation_delta(self, calls: list[Any], **identity: Any) -> None:
        metadata = dict(identity.pop("metadata", {}) or {})
        metadata["calls"] = tuple(calls)
        identity["metadata"] = metadata
        self.frontend.emit(
            SubagentEvent(
                phase="delegation_update",
                **self._identity(**identity),
            )
        )

    def stop_active_tools(self, status: str) -> None:
        self.frontend.emit(ToolEvent(phase="stop", status=status, **self._identity()))

    def _tool(
        self,
        phase: str,
        name: str,
        *,
        args: Any = None,
        result: Any = None,
        call_id: str = "",
        duration_ms: int | None = None,
        status: str = "",
        **identity: Any,
    ) -> None:
        self.frontend.emit(
            ToolEvent(
                phase=phase,  # type: ignore[arg-type]
                name=name,
                tool_call_id=call_id,
                arguments=args,
                result=result,
                duration_ms=duration_ms,
                status=status,
                **self._identity(**identity),
            )
        )

    def start_subagent_live(self) -> None:
        self.frontend.emit(SubagentEvent(phase="live_start", **self._identity()))

    def tick_subagents(self) -> None:
        self.frontend.emit(SubagentEvent(phase="live_tick", **self._identity()))

    def stop_subagent_live(self) -> None:
        self.frontend.emit(SubagentEvent(phase="live_stop", **self._identity()))

    def subagents_cancelled(self) -> None:
        self.frontend.emit(SubagentEvent(phase="cancel_all", **self._identity()))

    def subagent_label(self, subagent: Any) -> str:
        labeler = getattr(self.frontend, "subagent_label", None)
        if callable(labeler):
            return str(labeler(subagent))
        name = str(getattr(subagent, "name", "subagent") or "subagent")
        native_id = getattr(subagent, "id", None)
        return f"{name} [{native_id or id(subagent)}]"

    def subagent_started(self, name: str, task_input: str = "", **kwargs: Any) -> None:
        self._subagent("start", name, task_input=task_input, **kwargs)

    def subagent_request_updated(self, name: str, task_input: str) -> None:
        self._subagent("request_update", name, task_input=task_input)

    def subagent_finished(self, name: str, result: str = "", **kwargs: Any) -> None:
        self._subagent("finish", name, result=result, **kwargs)

    def subagent_cancelled(self, name: str, result: str = "", **kwargs: Any) -> None:
        self._subagent("cancel", name, result=result, **kwargs)

    def eval_subagent_started(self, name: str, task_input: str = "", **kwargs: Any) -> None:
        self._subagent("eval_start", name, task_input=task_input, **kwargs)

    def eval_subagent_finished(self, name: str, result: str = "", **kwargs: Any) -> None:
        self._subagent("eval_finish", name, result=result, **kwargs)

    def eval_subagent_cancelled(self, name: str, result: str = "", **kwargs: Any) -> None:
        self._subagent("eval_cancel", name, result=result, **kwargs)

    def _subagent(
        self,
        phase: str,
        name: str,
        *,
        task_input: str = "",
        result: str = "",
        origin: str = "",
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        label: str = "",
        duration_ms: int | None = None,
        **identity: Any,
    ) -> None:
        metadata = dict(identity.pop("metadata", {}) or {})
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        identity["metadata"] = metadata
        self.frontend.emit(
            SubagentEvent(
                phase=phase,  # type: ignore[arg-type]
                subagent_id=row_id or name,
                name=name,
                task_input=task_input,
                result=result,
                origin=origin,
                eval_id=eval_id,
                row_id=row_id,
                model=model,
                label=label,
                **self._identity(**identity),
            )
        )

    def rubric_evaluation_started(
        self,
        run_id: str,
        pass_number: int,
        max_iterations: int,
        *,
        grader_model: str = "",
        phase: RubricPhase = "verifying",
        **identity: Any,
    ) -> None:
        self.frontend.emit(
            RubricEvent(
                phase=phase,
                run_id=run_id,
                pass_number=pass_number,
                max_iterations=max_iterations,
                grader_model=grader_model,
                **self._identity(**identity),
            )
        )

    def rubric_lifecycle_event(self, event: Mapping[str, Any]) -> None:
        self.frontend.emit(
            RubricEvent(phase="lifecycle", lifecycle=dict(event), **self._identity())
        )

    def rubric_evaluation_finished(
        self,
        evaluation: Mapping[str, Any],
        max_iterations: int,
        **identity: Any,
    ) -> None:
        self.frontend.emit(
            RubricEvent(
                phase="finish",
                run_id=str(evaluation.get("grading_run_id") or ""),
                pass_number=int(evaluation.get("iteration") or 0) + 1,
                max_iterations=max_iterations,
                evaluation=dict(evaluation),
                **self._identity(**identity),
            )
        )

    def rubric_evaluations_cancelled(self) -> None:
        self.frontend.emit(RubricEvent(phase="cancel", **self._identity()))

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:
        self.frontend.emit(
            RubricEvent(
                phase="status",
                run_id=run_id,
                pass_number=pass_number,
                status=status,
                max_iterations=max_iterations,
                **self._identity(),
            )
        )

    def correction(self, event: Mapping[str, Any], **identity: Any) -> None:
        self.frontend.emit(
            InformationEvent(correction=dict(event), kind="correction", **self._identity(**identity))
        )

    def system_message(self, text: str, *, kind: str = "system", **identity: Any) -> None:
        self.frontend.emit(
            InformationEvent(text=text, kind=kind, **self._identity(**identity))
        )

    def usage_updated(self, usage: Mapping[str, Any] | None = None) -> None:
        self.frontend.emit(UsageEvent(usage=dict(usage or {}), **self._identity()))

    def compaction_started(self) -> None:
        self.frontend.emit(CompactionEvent(phase="start", **self._identity()))

    def compaction_finished(self) -> None:
        self.frontend.emit(CompactionEvent(phase="finish", **self._identity()))

    def waiting_started(self, label: str | None = None, **kwargs: Any) -> None:
        self.frontend.emit(
            RuntimeEvent(kind="waiting", state="start", detail={"label": label, **kwargs}, **self._identity())
        )

    def waiting_finished(self) -> None:
        self.frontend.emit(RuntimeEvent(kind="waiting", state="finish", **self._identity()))

    def finish_main(self) -> None:
        self.frontend.emit(RuntimeEvent(kind="message_group", state="finish", **self._identity()))

    def startup_progress(self, state: str) -> None:
        self.frontend.emit(RuntimeEvent(kind="startup", state=state, **self._identity()))

    def session_state(self, state: str, detail: Any = None) -> None:
        self.frontend.emit(
            RuntimeEvent(kind="session", state=state, detail=detail, **self._identity())
        )

    def artifact(
        self,
        artifact_type: ArtifactType,
        phase: ArtifactPhase,
        artifact: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
        **identity: Any,
    ) -> None:
        payload = dict(artifact) if artifact is not None else None
        self.frontend.emit(
            ArtifactEvent(
                artifact_type=artifact_type,
                phase=phase,
                artifact=payload,
                artifact_id=str((payload or {}).get("id") or ""),
                decision=dict(decision) if decision is not None else None,
                **self._identity(**identity),
            )
        )

    def mcp(self, phase: MCPPhase, *, server: str = "", detail: Any = None) -> None:
        self.frontend.emit(
            MCPEvent(phase=phase, server=server, detail=detail, **self._identity())
        )

    async def ask_approvals(self, interrupts: list[Any]) -> Any:
        return await self.frontend.request(ApprovalRequest(tuple(interrupts)))

    async def ask_user(self, interrupt: Any) -> Any:
        return await self.frontend.request(AskUserRequest(interrupt))

    async def finalize_goal(
        self,
        interrupt: Any,
        artifact: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self.frontend.request(ArtifactReviewRequest("goal", interrupt, artifact))

    async def finalize_plan(
        self,
        interrupt: Any,
        artifact: Mapping[str, Any] | None = None,
    ) -> Any:
        return await self.frontend.request(ArtifactReviewRequest("plan", interrupt, artifact))

    async def show_goal(self, interrupt: Any = None) -> Any:
        return await self.frontend.request(ArtifactDisplayRequest("goal", interrupt))

    async def show_plan(self, interrupt: Any = None) -> Any:
        return await self.frontend.request(ArtifactDisplayRequest("plan", interrupt))

    def context_notice_rendered(self) -> bool:
        return self._context_notice_rendered

    def mark_context_notice_rendered(self) -> None:
        self._context_notice_rendered = True


__all__ = ["FrontendEmitter"]
