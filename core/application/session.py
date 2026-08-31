"""Headless behavior for one MIRA session."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.planning.policy import PLANNING_STAGE_GOAL_RESEARCH, PLANNING_STAGE_PLAN_RESEARCH
from core.interface import FrontendEmitter, SessionSnapshot
from core.execution.turns import plan_thread_id, run_user_turn
from session.dashboard import normalize_dashboard, update_duration
from session.goals import clear_current_goal, current_goal, pause_current_goal, start_goal_attempt
from session.plans import clear_current_plan, current_plan, pause_current_plan, start_plan_attempt

if TYPE_CHECKING:
    from core.application.app import MiraApplication


class MiraSession:
    """One MIRA application session over native agents and graph threads."""

    def __init__(self, application: MiraApplication, record: dict[str, Any]) -> None:
        from core.application.app import initial_mode

        self.application = application
        self.record = record
        self.mode = initial_mode(
            application.agent,
            application.plan_agent,
            (application.config or {}).get("settings"),
            record,
        )
        self.runtime_state = "ready"
        self._active_task: asyncio.Task[Any] | None = None
        application._sessions[self.id] = self

    @property
    def id(self) -> str:
        return str(self.record.get("id") or "")

    async def prompt(self, text: str, **kwargs: Any) -> Any:
        """Execute one complete ACT/PLAN/Goal workflow through the MIRA API."""
        if self.runtime_state == "closed":
            raise RuntimeError("MIRA session is closed")
        if self._active_task is not None and not self._active_task.done():
            raise RuntimeError("MIRA session already has an active turn")
        if self.application.agent is None:
            raise RuntimeError(self.application.agent_unavailable_message or "Main model is not configured.")
        current = asyncio.current_task()
        self._active_task = current
        self.runtime_state = "running"
        emitter = FrontendEmitter(self.application.frontend, session_id=self.id)
        emitter.session_state("turn_started")
        model_name = str(kwargs.pop("model_name", self.application.model_name) or "")
        context_limit_tokens = kwargs.pop("context_limit_tokens", self.application.context_limit_tokens)
        context_limit_source = str(
            kwargs.pop("context_limit_source", self.application.context_limit_source) or "unknown"
        )
        try:
            return await run_user_turn(
                agent=self.application.agent,
                plan_agent=self.application.plan_agent,
                frontend=self.application.frontend,
                store=self.application.store,
                session=self.record,
                mode=self.mode,
                text=text,
                model_name=model_name,
                context_limit_tokens=context_limit_tokens,
                context_limit_source=context_limit_source,
                **kwargs,
            )
        except (Exception, asyncio.CancelledError):
            self.pause_active_artifacts()
            raise
        finally:
            self._active_task = None
            self.runtime_state = "ready"
            emitter.session_state("ready")

    async def cancel(self) -> None:
        """Cancel the active MIRA operation for this session."""
        task = self._active_task
        if task is None or task.done():
            return
        self.runtime_state = "cancelling"
        FrontendEmitter(self.application.frontend, session_id=self.id).session_state("cancelling")
        task.cancel()

    async def set_mode(self, mode: str) -> None:
        """Select ACT or PLAN without conflating the session and graph thread."""
        from core.application.app import select_mode

        select_mode(self.record, self.mode, mode)
        self.application.store.save(self.record)

    def begin_goal(self, objective: str) -> None:
        """Stage an explicit Goal request on an isolated planning thread."""
        self.mode["plan_runs"] = int(self.mode.get("plan_runs") or 0) + 1
        self.mode["goal_staging"] = {
            "authoritative_objective": objective,
            "objective": objective,
            "context_and_constraints": "",
            "research_evidence": "",
            "success_criteria": "",
            "stage": PLANNING_STAGE_GOAL_RESEARCH,
            "thread_id": plan_thread_id(self.record, self.mode["plan_runs"]),
            "replacement_confirmed": True,
        }

    def begin_artifact_revision(self, kind: str, artifact: dict[str, Any], feedback: str) -> None:
        """Stage revision of the exact retained formal artifact."""
        if kind == "plan":
            self.mode["planning"] = True
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.mode["plan_thread_id"] = plan_thread_id(self.record)
            self.mode["plan_revision"] = {"previous_plan": artifact, "feedback": feedback}
            return
        self.mode["plan_runs"] = int(self.mode.get("plan_runs") or 0) + 1
        self.mode["goal_revision"] = {"previous_goal": artifact, "feedback": feedback}
        self.mode["goal_staging"] = {
            "authoritative_objective": str(artifact.get("objective") or ""),
            "objective": str(artifact.get("objective") or ""),
            "context_and_constraints": "",
            "research_evidence": "",
            "success_criteria": "",
            "stage": PLANNING_STAGE_GOAL_RESEARCH,
            "thread_id": plan_thread_id(self.record, self.mode["plan_runs"]),
            "replacement_confirmed": True,
        }

    def start_artifact(self, kind: str) -> dict[str, Any] | None:
        """Start one explicit retained Goal/Plan execution attempt."""
        started = start_plan_attempt(self.record) if kind == "plan" else start_goal_attempt(self.record)
        self.mode[f"current_{kind}"] = started
        if started is None:
            return None
        self.mode[f"executing_{kind}"] = True
        self.mode[f"executing_{'goal' if kind == 'plan' else 'plan'}"] = False
        self.mode["planning"] = False
        if kind == "goal":
            self.mode["planning_stage"] = None
        self.application.store.save(self.record)
        FrontendEmitter(self.application.frontend, session_id=self.id).artifact(kind, "active", started)
        return started

    def clear_artifact(self, kind: str) -> dict[str, Any] | None:
        """Clear only the authoritative current artifact, retaining history."""
        value = clear_current_plan(self.record) if kind == "plan" else clear_current_goal(self.record)
        self.mode[f"current_{kind}"] = None
        self.application.store.save(self.record)
        if value is not None:
            FrontendEmitter(self.application.frontend, session_id=self.id).artifact(kind, "clear", value)
        return value

    def pause_active_artifacts(self) -> None:
        """Leave interrupted formal execution resumable in authoritative state."""
        if self.mode.get("executing_goal"):
            self.mode["current_goal"] = pause_current_goal(self.record)
            self.mode["executing_goal"] = False
        if self.mode.get("executing_plan"):
            self.mode["current_plan"] = pause_current_plan(self.record)
            self.mode["executing_plan"] = False
        self.application.store.save(self.record)

    def snapshot(self) -> SessionSnapshot:
        """Return the consumer projection of current authoritative state."""
        from core.application.app import available_tools, mcp_snapshot, normalize_resource_items

        resources = self.mode.get("resources") if isinstance(self.mode.get("resources"), dict) else {}
        mcp_state = mcp_snapshot(getattr(self.application, "mcp_manager", None))
        return SessionSnapshot(
            session_id=self.id,
            workspace=str(self.record.get("workspace") or self.application.workspace),
            mode="planning" if self.mode.get("planning") else "action",
            runtime_state=self.runtime_state,
            title=str(self.record.get("title") or "Untitled session"),
            turns=int(self.record.get("turns") or 0),
            current_goal=current_goal(self.record),
            current_plan=current_plan(self.record),
            transcript=tuple(dict(event) for event in self.record.get("events", ()) if isinstance(event, dict)),
            dashboard=dict(normalize_dashboard(self.record.get("dashboard"))),
            model={
                "name": self.application.model_name,
                "context_limit_tokens": self.application.context_limit_tokens,
                "context_limit_source": self.application.context_limit_source,
            },
            tools=tuple(available_tools(self.mode, planning=bool(self.mode.get("planning")))),
            resources={
                key: tuple(dict(item) for item in normalize_resource_items(value))
                for key, value in resources.items()
            },
            rubric={
                "enabled": bool(self.mode.get("rubric_enabled")),
                "max_iterations": int(self.mode.get("rubric_max_iterations") or 3),
            },
            mcp=mcp_state,
        )

    async def close(self, *, persist: bool = True) -> None:
        """Persist duration and close this consumer session projection."""
        if self.runtime_state == "closed":
            return
        task = self._active_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if persist:
            update_duration(self.record)
            self.application.store.save(self.record)
        self.runtime_state = "closed"
        FrontendEmitter(self.application.frontend, session_id=self.id).session_state("closed")


__all__ = ["MiraSession"]
