"""Headless behavior for one MIRA session."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from contextlib import suppress
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

    GRAPH_MODE_KEYS = (
        "planning", "executing_plan", "executing_goal", "plan_revision",
        "goal_staging", "goal_revision", "plan_counter",
        "plan_runs", "plan_thread_id", "rubric_enabled", "rubric_max_iterations",
        "pending_artifact",
    )

    def __init__(
        self, application: MiraApplication, record: dict[str, Any],
        *, agent: Any = None, plan_agent: Any = None, checkpointer: Any = None,
        semantic_graph: Any = None,
    ) -> None:
        from core.application.app import initial_mode

        self.application = application
        self.record = record
        self.agent = agent if agent is not None else application.agent
        self.plan_agent = plan_agent if plan_agent is not None else application.plan_agent
        self.checkpointer = checkpointer
        self.semantic_graph = semantic_graph
        self.mode = initial_mode(
            self.agent,
            self.plan_agent,
            (application.config or {}).get("settings"),
            record,
        )
        self.runtime_state = "ready"
        self._active_task: asyncio.Task[Any] | None = None
        self.recovery: dict[str, Any] | None = None
        application._sessions[self.id] = self

    async def restore_graph_state(self) -> None:
        """Restore application semantics and discover saved graph work without running it."""
        if self.semantic_graph is None:
            return
        snapshot = await self.semantic_graph.aget_state(
            {"configurable": {"thread_id": f"{self.id}:session"}}
        )
        saved = snapshot.values.get("mira_session") if isinstance(snapshot.values, dict) else None
        if isinstance(saved, dict):
            self.record["current_plan"] = deepcopy(saved.get("current_plan"))
            self.record["current_goal"] = deepcopy(saved.get("current_goal"))
            for key in self.GRAPH_MODE_KEYS:
                if key in saved:
                    self.mode[key] = deepcopy(saved[key])
            self.mode["current_plan"] = current_plan(self.record)
            self.mode["current_goal"] = current_goal(self.record)
        else:
            # Without a semantic checkpoint there is no retained formal artifact.
            self.record["current_plan"] = None
            self.record["current_goal"] = None
            self.mode["current_plan"] = None
            self.mode["current_goal"] = None
        self.recovery = None
        for agent, thread_id in (
            (self.agent, self.id),
            (self.plan_agent, str(self.mode.get("plan_thread_id") or plan_thread_id(self.record))),
            (self.plan_agent, str((self.mode.get("goal_staging") or {}).get("thread_id") or "")),
        ):
            if agent is None or not hasattr(agent, "aget_state") or not thread_id:
                continue
            state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
            if agent is self.plan_agent and isinstance(state.values, dict):
                stage = state.values.get("planning_stage")
                if isinstance(stage, str):
                    self.mode["planning_stage"] = stage
            if state.config is None or not state.next:
                continue
            interrupts = [interrupt for task in state.tasks for interrupt in task.interrupts]
            self.recovery = {
                "agent": agent, "thread_id": thread_id,
                "interrupts": interrupts, "incomplete": not bool(interrupts),
            }
            break
        if self.recovery is None and (
            self.mode.get("executing_plan") or self.mode.get("executing_goal")
        ):
            self.pause_active_artifacts()
            await self.persist_graph_state()

    async def persist_graph_state(self) -> None:
        """Commit one semantic transition before exposing it as successful."""
        if self.semantic_graph is None:
            return
        saved = {key: deepcopy(self.mode.get(key)) for key in self.GRAPH_MODE_KEYS}
        saved["current_plan"] = current_plan(self.record)
        saved["current_goal"] = current_goal(self.record)
        await self.semantic_graph.aupdate_state(
            {"configurable": {"thread_id": f"{self.id}:session"}},
            {"mira_session": saved},
            as_node="save",
        )

    @property
    def id(self) -> str:
        return str(self.record.get("id") or "")

    async def prompt(self, text: str, **kwargs: Any) -> Any:
        """Execute one complete ACT/PLAN/Goal workflow through the MIRA API."""
        if self.runtime_state == "closed":
            raise RuntimeError("MIRA session is closed")
        if self._active_task is not None and not self._active_task.done():
            raise RuntimeError("MIRA session already has an active turn")
        if self.recovery is not None:
            if self.recovery["incomplete"]:
                if not kwargs.get("continue_incomplete"):
                    raise RuntimeError(
                        "The previous operation stopped before a checkpointed result. "
                        "Its external side effects are uncertain; explicitly continue or start a new session."
                    )
            if "recovery" not in kwargs:
                raise RuntimeError("Answer the pending request before submitting another message.")
        if self.agent is None:
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
        kwargs.pop("continue_incomplete", None)
        turn_error: BaseException | None = None
        try:
            result = await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                frontend=self.application.frontend,
                store=self.application.store,
                session=self.record,
                mode=self.mode,
                text=text,
                model_name=model_name,
                context_limit_tokens=context_limit_tokens,
                context_limit_source=context_limit_source,
                persist_always_allow=self.application.persist_tool_always_allow,
                persist_graph_state=self.persist_graph_state,
                **kwargs,
            )
        except (Exception, asyncio.CancelledError) as exc:
            turn_error = exc
            with suppress(Exception):
                await self.restore_graph_state()
            self.pause_active_artifacts()
            await self.persist_graph_state()
            raise
        else:
            await self.persist_graph_state()
            return result
        finally:
            try:
                await self.application.refresh_agents_for_policy()
            except Exception as refresh_error:
                if turn_error is not None:
                    emitter.system_message(
                        f"tool policy refresh failed: {refresh_error}", kind="error"
                    )
                else:
                    raise
            finally:
                self._active_task = None
                self.runtime_state = "ready"
                emitter.session_state("ready")

    async def resume_pending(self) -> Any:
        """Redisplay a checkpointed interactive request and resume after its answer."""
        recovery = self.recovery
        if recovery is None or recovery["incomplete"]:
            return None
        result = await self.prompt("", record_user=False, recovery=recovery)
        self.recovery = None
        return result

    async def continue_incomplete(self) -> Any:
        """Explicitly retry a saved incomplete graph, which may repeat side effects."""
        if self.recovery is None or not self.recovery["incomplete"]:
            return None
        result = await self.prompt(
            "", record_user=False, recovery=self.recovery, continue_incomplete=True
        )
        self.recovery = None
        return result

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
        await self.persist_graph_state()
        self.application.store.save(self.record)

    async def begin_goal(self, objective: str) -> None:
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
        await self.persist_graph_state()

    async def begin_artifact_revision(self, kind: str, artifact: dict[str, Any], feedback: str) -> None:
        """Stage revision of the exact retained formal artifact."""
        if kind == "plan":
            self.mode["planning"] = True
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.mode["plan_thread_id"] = plan_thread_id(self.record)
            self.mode["plan_revision"] = {"previous_plan": artifact, "feedback": feedback}
            await self.persist_graph_state()
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
        await self.persist_graph_state()

    async def start_artifact(self, kind: str) -> dict[str, Any] | None:
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
        await self.persist_graph_state()
        self.application.store.save(self.record)
        FrontendEmitter(self.application.frontend, session_id=self.id).artifact(kind, "active", started)
        return started

    async def clear_artifact(self, kind: str) -> dict[str, Any] | None:
        """Clear only the authoritative current artifact, retaining history."""
        value = clear_current_plan(self.record) if kind == "plan" else clear_current_goal(self.record)
        self.mode[f"current_{kind}"] = None
        await self.persist_graph_state()
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
        if self.checkpointer is not None:
            await self.checkpointer.conn.close()
        self.runtime_state = "closed"
        FrontendEmitter(self.application.frontend, session_id=self.id).session_state("closed")


__all__ = ["MiraSession"]
