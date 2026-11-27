"""Headless behavior for one MIRA session."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent.planning.policy import PLANNING_STAGE_GOAL_RESEARCH, PLANNING_STAGE_PLAN_RESEARCH
from core.execution.handoff import synchronize_conversation
from core.execution.inspection.live import InspectionUpdate, LiveInspection, LiveInspectionStore
from core.execution.inspection.persistence import inspection_from_run
from core.execution.turns import plan_thread_id, run_user_turn
from core.interface import FrontendEmitter, SessionSnapshot
from core.interface.events import RuntimeEvent
from session.dashboard import ensure_dashboard, normalize_dashboard, update_duration
from session.goals import clear_current_goal, current_goal, pause_current_goal, start_goal_attempt
from session.plans import clear_current_plan, current_plan, pause_current_plan, start_plan_attempt
from session.subagent_runs import clear_runs

if TYPE_CHECKING:
    from core.application.app import MiraApplication


class MiraSession:
    """One MIRA application session over native agents and graph threads."""

    GRAPH_MODE_KEYS = (
        "planning", "executing_plan", "executing_goal", "plan_revision",
        "goal_staging", "goal_revision", "plan_counter",
        "plan_runs", "plan_thread_id", "rubric_enabled", "rubric_max_iterations",
        "pending_artifact",
        "goal_context_thread_id",
        "mode_transition",
        "turn_id",
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
        metadata = getattr(application, "resource_metadata", None)
        if isinstance(metadata, dict) and metadata:
            self.mode["resources"] = deepcopy(metadata)
        self.runtime_state = "ready"
        self._active_task: asyncio.Task[Any] | None = None
        self._transitioning = False
        self.recovery: dict[str, Any] | None = None
        self._inspections = LiveInspectionStore(self._inspection_updated)
        application._sessions[self.id] = self

    def inspection(self, inspection_id: str) -> LiveInspection | None:
        """Retrieve a detached live or historical transcript by inspection or run ID."""
        if self.runtime_state == "closed":
            return None
        live = self._inspections.get(inspection_id)
        if live is not None:
            return deepcopy(live)
        try:
            saved = self.application.store.read(self.application.store.path(self.id))
        except FileNotFoundError:
            return None
        runs = saved.get("runs", ())
        run = next((run for run in runs
                    if run.get("inspection_id") == inspection_id or run.get("id") == inspection_id), None)
        return deepcopy(inspection_from_run(run)) if run is not None else None

    def inspections(self, *, origin_event_id: int | None = None) -> list[dict[str, Any]]:
        """List detached durable runs, optionally for one parent transcript entry."""
        if self.runtime_state == "closed":
            return []
        return deepcopy([
            {key: value for key, value in run.items() if key != "events"}
            for run in self.record.get("runs", ())
            if origin_event_id is None or run.get("origin_event_id") == origin_event_id
        ])

    def _inspection_updated(self, inspection_id: str, update: InspectionUpdate) -> None:
        # Only the changed item crosses the event boundary, never the transcript.
        self.application.frontend.emit(RuntimeEvent(
            kind="inspection", state=update.operation, session_id=self.id,
            turn_id=str(self.mode.get("turn_id") or "") if self._active_task else "",
            detail={"inspection_id": inspection_id, "update": deepcopy(update)},
        ))

    def _clear_inspections(self) -> None:
        """Keep inspection cleanup coupled while transcript clearing migrates."""
        clear_runs(self.record)
        self._inspections.clear()

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
            transition = self.mode.get("mode_transition") or {}
            if transition.get("thread_id") == thread_id and not state.next and any(
                message.id == transition.get("message_id")
                for message in state.values.get("messages", [])
            ):
                # Completion can precede the semantic save during a crash.
                self.mode["mode_transition"] = None
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
                "turn_id": str(self.mode.get("turn_id") or ""),
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
        if self._transitioning or self.application._runtime_transition:
            raise RuntimeError("MIRA session or runtime is changing; retry after the transition")
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
        recovery = kwargs.get("recovery") or {}
        if recovery:
            for run in self.record.get("runs", ()):
                if run.get("inspection_id") and run.get("status") != "DONE":
                    inspection = inspection_from_run(run)
                    inspection.id = run["inspection_id"]
                    self._inspections.restore(deepcopy(inspection))
        self.mode["turn_id"] = str(recovery.get("turn_id") or uuid4().hex)
        emitter = FrontendEmitter(
            self.application.frontend, session_id=self.id, turn_id=self.mode["turn_id"],
        )
        model_name = str(kwargs.pop("model_name", self.application.model_name) or "")
        context_limit_tokens = kwargs.pop("context_limit_tokens", self.application.context_limit_tokens)
        context_limit_source = str(
            kwargs.pop("context_limit_source", self.application.context_limit_source) or "unknown"
        )
        kwargs.pop("continue_incomplete", None)
        turn_error: BaseException | None = None
        try:
            await self.application._refresh_model(self)
            # Save before invoking the graph so cold recovery keeps this interaction.
            await self.persist_graph_state()
            emitter.session_state("turn_started")
            result = await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                frontend=self.application.frontend,
                store=self.application.store,
                session=self.record,
                mode=self.mode,
                text=text,
                turn_id=self.mode["turn_id"],
                inspections=self._inspections,
                model_name=model_name,
                context_limit_tokens=context_limit_tokens,
                context_limit_source=context_limit_source,
                persist_always_allow=self.application.persist_tool_always_allow,
                persist_graph_state=self.persist_graph_state,
                synchronize_conversation=synchronize_conversation,
                **kwargs,
            )
        except (Exception, asyncio.CancelledError) as exc:
            from langchain_core.exceptions import ContextOverflowError
            from agent.middleware.context_overflow import context_notice_rendered, mark_context_notice_rendered, pop_context_overflow_notice

            if isinstance(exc, ContextOverflowError) and not context_notice_rendered(exc):
                emitter.system_message(pop_context_overflow_notice(exc), kind="info")
                mark_context_notice_rendered(exc)
            turn_error = exc
            with suppress(Exception):
                await self.restore_graph_state()
            self.pause_active_artifacts()
            if self.recovery is None:
                self.mode.update(goal_staging=None, goal_revision=None, plan_revision=None)
                if self.mode.get("planning"):
                    self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
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
        FrontendEmitter(
            self.application.frontend, session_id=self.id,
            turn_id=str(self.mode.get("turn_id") or ""),
        ).session_state("cancelling")
        task.cancel()

    async def set_mode(self, mode: str) -> None:
        """Select ACT or PLAN without conflating the session and graph thread."""
        from core.application.app import select_mode

        self._begin_transition()
        try:
            proposed = dict(self.mode)
            select_mode(self.record, proposed, mode)
            if proposed["planning"] != self.mode["planning"]:
                if self.agent is None or self.plan_agent is None:
                    raise RuntimeError("Both Act and Plan agents are required for a mode switch")
                source_agent, source_id = (
                    (self.plan_agent, plan_thread_id(self.record)) if self.mode["planning"]
                    else (self.agent, self.id)
                )
                target_agent, target_id = (
                    (self.plan_agent, plan_thread_id(self.record)) if proposed["planning"]
                    else (self.agent, self.id)
                )
                await synchronize_conversation(source_agent, source_id, target_agent, target_id)
            self.mode.update(proposed)
            await self.persist_graph_state()
            self.application.store.save(self.record)
        finally:
            self._transitioning = False

    def _begin_transition(self) -> None:
        if self.runtime_state == "closed":
            raise RuntimeError("MIRA session is closed")
        if self._transitioning or self.application._runtime_transition or (self._active_task is not None and not self._active_task.done()):
            raise RuntimeError("MIRA session already has an active operation")
        if self.recovery is not None:
            raise RuntimeError("Answer the pending checkpoint request before changing modes")
        self._transitioning = True

    async def begin_goal(self, objective: str) -> None:
        """Stage an explicit Goal request on an isolated planning thread."""
        self._begin_transition()
        try:
            if self.mode.get("goal_staging"):
                raise RuntimeError("Finish or clear the current Goal preparation before starting another")
            if self.agent is None or self.plan_agent is None:
                raise RuntimeError("Both Act and Plan agents are required for Goal preparation")
            run_id = int(self.mode.get("plan_runs") or 0) + 1
            thread_id = plan_thread_id(self.record, run_id)
            source_agent = self.plan_agent if self.mode["planning"] else self.agent
            source_id = plan_thread_id(self.record) if self.mode["planning"] else self.id
            await synchronize_conversation(source_agent, source_id, self.plan_agent, thread_id)
            self.mode["plan_runs"] = run_id
            self.mode["goal_staging"] = {
                "authoritative_objective": objective,
                "objective": objective,
                "context_and_constraints": "",
                "research_evidence": "",
                "success_criteria": "",
                "stage": PLANNING_STAGE_GOAL_RESEARCH,
                "thread_id": thread_id,
                "replacement_confirmed": True,
            }
            await self.persist_graph_state()
        finally:
            self._transitioning = False

    async def begin_artifact_revision(self, kind: str, artifact: dict[str, Any], feedback: str) -> None:
        """Stage revision of the exact retained formal artifact."""
        self._begin_transition()
        try:
            await self._begin_artifact_revision(kind, artifact, feedback)
        finally:
            self._transitioning = False

    async def _begin_artifact_revision(self, kind: str, artifact: dict[str, Any], feedback: str) -> None:
        if kind == "plan":
            if not self.mode["planning"]:
                await synchronize_conversation(self.agent, self.id, self.plan_agent, plan_thread_id(self.record))
                self.mode["mode_transition"] = {"mode": "plan"}
            self.mode["planning"] = True
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.mode["plan_thread_id"] = plan_thread_id(self.record)
            self.mode["plan_revision"] = {"previous_plan": artifact, "feedback": feedback}
            await self.persist_graph_state()
            return
        run_id = int(self.mode.get("plan_runs") or 0) + 1
        source_id = str(self.mode.get("goal_context_thread_id") or "")
        if source_id:
            source_agent = self.plan_agent
        elif self.mode["planning"]:
            source_agent, source_id = self.plan_agent, plan_thread_id(self.record)
        else:
            source_agent, source_id = self.agent, self.id
        thread_id = plan_thread_id(self.record, run_id)
        await synchronize_conversation(source_agent, source_id, self.plan_agent, thread_id)
        self.mode["plan_runs"] = run_id
        self.mode["goal_revision"] = {"previous_goal": artifact, "feedback": feedback}
        self.mode["goal_staging"] = {
            "authoritative_objective": str(artifact.get("objective") or ""),
            "objective": str(artifact.get("objective") or ""),
            "context_and_constraints": "",
            "research_evidence": "",
            "success_criteria": "",
            "stage": PLANNING_STAGE_GOAL_RESEARCH,
            "thread_id": thread_id,
            "replacement_confirmed": True,
        }
        await self.persist_graph_state()

    async def start_artifact(self, kind: str) -> dict[str, Any] | None:
        """Start one explicit retained Goal/Plan execution attempt."""
        self._begin_transition()
        try:
            return deepcopy(await self._start_artifact(kind))
        finally:
            self._transitioning = False

    async def _start_artifact(self, kind: str) -> dict[str, Any] | None:
        if (current_plan(self.record) if kind == "plan" else current_goal(self.record)) is None:
            return None
        if kind == "plan" and self.mode["planning"]:
            await synchronize_conversation(self.plan_agent, plan_thread_id(self.record), self.agent, self.id)
        if kind == "goal" and self.mode.get("goal_context_thread_id"):
            await synchronize_conversation(
                self.plan_agent, str(self.mode["goal_context_thread_id"]), self.agent, self.id,
            )
        started = start_plan_attempt(self.record) if kind == "plan" else start_goal_attempt(self.record)
        self.mode[f"current_{kind}"] = started
        if started is None:
            return None
        self.mode[f"executing_{kind}"] = True
        self.mode[f"executing_{'goal' if kind == 'plan' else 'plan'}"] = False
        if self.mode["planning"]:
            self.mode["mode_transition"] = {"mode": "act"}
        self.mode["planning"] = False
        if kind == "goal":
            self.mode["planning_stage"] = None
        await self.persist_graph_state()
        self.application.store.save(self.record)
        FrontendEmitter(self.application.frontend, session_id=self.id).artifact(kind, "active", started)
        return started

    async def clear_artifact(self, kind: str) -> dict[str, Any] | None:
        """Clear only the authoritative current artifact, retaining history."""
        self._begin_transition()
        try:
            return deepcopy(await self._clear_artifact(kind))
        finally:
            self._transitioning = False

    async def _clear_artifact(self, kind: str) -> dict[str, Any] | None:
        value = clear_current_plan(self.record) if kind == "plan" else clear_current_goal(self.record)
        self.mode[f"current_{kind}"] = None
        if kind == "goal":
            self.mode["goal_context_thread_id"] = None
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

    def snapshot(self, *, include_transcript: bool = True) -> SessionSnapshot:
        """Return the consumer projection of current authoritative state."""
        from core.application.app import available_tools, mcp_snapshot

        resources = self.mode.get("resources") if isinstance(self.mode.get("resources"), dict) else {}
        update_duration(self.record)
        mcp_state = mcp_snapshot(getattr(self.application, "mcp_manager", None))
        snapshot = SessionSnapshot(
            session_id=self.id,
            workspace=str(self.record.get("workspace") or self.application.workspace),
            mode="planning" if self.mode.get("planning") else "action",
            runtime_state=self.runtime_state,
            title=str(self.record.get("title") or "Untitled session"),
            turns=int(self.record.get("turns") or 0),
            current_goal=current_goal(self.record),
            current_plan=current_plan(self.record),
            transcript=tuple(event for event in self.record.get("events", ()) if isinstance(event, dict)) if include_transcript else (),
            dashboard=dict(normalize_dashboard(self.record.get("dashboard"))),
            model={
                "available": self.agent is not None and self.plan_agent is not None,
                "alias": self.application.effective_model_alias,
                "name": self.application.model_name,
                "context_limit_tokens": self.application.context_limit_tokens,
                "context_limit_source": self.application.context_limit_source,
            },
            tools=tuple(available_tools(self.mode, planning=bool(self.mode.get("planning")))),
            resources={
                key: tuple({field: str(item[field]) for field in (
                    "name", "path", "source", "replaces", "description", "group", "kind",
                    "source_model", "graph_id", "server", "uri", "original_name",
                ) if field in item}
                    for item in (value if isinstance(value, (list, tuple)) else ()) if isinstance(item, dict) and item.get("name"))
                for key, value in resources.items()
            },
            rubric={
                "enabled": bool(self.mode.get("rubric_enabled")),
                "max_iterations": int(self.mode.get("rubric_max_iterations") or 3),
            },
            mcp=mcp_state,
            pending=("incomplete" if self.recovery.get("incomplete") else "interaction") if self.recovery else "none",
            execution={
                "planning": bool(self.mode.get("planning")),
                "executing_plan": bool(self.mode.get("executing_plan")),
                "executing_goal": bool(self.mode.get("executing_goal")),
                "planning_stage": self.mode.get("planning_stage"),
                "goal_staging": bool(self.mode.get("goal_staging")),
                "goal_revision": bool(self.mode.get("goal_revision")),
                "plan_revision": bool(self.mode.get("plan_revision")),
            },
            history={key: self.record.get(key) for key in ("created_at", "updated_at", "custom_title", "pinned")},
        )
        return deepcopy(snapshot)

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
        self._inspections.clear()
        FrontendEmitter(self.application.frontend, session_id=self.id).session_state("closed")

    async def clear(self) -> None:
        """Clear this conversation's transcript, inspections and native threads."""
        self._begin_transition()
        try:
            if self.checkpointer is not None:
                for thread_id in (
                    self.id, f"{self.id}:plan",
                    *(f"{self.id}:plan:{index}" for index in range(1, int(self.mode.get("plan_runs") or 0) + 1)),
                ):
                    await self.checkpointer.adelete_thread(thread_id)
            self.record.update(title="Untitled session", turns=0, events=[], dashboard=normalize_dashboard(None))
            ensure_dashboard(self.record, model_name=self.application.model_name,
                             context_limit_tokens=self.application.context_limit_tokens,
                             context_limit_source=self.application.context_limit_source)
            self._clear_inspections()
            self.application.store.save(self.record)
            await self.persist_graph_state()
        finally:
            self._transitioning = False

    async def dismiss_artifact(self, kind: str) -> None:
        """Retain formal work while marking its historical controls closed."""
        from core.application.artifacts import update_artifact_event_status

        self._begin_transition()
        try:
            artifact = current_plan(self.record) if kind == "plan" else current_goal(self.record)
            if artifact is not None:
                update_artifact_event_status(self.record, kind, artifact["id"], "closed")
                self.application.store.save(self.record)
        finally:
            self._transitioning = False

    async def revise_artifact(self, kind: str, feedback: str) -> Any:
        """Revise the retained artifact through its native preparation pipeline."""
        from core.execution.turns import goal_revision_text, plan_revision_text

        artifact = current_plan(self.record) if kind == "plan" else current_goal(self.record)
        if artifact is None:
            return None
        await self.begin_artifact_revision(kind, artifact, feedback)
        instruction = plan_revision_text(artifact, feedback) if kind == "plan" else goal_revision_text(artifact, feedback)
        return await self.prompt(instruction, display_text=f"Revise {kind.title()}: {feedback}")

    async def create_goal(self, objective: str) -> Any:
        """Prepare and review an explicit Goal through the existing native flow."""
        await self.begin_goal(objective)
        return await self.prompt(objective, display_text=f"/goal {objective}")

    async def compact(self) -> Any:
        """Compact the selected native conversation without submitting a prompt."""
        from agent.middleware.compaction import PostTurnCompactionResult
        from core.execution.compaction import commit_forced_compaction
        from session.context import sync_deepagents_compaction

        self._begin_transition()
        agent = self.plan_agent if self.mode["planning"] else self.agent
        thread_id = str(self.mode.get("plan_thread_id") or plan_thread_id(self.record)) if self.mode["planning"] else self.id
        synchronize = False
        try:
            middleware = getattr(agent, "mira_compaction", None)
            if middleware is None:
                return PostTurnCompactionResult(reason="unavailable")
            snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
            if snapshot.next or snapshot.tasks:
                return PostTurnCompactionResult(reason="pending_work")
            plan = await middleware.aplan_forced_compaction_update(snapshot.values)
            result = await commit_forced_compaction(agent, thread_id, snapshot, plan)
            synchronize = result.compacted
            return result
        except Exception:
            synchronize = True
            raise
        finally:
            try:
                if synchronize and agent is not None:
                    with suppress(Exception):
                        if await sync_deepagents_compaction(self.record, agent, thread_id):
                            self.application.store.save(self.record)
            finally:
                self._transitioning = False

    async def context_report(self) -> Any:
        """Return the existing detached context accounting report for this session."""
        from core.context.report import (
            ContextReportMCPServer, ContextReportObservation,
            build_agent_context_report, mcp_tool_metadata,
        )
        from core.application.app import available_tools

        context = dict(normalize_dashboard(self.record.get("dashboard")).get("context") or {})
        observation = self.mode.get("context_report_observation")
        if not isinstance(observation, ContextReportObservation):
            observation = None
        planning = bool(self.mode.get("context_report_planning", self.mode.get("planning")))
        agent = self.mode.get("context_report_agent")
        if agent is None:
            agent = self.plan_agent if planning else self.agent
        thread_id = str(self.mode.get("context_report_thread_id") or (
            self.mode.get("plan_thread_id") if planning else self.id
        ) or "")
        manager = self.application.mcp_manager
        selected: dict[str, list[Any]] = {}
        if manager is not None:
            tools, _ = manager.tools_for_mode(self.application.config.get("settings"), planning=planning)
            for tool in tools:
                ownership = mcp_tool_metadata(tool)
                if ownership is not None:
                    selected.setdefault(str(ownership.get("server") or "unknown"), []).append(tool)
        servers = tuple(ContextReportMCPServer(
            name=str(state.name), status=str(state.status), error=str(state.error or ""),
            tools=tuple(selected.get(str(state.name), ())),
        ) for state in (manager.servers.values() if manager is not None else ()))
        return await build_agent_context_report(
            agent=agent, thread_id=thread_id, observation=observation,
            current_tokens=context.get("used_tokens"), limit_tokens=context.get("limit_tokens"),
            tool_metadata=tuple(dict(item) for item in available_tools(self.mode, planning=planning)),
            mcp_servers=servers,
        )

    async def run_workflow(self, invocation: str, *, inputs: dict[str, Any] | None = None, record_user: bool = True) -> Any:
        """Validate and execute an existing registered workflow command."""
        from agent.workflows.discovery import validate_runtime_graph, validate_workflow_input
        from core.execution.workflows import execute_workflow
        from session.workflows import PersistentWorkflowHistory

        self._begin_transition()
        try:
            demo = invocation in {"/workflow-demo", "/workflow-demo-phase2"}
            if invocation == "/workflow-demo":
                from core.execution.workflow_demo import build_workflow_demo

                graph = build_workflow_demo()
                inputs = inputs if inputs is not None else {"events": []}
                workflow_name = "workflow-demo"
            elif invocation == "/workflow-demo-phase2":
                from core.execution.workflow_demo_phase2 import build_workflow_demo_phase2

                child = self.application.workflows.agent(name="workflow-demo-agent", tools=[],
                    system_prompt="Reply in one short sentence confirming that the Workflow demo agent ran.")
                graph = build_workflow_demo_phase2(child)
                inputs = inputs if inputs is not None else {"messages": [{"role": "user", "content": "Confirm that the Workflow Phase 2 demo is running."}]}
                workflow_name = "workflow-demo-phase2"
            else:
                if inputs is None:
                    resolved = self.application.workflow_registry.resolve(invocation)
                    if resolved is None:
                        raise ValueError("Expected a registered workflow invocation")
                    spec, parsed = resolved
                else:
                    command = invocation.split(maxsplit=1)[0]
                    spec = self.application.workflow_registry.commands.get(command)
                    if spec is None:
                        raise ValueError(f"unknown workflow command: {command}")
                    parsed = inputs
                graph = spec.factory(self.application.workflows)
                model = validate_runtime_graph(spec, graph)
                inputs = inputs if inputs is not None else parsed
                validate_workflow_input(model, inputs, spec.usage)
                workflow_name = spec.name
        except BaseException:
            self._transitioning = False
            raise
        self._transitioning = False
        self._active_task = asyncio.current_task()
        self.runtime_state = "running"
        self.mode["turn_id"] = uuid4().hex
        emitter = FrontendEmitter(self.application.frontend, session_id=self.id, turn_id=self.mode["turn_id"])
        history = None
        try:
            if not demo:
                history = PersistentWorkflowHistory(self.application.frontend, self.record,
                    self.application.store, command=invocation, record_user=record_user)
                emitter = FrontendEmitter(history, session_id=self.id, turn_id=self.mode["turn_id"])
            emitter.session_state("turn_started")
            return await execute_workflow(
                graph, inputs, emitter=emitter, session=self,
                workflow_id=f"workflow-{workflow_name}:{uuid4()}", workflow_name=workflow_name,
                config={"configurable": {"thread_id": f"workflow:{workflow_name}:{uuid4()}"}},
                context=None if demo else self.application.workflows.context, action_agent=self.agent,
                inspection_title="workflow-demo-agent" if invocation == "/workflow-demo-phase2" else "",
                persist_always_allow=self.application.persist_tool_always_allow,
            )
        finally:
            if history is not None:
                history.close()
            self._active_task = None
            self.runtime_state = "ready"
            emitter.session_state("ready")

    async def upload_files(self, paths: list[Path]) -> tuple[list[str], list[str]]:
        """Import selected local files through the active project backend."""
        from core.application.uploads import ingest_local_files

        self._begin_transition()
        try:
            return await ingest_local_files(paths, self.application.project_backend, self.id)
        finally:
            self._transitioning = False

    async def clear_uploads(self) -> bool:
        """Remove this conversation's imported files through the project backend."""
        from core.application.uploads import clear_local_uploads

        self._begin_transition()
        try:
            return await clear_local_uploads(self.application.project_backend, self.id)
        finally:
            self._transitioning = False


__all__ = ["MiraSession"]
