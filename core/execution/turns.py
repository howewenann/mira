"""Headless orchestration for one semantic MIRA interaction."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage

from agent.middleware.context_overflow import mark_context_notice_rendered, pop_context_overflow_notice
from agent.planning.policy import (
    APPROVED_PLAN_EXECUTION_INSTRUCTIONS,
    OPTIONAL_RESEARCH_POLICY,
    PLAN_BEHAVIOR_POLICY,
    PLAN_BLOCKED_RESULT_MARKERS,
    PLAN_PROJECT_WRITE_TOOLS,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_RESEARCH,
    plan_disabled_tools_text,
)
from core.context.observation import context_usage_scope
from core.interface import Frontend, FrontendEmitter
from core.execution.runner import TurnResult, run_turn
from session.context import session_mcp_attachments, update_title, with_resume_context
from session.dashboard import apply_context_usage, apply_turn_usage, ensure_dashboard
from session.goals import current_goal, finish_goal_attempt, goal_artifact_text
from session.plans import current_plan, finish_plan_attempt, plan_artifact_text
from session.recorder import SessionEventEmitter, SessionRecorder, poll_compactions
from tracing.bootstrap import trace_user_turn


PLAN_CONTEXT_TEMPLATE = """Execute the exact retained Plan in Act mode.

<current_plan>
{plan}
</current_plan>

The Plan, Success Criteria, approved assumptions, and constraints are binding.
{execution_instructions}

Current instruction:
{text}"""

GOAL_CONTEXT_TEMPLATE = """Work toward the exact retained Goal in Act mode.

<current_goal>
{goal}
</current_goal>

The Objective and Success Criteria are binding.
Choose the implementation approach using available context and tools.
Report the result and evidence actually completed.

Current instruction:
{text}"""

PLAN_REVISION_TEMPLATE = """Revise this structured plan.

Current plan:
{plan}

User feedback:
{feedback}

The revised Plan must be a complete replacement. Preserve the existing outcome
and Success Criteria unless the feedback changes the required outcome, scope,
constraints, deliverables, or conditions of completion. Perform only necessary
read-only discovery, resolve required decisions through ask_user, then call
prepare_plan. Do not draft the final Plan in prose."""

GOAL_REVISION_TEMPLATE = """Revise this complete Goal through MIRA's read-only Goal pipeline.

The original user objective is authoritative.

<objective>
{objective}
</objective>

<previous_success_criteria>
{criteria}
</previous_success_criteria>

<user_feedback>
{feedback}
</user_feedback>

Perform only necessary read-only discovery, resolve required decisions through ask_user, then call prepare_goal. The revised Goal is a complete replacement and may change the Objective when the feedback changes the desired outcome. Do not create an implementation Plan or return the Goal in prose."""

EXPLICIT_GOAL_REQUEST_TEMPLATE = """You are handling an explicit Goal request through MIRA's read-only Goal pipeline.
This request is IMPLEMENTATION intent. Do not classify it as SAFE_CONVERSATION or answer with ordinary prose.
The following tools are disabled: {disabled_tools}.
Do not call a disabled tool or attempt the requested change.

{optional_research}

Authoritative Goal objective:
{text}

Treat that request as authoritative for meaning. For prepare_goal, write a concise user-facing Objective that may improve wording but must not add, remove, or materially change the intended outcome, scope, deliverables, or constraints. If a material user decision is required, call ask_user. Otherwise call prepare_goal as soon as the Goal is decision-complete. MIRA will generate Success Criteria and then require finalize_goal. Never create an implementation Plan."""

PLAN_REQUEST_TEMPLATE = """You are in planning mode (Plan mode).
The following tools are disabled: {disabled_tools}.
Do not call a disabled tool or attempt the requested change.
{behavior_policy}

User request:
{text}
"""


@trace_user_turn
async def run_user_turn(
    *,
    agent: Any,
    plan_agent: Any,
    frontend: Frontend,
    store: Any,
    session: dict[str, Any],
    mode: dict[str, Any],
    text: str,
    display_text: str | None = None,
    record_user: bool = True,
    model_name: str = "",
    context_limit_tokens: int | None = None,
    context_limit_source: str = "unknown",
    prepared_messages: list[Any] | None = None,
    attachments: list[dict[str, str]] | None = None,
    turn_runner: Any | None = None,
    rubric_override: str | None = None,
) -> TurnResult:
    """Run one submitted MIRA interaction across all immediate native phases."""
    emitter = FrontendEmitter(
        frontend,
        session_id=str(session.get("id") or ""),
        turn_id=str(int(session.get("turns") or 0) + 1),
    )
    live_usage_applied = False

    def apply_live_usage(usage: dict[str, Any]) -> None:
        nonlocal live_usage_applied
        apply_turn_usage(
            session,
            SimpleNamespace(usage=usage),
            model_name=model_name,
            context_limit_tokens=context_limit_tokens,
            context_limit_source=context_limit_source,
        )
        store.save(session)
        live_usage_applied = True
        emitter.usage_updated(usage)

    def apply_deepagents_context_usage(
        usage: dict[str, Any],
        *,
        phase_agent: Any,
        thread_id: str,
    ) -> None:
        observation = usage.get("context_report_observation")
        if observation is not None:
            mode["context_report_observation"] = observation
            mode["context_report_agent"] = phase_agent
            mode["context_report_thread_id"] = thread_id
            mode["context_report_planning"] = phase_agent is plan_agent
        if not usage.get("context_tokens"):
            return
        apply_context_usage(
            session,
            usage.get("context_tokens", 0),
            model_name=model_name,
            context_limit_tokens=context_limit_tokens,
            context_limit_source=context_limit_source,
            source=str(usage.get("context_source") or "unknown"),
        )
        store.save(session)
        emitter.usage_updated(usage)

    goal_staging = mode.get("goal_staging") if isinstance(mode.get("goal_staging"), dict) else None
    using_planning_agent = bool(mode["planning"] or goal_staging is not None)
    ensure_dashboard(
        session,
        model_name=model_name,
        context_limit_tokens=context_limit_tokens,
        context_limit_source=context_limit_source,
    )
    initial_mode_name = "planning" if mode.get("planning") else "action"
    recorder = SessionRecorder(session, store, initial_mode_name)
    visible_text = display_text if display_text is not None else text
    if record_user:
        user_event = recorder.user_message(visible_text, attachments=attachments)
        update_title(session)
        recorder.save()
        emitter.user_message(
            visible_text,
            planning=initial_mode_name == "planning",
            attachments=attachments,
            created_at=str(user_event.get("created_at") or ""),
        )

    all_attachments = session_mcp_attachments(session)
    aggregate = TurnResult()
    compact_targets: list[tuple[SessionRecorder, Any, str]] = []

    def invocation_messages(request_text: str, supplied: list[Any] | None = None) -> list[Any]:
        messages = list(supplied) if supplied is not None else [HumanMessage(content=request_text)]
        if all_attachments:
            for index, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    extra = dict(message.additional_kwargs)
                    extra["mira_mcp_attachments"] = all_attachments
                    messages[index] = message.model_copy(update={"additional_kwargs": extra})
                    break
        return messages

    async def run_phase(
        *,
        phase_agent: Any,
        request_text: str,
        thread_id: str,
        phase_mode: str,
        stage: str | None = None,
        state: dict[str, str] | None = None,
        rubric: str | None = None,
        rubric_iterations: int = 3,
        supplied_messages: list[Any] | None = None,
    ) -> TurnResult:
        phase_recorder = SessionRecorder(session, store, phase_mode)
        recording_emitter = SessionEventEmitter(
            emitter,
            phase_recorder,
            semantic_state=mode,
        )
        poller = asyncio.create_task(poll_compactions(phase_recorder, phase_agent, thread_id))
        try:
            with context_usage_scope(
                lambda usage: apply_deepagents_context_usage(
                    usage,
                    phase_agent=phase_agent,
                    thread_id=thread_id,
                )
            ):
                phase_result = await (turn_runner or run_turn)(
                    agent=phase_agent,
                    text=request_text,
                    renderer=recording_emitter,
                    thread_id=thread_id,
                    usage_callback=apply_live_usage,
                    rubric=rubric,
                    rubric_max_iterations=rubric_iterations,
                    include_rubric_state=(bool(mode.get("rubric_enabled")) or bool(rubric))
                    and stage is None,
                    planning_stage=stage,
                    planning_state=state,
                    planning_context=getattr(phase_agent, "mira_planning_context", None)
                    if stage is not None
                    else None,
                    messages=invocation_messages(request_text, supplied_messages),
                )
        except asyncio.CancelledError:
            recording_emitter.stop_active_tools("cancelled")
            await sync_compaction_safely(phase_recorder, phase_agent, thread_id)
            phase_recorder.interrupted("turn interrupted before completion")
            raise
        except ContextOverflowError as exc:
            recording_emitter.stop_active_tools("interrupted")
            await sync_compaction_safely(phase_recorder, phase_agent, thread_id)
            notice = pop_context_overflow_notice(exc)
            if notice and not recording_emitter.context_notice_rendered():
                phase_recorder.info(notice)
                emitter.system_message(notice, kind="info")
                recording_emitter.mark_context_notice_rendered()
            mark_context_notice_rendered(exc)
            raise
        except Exception as exc:
            recording_emitter.stop_active_tools("interrupted")
            await sync_compaction_safely(phase_recorder, phase_agent, thread_id)
            phase_recorder.system_error(f"turn error: {exc}")
            raise
        finally:
            poller.cancel()
            with suppress(asyncio.CancelledError):
                await poller
        phase_recorder.ensure_assistant(phase_result.final_text)
        compact_targets.append((phase_recorder, phase_agent, thread_id))
        merge_turn_results(aggregate, phase_result)
        return phase_result

    first_phase = True
    workflow = "goal" if goal_staging is not None else "plan" if using_planning_agent else ""
    existing_revision = mode.get("goal_revision") if workflow == "goal" else mode.get("plan_revision")
    previous_key = "previous_goal" if workflow == "goal" else "previous_plan"
    previous_artifact = (
        existing_revision.get(previous_key)
        if isinstance(existing_revision, dict)
        and isinstance(existing_revision.get(previous_key), dict)
        else None
    )
    feedback = str(existing_revision.get("feedback") or "").strip() if isinstance(existing_revision, dict) else ""
    phase_text = feedback if previous_artifact is not None and feedback else text
    result = aggregate

    while workflow:
        if workflow == "goal":
            authoritative = str(
                (goal_staging or {}).get("authoritative_objective")
                or (previous_artifact or {}).get("objective")
                or text
            )
            proposal = (
                explicit_goal_request_text(phase_text)
                if previous_artifact is None
                else goal_revision_text(previous_artifact, feedback)
            )
            stage = PLANNING_STAGE_GOAL_RESEARCH
            if first_phase:
                thread_id = str((goal_staging or {}).get("thread_id") or mode.get("plan_thread_id"))
            else:
                mode["plan_runs"] = int(mode.get("plan_runs") or 0) + 1
                thread_id = plan_thread_id(session, mode["plan_runs"])
        else:
            authoritative = str((previous_artifact or {}).get("objective") or text)
            proposal = (
                plan_request_text(phase_text)
                if previous_artifact is None
                else plan_revision_text(previous_artifact, feedback)
            )
            stage = PLANNING_STAGE_PLAN_RESEARCH
            thread_id = str(mode.get("plan_thread_id") or plan_thread_id(session))

        mode["planning_stage"] = stage
        planning_state = {
            "planning_authoritative_request": authoritative,
            "planning_previous_criteria": str((previous_artifact or {}).get("success_criteria") or ""),
            "planning_revision_feedback": feedback,
            "planning_previous_artifact": (
                plan_artifact_text(previous_artifact)
                if workflow == "plan" and previous_artifact is not None
                else goal_artifact_text(previous_artifact)
                if previous_artifact is not None
                else ""
            ),
        }
        request_text = with_resume_context(session, proposal)
        result = await run_phase(
            phase_agent=plan_agent,
            request_text=request_text,
            thread_id=thread_id,
            phase_mode="planning" if mode.get("planning") else "action",
            stage=stage,
            state=planning_state,
            supplied_messages=prepared_messages if first_phase else None,
        )
        first_phase = False
        review = result.formal_review
        if not isinstance(review, dict):
            break
        action = str(review.get("action") or "")
        if action != "revise":
            break
        candidate = review.get("artifact")
        if not isinstance(candidate, dict):
            raise RuntimeError("formal revision requires the reviewed provisional artifact")
        previous_artifact = candidate
        feedback = str(review.get("feedback") or "").strip()
        if not feedback:
            raise RuntimeError("formal revision requires explicit feedback")
        phase_text = feedback

    if not using_planning_agent or (
        isinstance(result.formal_review, dict)
        and result.formal_review.get("action") == "implement"
    ):
        retained_plan = current_plan(session) if mode.get("executing_plan") else None
        retained_goal = current_goal(session) if mode.get("executing_goal") and retained_plan is None else None
        approved_rubric = (
            str(retained_plan.get("success_criteria") or "")
            if retained_plan and retained_plan.get("rubric_enabled")
            else str(retained_goal.get("success_criteria") or "")
            if retained_goal and retained_goal.get("rubric_enabled")
            else None
        )
        action_text = action_request_text(
            mode,
            text,
            retained_goal=retained_goal,
            retained_plan=retained_plan,
        )
        request_text = with_resume_context(
            session,
            action_text,
            exclude_current_goal=retained_goal is not None,
        )
        result = await run_phase(
            phase_agent=agent,
            request_text=request_text,
            thread_id=str(session["id"]),
            phase_mode="action",
            rubric=rubric_override if rubric_override is not None else approved_rubric,
            rubric_iterations=int(
                (retained_goal or {}).get("rubric_iterations")
                or (retained_plan or {}).get("rubric_iterations")
                or mode.get("rubric_max_iterations")
                or 3
            ),
            supplied_messages=prepared_messages if first_phase else None,
        )

    session["turns"] = int(session.get("turns") or 0) + 1
    update_title(session)
    for phase_recorder, phase_agent, phase_thread_id in compact_targets:
        await sync_compaction_safely(phase_recorder, phase_agent, phase_thread_id)
    if not live_usage_applied:
        apply_turn_usage(
            session,
            aggregate,
            model_name=model_name,
            context_limit_tokens=context_limit_tokens,
            context_limit_source=context_limit_source,
        )
    if mode.get("executing_goal"):
        finish_goal_attempt(session, rubric_status=result.rubric_status)
        mode["current_goal"] = current_goal(session)
        mode["executing_goal"] = False
    if mode.get("executing_plan"):
        finish_plan_attempt(session, rubric_status=result.rubric_status)
        mode["current_plan"] = current_plan(session)
        mode["executing_plan"] = False
    store.save(session)
    return aggregate


def merge_turn_results(target: TurnResult, source: TurnResult) -> None:
    """Accumulate existing per-run results across phases of one MIRA turn."""
    target.final_text = source.final_text or target.final_text
    target.tool_calls.extend(source.tool_calls)
    target.tool_results.extend(source.tool_results)
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    target.total_tokens += source.total_tokens
    if source.context_tokens:
        target.context_tokens = source.context_tokens
        target.context_source = source.context_source
    if target.usage_source == "unknown" and source.usage_source != "unknown":
        target.usage_source = source.usage_source
    target.rubric_status = source.rubric_status or target.rubric_status
    target.rubric_evaluations.extend(source.rubric_evaluations)
    target.formal_review = source.formal_review or target.formal_review


async def sync_compaction_safely(recorder: SessionRecorder, agent: Any, thread_id: str) -> None:
    """Best-effort compaction sync for exception cleanup paths."""
    with suppress(Exception):
        await recorder.sync_compaction(agent, thread_id)


def plan_thread_id(session: dict[str, Any], run_id: int | None = None) -> str:
    """Return the LangGraph thread id used for planning-mode memory."""
    return f"{session['id']}:plan" if run_id is None else f"{session['id']}:plan:{run_id}"


def plan_command_prompt(text: str) -> str | None:
    """Return the normal message suffix for `/plan [prompt]`, or None otherwise."""
    if text == "/plan":
        return ""
    if not text.startswith("/plan "):
        return None
    return text[len("/plan"):].strip()


def has_clean_plan(result: TurnResult) -> bool:
    """Return whether a planning result is safe to reuse in action mode."""
    final_text = getattr(result, "final_text", "").strip()
    return bool(
        final_text
        and not any(marker in final_text.lower() for marker in PLAN_BLOCKED_RESULT_MARKERS)
        and not write_tool_was_used(result)
        and not write_was_blocked(result)
    )


def write_tool_was_used(result: TurnResult) -> bool:
    return bool(set(PLAN_PROJECT_WRITE_TOOLS).intersection(getattr(result, "tool_calls", [])))


def write_was_blocked(result: TurnResult) -> bool:
    tool_results = getattr(result, "tool_results", [])
    return any(marker in value.lower() for value in tool_results for marker in PLAN_BLOCKED_RESULT_MARKERS)


def plan_request_text(text: str) -> str:
    return PLAN_REQUEST_TEMPLATE.format(
        disabled_tools=plan_disabled_tools_text(),
        behavior_policy=PLAN_BEHAVIOR_POLICY,
        text=text,
    )


def plan_revision_text(plan: dict[str, Any], feedback: str) -> str:
    return PLAN_REVISION_TEMPLATE.format(plan=plan_artifact_text(plan), feedback=feedback.strip())


def goal_revision_text(value: dict[str, Any], feedback: str) -> str:
    return GOAL_REVISION_TEMPLATE.format(
        objective=str(value.get("objective") or "").strip(),
        criteria=str(value.get("success_criteria") or "").strip(),
        feedback=feedback.strip(),
    )


def action_request_text(
    mode: dict[str, Any],
    text: str,
    *,
    retained_goal: dict[str, Any] | None = None,
    retained_plan: dict[str, Any] | None = None,
) -> str:
    if isinstance(retained_plan, dict):
        return PLAN_CONTEXT_TEMPLATE.format(
            plan=plan_artifact_text(retained_plan),
            execution_instructions=APPROVED_PLAN_EXECUTION_INSTRUCTIONS,
            text=text,
        )
    if isinstance(retained_goal, dict):
        return GOAL_CONTEXT_TEMPLATE.format(goal=goal_artifact_text(retained_goal), text=text)
    return text


def explicit_goal_request_text(text: str) -> str:
    return EXPLICIT_GOAL_REQUEST_TEMPLATE.format(
        disabled_tools=plan_disabled_tools_text(),
        optional_research=OPTIONAL_RESEARCH_POLICY,
        text=text,
    )


__all__ = [
    "action_request_text",
    "explicit_goal_request_text",
    "goal_revision_text",
    "has_clean_plan",
    "merge_turn_results",
    "plan_command_prompt",
    "plan_request_text",
    "plan_revision_text",
    "plan_thread_id",
    "run_user_turn",
    "sync_compaction_safely",
]
