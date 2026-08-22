"""Interactive-mode state and slash-command helpers."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ui.command_help import COMMAND_HELP_SECTIONS

from agent.context_overflow import mark_context_notice_rendered, pop_context_overflow_notice
from agent.planning.policy import (
    APPROVED_PLAN_EXECUTION_INSTRUCTIONS,
    FINALIZE_GOAL_TOOL,
    FINALIZE_PLAN_TOOL,
    PLAN_BEHAVIOR_POLICY,
    PLAN_BLOCKED_RESULT_MARKERS,
    PLAN_DISABLED_TOOLS,
    PLAN_PROJECT_WRITE_TOOLS,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_GOAL_FINALIZE,
    PLANNING_STAGE_PLAN_FINALIZE,
    PLANNING_STAGE_PLAN_RESEARCH,
    OPTIONAL_RESEARCH_POLICY,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
    SHOW_GOAL_TOOL,
    SHOW_PLAN_TOOL,
    plan_disabled_tools_text,
)
from agent.tools.specs import mira_environment_label
from runtime.context_usage import context_usage_scope
from runtime.runner import TurnResult, run_turn
from session.dashboard import apply_context_usage, apply_turn_usage, ensure_dashboard
from session.context import mark_resume_context_pending, update_title, with_resume_context
from session.goals import current_goal, finish_goal_attempt, goal_artifact_text
from session.plans import (
    current_plan,
    finish_plan_attempt,
    plan_artifact_text,
)
from session.recorder import RecordingRenderer, SessionRecorder, call_renderer, poll_compactions
from config.settings import rubric_enabled, rubric_max_iterations
from tracing.bootstrap import trace_user_turn
from ui.runtime_snapshot import resources_table, tools_table

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

HELP_SECTION_STYLE = "bold #7aa2f7"

DEFAULT_TOOL_SPECS = [
    {
        "name": "ask_user",
        "description": "Ask the user to choose between concrete next steps when MIRA is blocked.",
    },
    {
        "name": "finalize_plan",
        "description": "Finalize a structured implementation Plan for explicit user review.",
    },
    {
        "name": "prepare_plan",
        "description": "Begin criteria-first construction of a decision-complete Plan.",
    },
    {
        "name": "prepare_goal",
        "description": "Begin criteria-first construction of a decision-complete Goal.",
    },
    {
        "name": "finalize_goal",
        "description": "Finalize a Goal title after Success Criteria generation.",
    },
    {
        "name": "show_plan",
        "description": "Render the exact retained current Plan.",
    },
    {
        "name": "show_goal",
        "description": "Render the exact retained current Goal.",
    },
    {"name": "write_todos", "description": ""},
    {"name": "ls", "description": ""},
    {"name": "read_file", "description": ""},
    {"name": "write_file", "description": ""},
    {"name": "edit_file", "description": ""},
    {"name": "glob", "description": ""},
    {"name": "grep", "description": ""},
    {"name": "eval", "description": ""},
    {"name": "task", "description": ""},
]


def initial_mode(
    agent: Any,
    plan_agent: Any,
    settings: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the mutable interactive state for one TUI session."""
    return {
        "planning": False,
        "current_plan": current_plan(session or {}),
        "executing_plan": False,
        "current_goal": current_goal(session or {}),
        "executing_goal": False,
        "plan_revision": None,
        "goal_staging": None,
        "goal_revision": None,
        "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
        "rubric_enabled": rubric_enabled(settings),
        "rubric_max_iterations": rubric_max_iterations(settings),
        "plan_counter": 0,
        "plan_runs": 0,
        "plan_thread_id": plan_thread_id(session) if session else "",
        "action_tools": tool_specs(agent),
        "planning_tools": tool_specs(plan_agent),
        "resources": resource_specs(agent),
    }


def refresh_agent_specs(mode: dict[str, Any], agent: Any, plan_agent: Any) -> None:
    """Refresh tool/resource metadata after agents are rebuilt."""
    mode["action_tools"] = tool_specs(agent)
    mode["planning_tools"] = tool_specs(plan_agent)
    mode["resources"] = resource_specs(agent)


@trace_user_turn
async def run_user_turn(
    *,
    agent: Any,
    plan_agent: Any,
    renderer: Any,
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
) -> TurnResult:
    """Run one submitted MIRA interaction across all immediate internal phases."""
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
        usage_updated = getattr(renderer, "usage_updated", None)
        if callable(usage_updated):
            usage_updated()

    def apply_deepagents_context_usage(usage: dict[str, Any]) -> None:
        apply_context_usage(
            session,
            usage.get("context_tokens", 0),
            model_name=model_name,
            context_limit_tokens=context_limit_tokens,
            context_limit_source=context_limit_source,
            source=str(usage.get("context_source") or "unknown"),
        )
        store.save(session)
        usage_updated = getattr(renderer, "usage_updated", None)
        if callable(usage_updated):
            usage_updated()

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
        user_renderer = getattr(renderer, "user_message", None)
        if callable(user_renderer):
            call_renderer(
                user_renderer,
                visible_text,
                planning=initial_mode_name == "planning",
                created_at=str(user_event.get("created_at") or ""),
            )
    from langchain_core.messages import HumanMessage
    from session.context import session_mcp_attachments

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
        wrapped_renderer = RecordingRenderer(renderer, phase_recorder)
        poller = asyncio.create_task(poll_compactions(phase_recorder, phase_agent, thread_id))
        try:
            with context_usage_scope(apply_deepagents_context_usage):
                phase_result = await run_turn(
                    agent=phase_agent,
                    text=request_text,
                    renderer=wrapped_renderer,
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
            wrapped_renderer.stop_active_tools("cancelled")
            await sync_compaction_safely(phase_recorder, phase_agent, thread_id)
            phase_recorder.interrupted("turn interrupted before completion")
            raise
        except ContextOverflowError as exc:
            wrapped_renderer.stop_active_tools("interrupted")
            await sync_compaction_safely(phase_recorder, phase_agent, thread_id)
            notice = pop_context_overflow_notice(exc)
            if notice and not wrapped_renderer.context_notice_rendered():
                phase_recorder.info(notice)
                write_line(renderer, notice, kind="info")
                wrapped_renderer.mark_context_notice_rendered()
            mark_context_notice_rendered(exc)
            raise
        except Exception as exc:
            wrapped_renderer.stop_active_tools("interrupted")
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
    existing_revision = (
        mode.get("goal_revision") if workflow == "goal" else mode.get("plan_revision")
    )
    previous_key = "previous_goal" if workflow == "goal" else "previous_plan"
    previous_artifact = (
        existing_revision.get(previous_key)
        if isinstance(existing_revision, dict)
        and isinstance(existing_revision.get(previous_key), dict)
        else None
    )
    feedback = (
        str(existing_revision.get("feedback") or "").strip()
        if isinstance(existing_revision, dict)
        else ""
    )
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
            "planning_previous_criteria": str(
                (previous_artifact or {}).get("success_criteria") or ""
            ),
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
            workflow = ""
            break
        action = str(review.get("action") or "")
        if action == "revise":
            candidate = review.get("artifact")
            if not isinstance(candidate, dict):
                raise RuntimeError("formal revision requires the reviewed provisional artifact")
            previous_artifact = candidate
            feedback = str(review.get("feedback") or "").strip()
            if not feedback:
                raise RuntimeError("formal revision requires explicit feedback")
            phase_text = feedback
            continue
        if action == "implement":
            workflow = ""
            break
        workflow = ""
        break

    if not using_planning_agent or (
        isinstance(result.formal_review, dict)
        and result.formal_review.get("action") == "implement"
    ):
        retained_plan = current_plan(session) if mode.get("executing_plan") else None
        retained_goal = (
            current_goal(session) if mode.get("executing_goal") and retained_plan is None else None
        )
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
            rubric=approved_rubric,
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


async def handle_command(
    text: str,
    renderer: Any,
    session: dict[str, Any],
    model_name: str,
    mode: dict[str, Any] | None = None,
) -> bool:
    """Handle slash commands and return whether the input was consumed."""
    if not text.startswith("/"):
        return False

    mode = mode if mode is not None else {"planning": False}

    if text in {"/exit", "/quit"}:
        write_line(renderer, "bye", kind="muted")
        return True

    if text == "/help":
        print_help(renderer)
        return True

    if text == "/mira":
        callback = getattr(renderer, "show_mira_splash", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "MIRA splash is unavailable in this interface.", kind="warning")
        return True

    if text == "/tools":
        print_tools(renderer, mode)
        return True

    if text == "/memories":
        print_resources(renderer, "Memories", resources_for(mode, "memories"))
        return True

    if text == "/skills":
        print_resources(renderer, "Skills", resources_for(mode, "skills"))
        return True

    if text == "/subagents":
        print_resources(renderer, "Subagents", resources_for(mode, "subagents"))
        return True

    if text == "/plan" or (text.startswith("/plan ") and not text[len("/plan"):].strip()):
        mode["planning"] = True
        mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
        mode["plan_thread_id"] = plan_thread_id(session)
        mark_resume_context_pending(session, resumed=True)
        write_line(
            renderer,
            f"Plan mode: {plan_disabled_tools_text()} disabled; use /act to leave",
            kind="status",
        )
        return True

    if text == "/plan-show":
        callback = getattr(renderer, "show_plan", None)
        if callable(callback):
            outcome = callback(None)
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Plan display is unavailable in this interface.", kind="warning")
        return True

    if text == "/plan-clear":
        callback = getattr(renderer, "clear_plan", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Plan clearing is unavailable in this interface.", kind="warning")
        return True

    if text == "/plan-resume":
        callback = getattr(renderer, "resume_plan", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Plan resume is unavailable in this interface.", kind="warning")
        return True

    if text == "/goal-show":
        callback = getattr(renderer, "show_goal", None)
        if callable(callback):
            outcome = callback(None)
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Goal display is unavailable in this interface.", kind="warning")
        return True

    if text == "/goal-clear":
        callback = getattr(renderer, "clear_goal", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Goal clearing is unavailable in this interface.", kind="warning")
        return True

    if text == "/goal-resume":
        callback = getattr(renderer, "resume_goal", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Goal resume is unavailable in this interface.", kind="warning")
        return True

    if text == "/act":
        mode["planning"] = False
        mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
        write_line(renderer, "action mode", kind="status")
        return True

    if text == "/clear":
        clear(renderer)
        return True

    if text in {"/clear-chat", "/clear-all-chats", "/clear-errors", "/clear-prompts"}:
        write_line(renderer, f"{text} is available in the Textual app with confirmation", kind="warning")
        return True

    if text == "/session":
        write_line(renderer, session_summary_text(session, mode))
        return True

    if text in {"/reload", "/reload-runtime"}:
        write_line(renderer, f"{text} is available in the Textual app", kind="warning")
        return True

    if text == "/issues":
        write_line(renderer, "/issues is available in the Textual app", kind="warning")
        return True

    if text == "/runtime":
        write_line(renderer, "/runtime is available in the Textual app", kind="warning")
        return True

    if text == "/mcp":
        write_line(renderer, "/mcp is available in the Textual app", kind="warning")
        return True

    if text == "/prompts":
        write_line(renderer, "/prompts is available in the Textual app", kind="warning")
        return True

    if text == "/compact":
        write_line(renderer, "/compact is available in the Textual app", kind="warning")
        return True

    if text == "/new-chat":
        write_line(renderer, "/new-chat is available in the Textual app", kind="warning")
        return True

    write_line(renderer, f"unknown command: {text}", kind="muted")
    return True


def print_help(renderer: Any) -> None:
    """Print the compact interactive reference as one output block."""
    write_renderable(renderer, help_report())


def session_summary_text(session: dict[str, Any], mode: dict[str, Any]) -> str:
    """Return session details as one command output block."""
    has_goal = current_goal(session) is not None
    has_plan = current_plan(session) is not None or bool(mode.get("current_plan"))
    return "\n".join(
        [
            f"session: {session['id']}",
            f"title: {session.get('title', 'Untitled session')}",
            f"mode: {'planning' if mode['planning'] else 'action'}",
            f"current goal: {'yes' if has_goal else 'no'}",
            f"current plan: {'yes' if has_plan else 'no'}",
            f"workspace: {session['workspace']}",
            f"turns: {session['turns']}",
        ]
    )


def key_bindings_table() -> Table:
    """Build the useful global key-binding reference."""
    table = Table(title="Key bindings", title_style="bold cyan", expand=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Action")
    table.add_row("Shift+Enter", "Insert a newline")
    table.add_row("Ctrl+C", "Copy selected text")
    table.add_row("Ctrl+L", "Clear the chat display")
    table.add_row("Alt+Q", "Cancel active work or quit")
    return table


def autocomplete_table() -> Table:
    """Build the trigger and insertion reference for autocomplete."""
    table = Table(title="Autocomplete", title_style="bold cyan", expand=True)
    table.add_column("Trigger", style="cyan", no_wrap=True)
    table.add_column("Finds")
    table.add_column("Selection")
    table.add_row(
        "/",
        "CMND commands and PRMT prompts",
        "Inserts the command without a trailing space",
    )
    table.add_row(
        "@",
        "FILE files, RSRC resources, TOOL tools and SUBA subagents",
        "Files/resources keep @; tools/subagents remove it",
    )
    return table


def usage_notes_table() -> Table:
    """Build reusable guidance for important user-facing conventions."""
    table = Table(title="Usage notes", title_style="bold cyan", expand=True)
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Note")
    table.add_row(
        "Prompt arguments",
        escape(
            "Required-only prompts use positional values; prompts with any [optional] "
            "argument use name=value for every argument."
        ),
    )
    return table


def help_report() -> Group:
    """Build all help sections in their user-facing order."""
    return Group(
        key_bindings_table(),
        Text(""),
        autocomplete_table(),
        Text(""),
        usage_notes_table(),
        Text(""),
        help_table(),
    )


def help_table() -> Table:
    """Build one Rich help table grouped by command purpose."""
    table = Table(title="Commands", title_style="bold cyan", expand=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    for section, commands in COMMAND_HELP_SECTIONS:
        table.add_row(Text(section, style=HELP_SECTION_STYLE), "")
        for index, (command, description) in enumerate(commands):
            table.add_row(escape(command), escape(description), end_section=index == len(commands) - 1)
    return table


def print_tools(renderer: Any, mode: dict[str, Any]) -> None:
    """Print tools available in the current mode as one command block."""
    planning = bool(mode.get("planning"))
    write_renderable(renderer, tools_table(available_tools(mode, planning=planning), planning=planning))


def print_resources(renderer: Any, title: str, items: list[dict[str, str]]) -> None:
    """Print one loaded-resource type as one command block."""
    write_renderable(renderer, resources_table(title, items))


def available_tools(mode: dict[str, Any], *, planning: bool) -> list[dict[str, str]]:
    """Return tool display specs for the current mode."""
    key = "planning_tools" if planning else "action_tools"
    tools = mode.get(key)
    if isinstance(tools, list) and tools:
        normalized = normalize_tool_specs(tools)
        if planning:
            if mode.get("planning_stage") in {PLANNING_STAGE_PLAN_FINALIZE, PLANNING_STAGE_GOAL_FINALIZE}:
                expected = FINALIZE_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_FINALIZE else FINALIZE_PLAN_TOOL
                return [tool for tool in normalized if tool["name"] == expected]
            expected_prepare = PREPARE_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_RESEARCH else PREPARE_PLAN_TOOL
            expected_show = SHOW_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_RESEARCH else SHOW_PLAN_TOOL
            hidden = {PREPARE_PLAN_TOOL, PREPARE_GOAL_TOOL, FINALIZE_PLAN_TOOL, FINALIZE_GOAL_TOOL} - {expected_prepare}
            visible = [tool for tool in normalized if tool["name"] not in hidden]
            visible.sort(key=lambda tool: tool["name"] != expected_show)
            return visible
        return normalized

    if not planning:
        blocked = {FINALIZE_PLAN_TOOL, FINALIZE_GOAL_TOOL, PREPARE_PLAN_TOOL, PREPARE_GOAL_TOOL}
        return [tool for tool in DEFAULT_TOOL_SPECS if tool["name"] not in blocked]

    blocked = set(PLAN_DISABLED_TOOLS)
    return [tool for tool in DEFAULT_TOOL_SPECS if tool["name"] not in blocked]


def tool_specs(agent: Any) -> list[dict[str, str]]:
    """Extract displayable tool specs from an agent-like object."""
    explicit = getattr(agent, "mira_tool_specs", None)
    if isinstance(explicit, list) and explicit:
        return normalize_tool_specs(explicit)

    get_tools = getattr(agent, "get_tools", None)
    if callable(get_tools):
        return normalize_tool_specs(get_tools())

    tools = getattr(agent, "tools", None)
    if isinstance(tools, list | tuple):
        return normalize_tool_specs(tools)

    return DEFAULT_TOOL_SPECS.copy()


def resource_specs(agent: Any) -> dict[str, list[dict[str, str]]]:
    """Extract resource display metadata from an agent-like object."""
    resources = getattr(agent, "mira_resources", None)
    if not isinstance(resources, dict):
        return {"memories": [], "skills": [], "subagents": [], "tools": []}

    return {
        "memories": normalize_resource_items(resources.get("memories", [])),
        "skills": normalize_resource_items(resources.get("skills", [])),
        "subagents": normalize_resource_items(resources.get("subagents", [])),
        "tools": normalize_resource_items(resources.get("tools", [])),
    }


def resources_for(mode: dict[str, Any], key: str) -> list[dict[str, str]]:
    """Return display metadata for a resource type."""
    resources = mode.get("resources")
    if not isinstance(resources, dict):
        return []
    return normalize_resource_items(resources.get(key, []))


def normalize_resource_items(items: Any) -> list[dict[str, str]]:
    """Normalize resource metadata for display."""
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        path = str(item.get("path") or "")
        source = str(item.get("source") or "")
        if not name or not path or not source:
            continue
        normalized.append(
            {
                "name": name,
                "path": path,
                "source": source,
                "replaces": str(item.get("replaces") or ""),
            }
        )
    return normalized


def normalize_tool_specs(tools: list[Any] | tuple[Any, ...]) -> list[dict[str, str]]:
    """Normalize tool objects, callables, and dicts for display."""
    specs: list[dict[str, str]] = []
    environment = mira_environment_label()
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        spec = {
            "name": name,
            "description": first_sentence(tool_description(tool)),
            "source": "built-in",
            "runtime": "MIRA",
            "environment": environment,
        }
        if isinstance(tool, dict):
            for key in ("source", "replaces", "path", "runtime", "environment"):
                if key in tool:
                    spec[key] = str(tool.get(key) or "")
        specs.append(spec)
    return specs


def tool_name(tool: Any) -> str:
    """Return a display name for a supported tool shape."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return str(name) if name else ""

    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    return str(name) if name else ""


def tool_description(tool: Any) -> str:
    """Return a display description from metadata or docstring."""
    if isinstance(tool, dict):
        description = tool.get("description")
        return str(description).strip() if description else ""

    description = getattr(tool, "description", None)
    if description:
        return str(description).strip()

    doc = getattr(tool, "__doc__", None)
    return doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else ""


def first_sentence(value: str) -> str:
    """Return the first sentence or first non-empty line from text."""
    text = " ".join(line.strip() for line in value.splitlines() if line.strip())
    if not text:
        return ""

    for index, character in enumerate(text):
        if character in {".", "!", "?"}:
            return text[: index + 1]

    return text


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
    if not final_text:
        return False

    if any(marker in final_text.lower() for marker in PLAN_BLOCKED_RESULT_MARKERS):
        return False

    if write_tool_was_used(result):
        return False

    if write_was_blocked(result):
        return False

    return True


def write_tool_was_used(result: TurnResult) -> bool:
    """Return whether the planning agent called a project write tool."""
    return bool(set(PLAN_PROJECT_WRITE_TOOLS).intersection(getattr(result, "tool_calls", [])))


def write_was_blocked(result: TurnResult) -> bool:
    """Return whether any tool result reports a blocked planning-mode write."""
    tool_results = getattr(result, "tool_results", [])
    return any(marker in value.lower() for value in tool_results for marker in PLAN_BLOCKED_RESULT_MARKERS)


def plan_request_text(text: str) -> str:
    """Wrap user input in the planning-mode instruction template."""
    return PLAN_REQUEST_TEMPLATE.format(
        disabled_tools=plan_disabled_tools_text(),
        behavior_policy=PLAN_BEHAVIOR_POLICY,
        text=text,
    )


def plan_revision_text(plan: dict[str, Any], feedback: str) -> str:
    """Return a planning-mode request that keeps revision context explicit."""
    return PLAN_REVISION_TEMPLATE.format(plan=plan_artifact_text(plan), feedback=feedback.strip())


def goal_revision_text(value: dict[str, Any], feedback: str) -> str:
    """Return a complete Goal revision request for read-only discovery."""
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
    """Inject the exact retained Plan or an existing active Goal."""
    if isinstance(retained_plan, dict):
        return PLAN_CONTEXT_TEMPLATE.format(
            plan=plan_artifact_text(retained_plan),
            execution_instructions=APPROVED_PLAN_EXECUTION_INSTRUCTIONS,
            text=text,
        )
    if isinstance(retained_goal, dict):
        return GOAL_CONTEXT_TEMPLATE.format(
            goal=goal_artifact_text(retained_goal),
            text=text,
        )
    return text


def explicit_goal_request_text(text: str) -> str:
    """Force an explicit Goal command or revision through prepare_goal."""
    return EXPLICIT_GOAL_REQUEST_TEMPLATE.format(
        disabled_tools=plan_disabled_tools_text(),
        optional_research=OPTIONAL_RESEARCH_POLICY,
        text=text,
    )


def write_line(renderer: Any, text: str, *, kind: str = "system") -> None:
    """Write one command/status line through the current UI adapter."""
    if hasattr(renderer, "system_message"):
        renderer.system_message(text, kind=kind)
        return
    renderer.console.print(text)


def write_renderable(renderer: Any, renderable: Any) -> None:
    """Write a Rich renderable through the current UI adapter."""
    if hasattr(renderer, "command_output"):
        renderer.command_output(renderable)
        return
    renderer.console.print(renderable)


def clear(renderer: Any) -> None:
    """Clear the current interactive output surface."""
    if hasattr(renderer, "clear_log"):
        renderer.clear_log()
        return
    renderer.console.clear()
