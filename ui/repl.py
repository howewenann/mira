"""Interactive-mode state and slash-command helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from rich.table import Table

from agent.context_overflow import mark_context_notice_rendered, pop_context_overflow_notice
from agent.plan_policy import PLAN_BLOCKED_RESULT_MARKERS, PLAN_PROJECT_WRITE_TOOLS, project_write_tools_text
from runtime.runner import TurnResult, run_turn
from session.dashboard import apply_turn_usage, ensure_dashboard
from session.context import update_title, with_resume_context
from session.recorder import RecordingRenderer, SessionRecorder, poll_compactions

PLAN_CONTEXT_TEMPLATE = """Previous planning context:
{plan}

You are now in action mode. Write/edit tools are available again, subject to normal approval prompts. Do not assume planning-mode permission errors still apply.

User request:
{text}"""

PLAN_REQUEST_TEMPLATE = """You are in planning mode.
Do not call write_file, edit_file, or any other tool that modifies files.
Do not attempt the requested change.
Respond only with a concrete plan for how the change should be done later in action mode.
If the user asks to create, write, edit, or delete files, describe the exact file operation that should happen after /act.

User request:
{text}"""

COMMAND_HELP = {
    "/help": "show commands and what they do",
    "/tools": "list tools available in the current mode",
    "/memories": "list loaded memory files and replacements",
    "/skills": "list loaded skills and replacements",
    "/subagents": "list loaded subagents and replacements",
    "/plan": "enter planning mode; write/edit tools are disabled",
    "/act": "return to action mode; include the latest saved plan once",
    "/plans": "show saved plans from this session",
    "/session": "show session id, mode, workspace, and turn count",
    "/model": "show the configured model name",
    "/clear": "clear the log",
    "/exit": "quit MIRA",
}

DEFAULT_TOOL_SPECS = [
    {
        "name": "ask_user",
        "description": "Ask the user to choose between concrete next steps when MIRA is blocked.",
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


def initial_mode(agent: Any, plan_agent: Any) -> dict[str, Any]:
    """Return the mutable interactive state for one TUI session."""
    return {
        "planning": False,
        "last_plan": "",
        "plan_pending": False,
        "plans": [],
        "plan_runs": 0,
        "action_tools": tool_specs(agent),
        "planning_tools": tool_specs(plan_agent),
        "resources": resource_specs(agent),
    }


def refresh_agent_specs(mode: dict[str, Any], agent: Any, plan_agent: Any) -> None:
    """Refresh tool/resource metadata after agents are rebuilt."""
    mode["action_tools"] = tool_specs(agent)
    mode["planning_tools"] = tool_specs(plan_agent)
    mode["resources"] = resource_specs(agent)


async def run_user_turn(
    *,
    agent: Any,
    plan_agent: Any,
    renderer: Any,
    store: Any,
    session: dict[str, Any],
    mode: dict[str, Any],
    text: str,
    model_name: str = "",
    context_limit_tokens: int | None = None,
    context_limit_source: str = "unknown",
    token_counter: Any | None = None,
) -> TurnResult:
    """Route one submitted user prompt through planning or action mode."""
    run_kwargs = {"token_counter": token_counter} if token_counter is not None else {}
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

    if mode["planning"]:
        active_agent = plan_agent
        thread_id = mode["plan_thread_id"]
        mode_name = "planning"
        request_text = with_resume_context(session, plan_request_text(text))
    else:
        active_agent = agent
        thread_id = session["id"]
        mode_name = "action"
        action_text = action_request_text(mode, text)
        request_text = with_resume_context(session, action_text)

    ensure_dashboard(
        session,
        model_name=model_name,
        context_limit_tokens=context_limit_tokens,
        context_limit_source=context_limit_source,
    )
    recorder = SessionRecorder(session, store, mode_name)
    recorder.user_message(text)
    update_title(session)
    recorder.save()
    wrapped_renderer = RecordingRenderer(renderer, recorder)
    poller = asyncio.create_task(poll_compactions(recorder, active_agent, thread_id))

    try:
        result = await run_turn(
            agent=active_agent,
            text=request_text,
            renderer=wrapped_renderer,
            thread_id=thread_id,
            usage_callback=apply_live_usage,
            **run_kwargs,
        )
    except asyncio.CancelledError:
        await sync_compaction_safely(recorder, active_agent, thread_id)
        recorder.interrupted("turn interrupted before completion")
        raise
    except ContextOverflowError as exc:
        await sync_compaction_safely(recorder, active_agent, thread_id)
        notice = pop_context_overflow_notice(exc)
        if notice and not wrapped_renderer.context_notice_rendered():
            recorder.info(notice)
            write_line(renderer, notice, kind="info")
            wrapped_renderer.mark_context_notice_rendered()
        mark_context_notice_rendered(exc)
        raise
    except Exception as exc:
        await sync_compaction_safely(recorder, active_agent, thread_id)
        recorder.system_error(f"turn error: {exc}")
        raise
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller

    if mode_name == "planning":
        save_clean_plan(mode, result, renderer)
    recorder.ensure_assistant(getattr(result, "final_text", ""))

    session["turns"] = int(session.get("turns") or 0) + 1
    update_title(session)
    await sync_compaction_safely(recorder, active_agent, thread_id)
    if not live_usage_applied:
        apply_turn_usage(
            session,
            result,
            model_name=model_name,
            context_limit_tokens=context_limit_tokens,
            context_limit_source=context_limit_source,
        )
    store.save(session)
    return result


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

    if text == "/plan":
        mode["planning"] = True
        mode["last_plan"] = ""
        mode["plan_pending"] = False
        mode["plan_runs"] = mode.get("plan_runs", 0) + 1
        mode["plan_thread_id"] = plan_thread_id(session, mode["plan_runs"])
        write_line(renderer, f"planning mode: {project_write_tools_text()} disabled; use /act to leave", kind="status")
        return True

    if text == "/act":
        mode["planning"] = False
        if mode.get("last_plan"):
            mode["plan_pending"] = True
            write_line(renderer, "action mode: last plan will be included in your next request", kind="status")
        else:
            write_line(renderer, "action mode", kind="status")
        return True

    if text == "/clear":
        clear(renderer)
        return True

    if text == "/plans":
        print_plans(renderer, mode)
        return True

    if text == "/session":
        write_line(renderer, f"session: {session['id']}")
        write_line(renderer, f"title: {session.get('title', 'Untitled session')}")
        write_line(renderer, f"mode: {'planning' if mode['planning'] else 'action'}")
        write_line(renderer, f"saved plans: {len(mode.get('plans', []))}")
        write_line(renderer, f"workspace: {session['workspace']}")
        write_line(renderer, f"turns: {session['turns']}")
        return True

    if text == "/model":
        write_line(renderer, f"model: {model_name}")
        return True

    write_line(renderer, f"unknown command: {text}", kind="muted")
    return True


def print_help(renderer: Any) -> None:
    """Print command descriptions."""
    write_renderable(renderer, help_table())


def help_table() -> Table:
    """Build a single Rich table for slash-command help."""
    table = Table(title="Commands", title_style="bold cyan")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    for command, description in COMMAND_HELP.items():
        table.add_row(command, description)
    return table


def print_tools(renderer: Any, mode: dict[str, Any]) -> None:
    """Print tools available in the current mode."""
    planning = bool(mode.get("planning"))
    mode_name = "planning" if planning else "action"
    write_renderable(renderer, tools_table(f"Tools ({mode_name})", available_tools(mode, planning=planning)))


def print_resources(renderer: Any, title: str, items: list[dict[str, str]]) -> None:
    """Print loaded resources for one resource type."""
    if not items:
        write_line(renderer, title, kind="heading")
        write_line(renderer, "none loaded", kind="muted")
        return

    write_renderable(renderer, resources_table(title, items))


def tools_table(title: str, tools: list[dict[str, str]]) -> Table:
    """Build a Rich table for tool metadata."""
    table = Table(title=title, title_style="bold cyan")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Replaces", no_wrap=True)
    table.add_column("Description")

    for tool in tools:
        table.add_row(
            tool["name"],
            tool.get("source") or "-",
            tool.get("replaces") or "-",
            tool.get("description") or "-",
        )
    return table


def resources_table(title: str, items: list[dict[str, str]]) -> Table:
    """Build a Rich table for resource metadata."""
    table = Table(title=title, title_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Replaces", no_wrap=True)
    table.add_column("Path")

    for item in items:
        table.add_row(
            item["name"],
            item.get("source") or "-",
            item.get("replaces") or "-",
            item.get("path") or "-",
        )
    return table


def available_tools(mode: dict[str, Any], *, planning: bool) -> list[dict[str, str]]:
    """Return tool display specs for the current mode."""
    key = "planning_tools" if planning else "action_tools"
    tools = mode.get(key)
    if isinstance(tools, list) and tools:
        return normalize_tool_specs(tools)

    if not planning:
        return DEFAULT_TOOL_SPECS.copy()

    blocked = set(PLAN_PROJECT_WRITE_TOOLS)
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
        return {"memories": [], "skills": [], "subagents": []}

    return {
        "memories": normalize_resource_items(resources.get("memories", [])),
        "skills": normalize_resource_items(resources.get("skills", [])),
        "subagents": normalize_resource_items(resources.get("subagents", [])),
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
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        spec = {"name": name, "description": first_sentence(tool_description(tool))}
        if isinstance(tool, dict):
            for key in ("source", "replaces", "path"):
                value = tool.get(key)
                if value:
                    spec[key] = str(value)
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
    if run_id is None:
        return f"{session['id']}:plan"
    return f"{session['id']}:plan:{run_id}"


def save_clean_plan(mode: dict[str, Any], result: TurnResult, renderer: Any) -> None:
    """Save a planning result only when it did not try to edit the project."""
    if not has_clean_plan(result):
        mode["last_plan"] = ""
        mode["plan_pending"] = False
        if write_was_blocked(result) or write_tool_was_used(result):
            write_line(renderer, "planning mode: write/edit was blocked; no plan was saved", kind="warning")
        return

    plan_text = result.final_text.strip()
    plan = {"id": len(mode.setdefault("plans", [])) + 1, "text": plan_text}
    mode["plans"].append(plan)
    mode["last_plan"] = plan_text
    mode["plan_pending"] = False


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


def print_plans(renderer: Any, mode: dict[str, Any]) -> None:
    """Print all saved planning-mode responses."""
    plans = mode.get("plans", [])
    if not plans:
        if hasattr(renderer, "no_plans"):
            renderer.no_plans()
        else:
            write_line(renderer, "no saved plans", kind="muted")
        return

    for plan in plans:
        if hasattr(renderer, "plan"):
            renderer.plan(plan["id"], plan["text"])
        else:
            write_line(renderer, plan["text"])


def plan_request_text(text: str) -> str:
    """Wrap user input in the planning-mode instruction template."""
    return PLAN_REQUEST_TEMPLATE.format(text=text)


def action_request_text(mode: dict[str, Any], text: str) -> str:
    """Inject the latest saved plan into the next action-mode request once."""
    if not mode.get("plan_pending") or not mode.get("last_plan"):
        return text

    plan = mode["last_plan"]
    mode["last_plan"] = ""
    mode["plan_pending"] = False
    return PLAN_CONTEXT_TEMPLATE.format(plan=plan, text=text)


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
