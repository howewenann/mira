"""Interactive REPL loop and slash-command handling."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from agent.plan_policy import PLAN_BLOCKED_RESULT_MARKERS, PLAN_PROJECT_WRITE_TOOLS, project_write_tools_text
from runtime.runner import TurnResult, run_turn

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
    "/plan": "enter planning mode; write/edit tools are disabled",
    "/act": "return to action mode; include the latest saved plan once",
    "/plans": "show saved plans from this REPL session",
    "/session": "show session id, mode, workspace, and turn count",
    "/model": "show the configured model name",
    "/clear": "clear the terminal",
    "/exit": "quit MIRA",
}

DEFAULT_TOOL_SPECS = [
    {"name": "write_todos", "description": ""},
    {"name": "ls", "description": ""},
    {"name": "read_file", "description": ""},
    {"name": "write_file", "description": ""},
    {"name": "edit_file", "description": ""},
    {"name": "glob", "description": ""},
    {"name": "grep", "description": ""},
    {"name": "execute", "description": ""},
    {"name": "task", "description": ""},
]


async def start_repl(
    agent: Any,
    plan_agent: Any,
    renderer: Any,
    store: Any,
    session: dict[str, Any],
    model_name: str,
) -> None:
    """Run the interactive prompt loop.

    The REPL keeps mode state local because planning and action transitions are
    only needed inside the interactive prompt loop.
    """
    renderer.splash(model_name=model_name, session_id=session["id"], workspace=session["workspace"])
    mode: dict[str, Any] = {
        "planning": False,
        "last_plan": "",
        "plan_pending": False,
        "plans": [],
        "plan_runs": 0,
        "action_tools": tool_specs(agent),
        "planning_tools": tool_specs(plan_agent),
    }

    history_path = Path(session["workspace"]) / ".mira" / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = PromptSession(history=FileHistory(str(history_path)))

    while True:
        try:
            text = await prompt.prompt_async(prompt_label(mode))
        except (EOFError, KeyboardInterrupt):
            renderer.newline()
            break

        text = text.strip()
        if not text:
            continue

        if await handle_command(text, renderer, session, model_name, mode):
            if text in {"/exit", "/quit"}:
                break
            continue

        if mode["planning"]:
            result = await run_turn(
                agent=plan_agent,
                text=plan_request_text(text),
                renderer=renderer,
                thread_id=mode["plan_thread_id"],
            )
            save_clean_plan(mode, result, renderer)
        else:
            action_text = action_request_text(mode, text)
            await run_turn(agent=agent, text=action_text, renderer=renderer, thread_id=session["id"])

        session["turns"] += 1
        store.save(session)


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
        renderer.console.print("[dim]bye[/dim]")
        return True

    if text == "/help":
        print_help(renderer)
        return True

    if text == "/tools":
        print_tools(renderer, mode)
        return True

    if text == "/plan":
        mode["planning"] = True
        mode["last_plan"] = ""
        mode["plan_pending"] = False
        mode["plan_runs"] = mode.get("plan_runs", 0) + 1
        mode["plan_thread_id"] = plan_thread_id(session, mode["plan_runs"])
        renderer.console.print(f"[cyan]planning mode[/cyan]: {project_write_tools_text()} disabled; use /act to leave")
        return True

    if text == "/act":
        mode["planning"] = False
        if mode.get("last_plan"):
            mode["plan_pending"] = True
            renderer.console.print("[cyan]action mode[/cyan]: last plan will be included in your next request")
        else:
            renderer.console.print("[cyan]action mode[/cyan]")
        return True

    if text == "/clear":
        renderer.console.clear()
        return True

    if text == "/plans":
        print_plans(renderer, mode)
        return True

    if text == "/session":
        renderer.console.print(f"session: {session['id']}")
        renderer.console.print(f"mode: {'planning' if mode['planning'] else 'action'}")
        renderer.console.print(f"saved plans: {len(mode.get('plans', []))}")
        renderer.console.print(f"workspace: {session['workspace']}")
        renderer.console.print(f"turns: {session['turns']}")
        return True

    if text == "/model":
        renderer.console.print(f"model: {model_name}")
        return True

    renderer.console.print(f"[dim]unknown command:[/dim] {text}")
    return True


def prompt_label(mode: dict[str, Any]) -> HTML:
    """Return the prompt label for the current REPL mode."""
    if mode.get("planning"):
        return HTML("<b><ansicyan>you</ansicyan> <ansiyellow>[plan]</ansiyellow>&gt;</b> ")

    return HTML("<b><ansicyan>you&gt;</ansicyan></b> ")


def print_help(renderer: Any) -> None:
    """Print command descriptions."""
    renderer.console.print("[bold cyan]Commands[/bold cyan]")
    for command, description in COMMAND_HELP.items():
        renderer.console.print(f"  {command:<8} {description}")


def print_tools(renderer: Any, mode: dict[str, Any]) -> None:
    """Print tools available in the current mode."""
    planning = bool(mode.get("planning"))
    mode_name = "planning" if planning else "action"
    renderer.console.print(f"[bold cyan]Tools ({mode_name})[/bold cyan]")
    renderer.console.print(tool_table(available_tools(mode, planning=planning), width=console_width(renderer)))


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


def normalize_tool_specs(tools: list[Any] | tuple[Any, ...]) -> list[dict[str, str]]:
    """Normalize tool objects, callables, and dicts for display."""
    specs: list[dict[str, str]] = []
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        specs.append({"name": name, "description": first_sentence(tool_description(tool))})
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


def console_width(renderer: Any) -> int:
    """Return the current console width with a stable fallback."""
    width = getattr(getattr(renderer, "console", None), "width", None)
    return width if isinstance(width, int) and width > 40 else 88


def tool_table(tools: list[dict[str, str]], width: int) -> str:
    """Render tools as a fixed-width markdown-style table."""
    name_width = min(max([len("Tool"), *(len(tool["name"]) for tool in tools)], default=4), 24)
    description_width = max(width - name_width - 7, 24)
    lines = [
        f"| {'Tool'.ljust(name_width)} | {'Description'.ljust(description_width)} |",
        f"| {'-' * name_width} | {'-' * description_width} |",
    ]

    for tool in tools:
        wrapped = textwrap.wrap(tool["description"] or "-", width=description_width) or ["-"]
        lines.append(f"| {tool['name'].ljust(name_width)} | {wrapped[0].ljust(description_width)} |")
        for continuation in wrapped[1:]:
            lines.append(f"| {' '.ljust(name_width)} | {continuation.ljust(description_width)} |")

    return "\n".join(lines)


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
            renderer.console.print("[yellow]planning mode[/yellow]: write/edit was blocked; no plan was saved")
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
    return any(
        marker in value.lower()
        for value in tool_results
        for marker in PLAN_BLOCKED_RESULT_MARKERS
    )


def print_plans(renderer: Any, mode: dict[str, Any]) -> None:
    """Print all saved planning-mode responses."""
    plans = mode.get("plans", [])
    if not plans:
        if hasattr(renderer, "no_plans"):
            renderer.no_plans()
        else:
            renderer.console.print("[dim]no saved plans[/dim]")
        return

    for plan in plans:
        if hasattr(renderer, "plan"):
            renderer.plan(plan["id"], plan["text"])
        else:
            renderer.console.print(plan["text"])


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
