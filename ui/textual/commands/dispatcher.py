"""Textual slash-command parsing and presentation helpers."""

from __future__ import annotations

import inspect
from typing import Any

from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ui.textual.commands.help import COMMAND_HELP_SECTIONS

from agent.planning.policy import plan_disabled_tools_text
from core.application import available_tools, resources_for, select_mode
from session.goals import current_goal
from session.plans import current_plan
from ui.textual.runtime_report import resources_table, tools_table

HELP_SECTION_STYLE = "bold #7aa2f7"


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

    if text == "/context-report":
        callback = getattr(renderer, "open_context_report", None)
        if callable(callback):
            outcome = callback()
            if inspect.isawaitable(outcome):
                await outcome
        else:
            write_line(renderer, "Context Report is unavailable in this interface.", kind="warning")
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
        select_mode(session, mode, "plan")
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
        select_mode(session, mode, "act")
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
