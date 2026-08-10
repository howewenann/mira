"""Shared slash-command help metadata."""

from __future__ import annotations

from collections.abc import Iterator


COMMAND_HELP_SECTIONS = (
    (
        "General",
        (
            ("/help", "show key bindings, autocomplete, usage notes, and commands"),
            ("/mira", "display a fresh MIRA splash"),
            ("/session", "show conversation identity, mode, goals, plans, workspace, and turns"),
            ("/exit", "quit MIRA"),
        ),
    ),
    (
        "Inspect",
        (
            ("/runtime", "inspect the active model, connection, and launch options"),
            ("/tools", "list tools available in the current mode"),
            ("/memories", "list loaded memory files and replacements"),
            ("/skills", "list loaded skills and replacements"),
            ("/subagents", "list loaded subagents and replacements"),
            ("/issues", "view current configuration and resource issues"),
            ("/prompts", "list local and MCP prompt commands"),
            ("/mcp", "open MCP server status and controls in the TUI"),
        ),
    ),
    (
        "Workflow",
        (
            ("/plan [prompt]", "enter conversational read-only Plan mode, optionally sending a prompt"),
            ("/plan-show", "show the exact retained current Plan"),
            ("/plan-resume", "resume an incomplete retained Plan in Act mode"),
            ("/plan-clear", "remove the retained current Plan without deleting history"),
            ("/goal <prompt>", "create a durable Objective + Success Criteria Goal"),
            ("/goal-show", "show the exact retained current Goal"),
            ("/goal-resume", "resume an incomplete retained Goal in Act mode"),
            ("/goal-clear", "remove the retained current Goal without deleting history"),
            ("/act", "return to action mode"),
        ),
    ),
    (
        "Configuration",
        (
            ("/settings", "configure workspace settings in the TUI"),
            ("/models", "open Settings directly on model management"),
            ("/reload", "reload .env/project resources and rebuild agents in the TUI"),
        ),
    ),
    (
        "Chat & history",
        (
            ("/compact", "summarize older context now"),
            ("/new-chat", "start a fresh saved chat session in the TUI"),
            ("/clear", "clear the chat display"),
            ("/clear-chat", "clear the current saved chat transcript in the TUI"),
            ("/clear-all-chats", "delete all saved chats and compaction archives in the TUI"),
            ("/clear-errors", "delete saved error reports in the TUI"),
            ("/clear-prompts", "clear prompt input history in the TUI"),
        ),
    ),
)


def command_help_entries() -> Iterator[tuple[str, str]]:
    """Yield command usage and descriptions in help-section order."""
    for _section, commands in COMMAND_HELP_SECTIONS:
        yield from commands


def command_insertion(usage: str) -> str:
    """Return only the exact slash-command token."""
    return usage.partition(" ")[0]


__all__ = ["COMMAND_HELP_SECTIONS", "command_help_entries", "command_insertion"]
