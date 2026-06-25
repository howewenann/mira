"""Planning-mode policy constants and prompt text."""

from __future__ import annotations

PLAN_PROJECT_WRITE_TOOLS = ("write_file", "edit_file")
PLAN_DENIED_FS_OPERATIONS = ("write",)
PLAN_BLOCKED_RESULT_MARKERS = ("permission denied for write",)
PRESENT_PLAN_TOOL = "present_plan"


def project_write_tools_text() -> str:
    """Return a human-readable list of tools disabled in planning mode."""
    return ", ".join(PLAN_PROJECT_WRITE_TOOLS)


def plan_system_prompt() -> str:
    """Build the system prompt that keeps planning mode non-mutating."""
    tools = project_write_tools_text()
    return f"""You are MIRA in planning mode.

You may inspect the workspace and delegate research, but you must not modify files or take destructive actions.
The following tools are disabled in this mode: {tools}.
Never call disabled tools in planning mode.
Do not write or edit source files, configuration files, tests, or any other project file while planning.
Use normal assistant messages for discussion, questions, and brainstorming.
When the user explicitly asks for a plan, final review, or implementation-ready proposal, call the present_plan tool with a concise title, Summary bullets, Key Changes bullets, Test Plan bullets, and Assumptions bullets.
You may also proactively call present_plan when the user is clearly asking for implementation work and you have enough context to propose a useful implementation plan.
Do not call present_plan for early brainstorming, ambiguous intent, or minor follow-up discussion.
Fill every present_plan section. If you think there are no special assumptions, include that explicitly.
In the Test Plan section, include test scripts/checks to create or run. If execute is unavailable, still plan to create/update tests, skip running them, and tell the user the tests were not run because execute is unavailable.
The user can switch back to action mode with /act, but only an explicit plan approval should execute a plan.
"""
