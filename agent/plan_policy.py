from __future__ import annotations

PLAN_PROJECT_WRITE_TOOLS = ("write_file", "edit_file")
PLAN_DENIED_FS_OPERATIONS = ("write",)
PLAN_BLOCKED_RESULT_MARKERS = ("permission denied for write",)


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
If the user asks you to modify project files, write a concrete implementation plan instead of modifying those files.
The user can switch back to action mode with /act when they are ready to execute the plan.
"""
