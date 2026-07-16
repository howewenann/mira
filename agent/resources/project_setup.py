"""Creation of editable `.mira` project examples."""

from __future__ import annotations

from pathlib import Path

from agent.resources.paths import MEMORIES_DIR, PROJECT_DIR, SKILLS_DIR, SUBAGENTS_DIR, TOOLS_DIR


def ensure_project_examples(workspace: Path) -> None:
    mira_dir = workspace / PROJECT_DIR
    memories_dir = mira_dir / MEMORIES_DIR
    skills_dir = mira_dir / SKILLS_DIR / "example-skill"
    subagents_dir = mira_dir / SUBAGENTS_DIR
    tools_dir = mira_dir / TOOLS_DIR

    memories_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    subagents_dir.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)

    write_example(mira_dir / "README.md", PROJECT_README)
    write_example(memories_dir / "AGENTS.md", EXAMPLE_MEMORY)
    write_example(skills_dir / "SKILL.md", EXAMPLE_SKILL)
    write_example(subagents_dir / "example_subagent.py", EXAMPLE_SUBAGENT)
    write_example(tools_dir / "example_tool.py", EXAMPLE_TOOL)


def write_example(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


PROJECT_README = """# MIRA Project Resources

MIRA loads project resources from this folder on top of its defaults.

- `memories/*.md`: always-on project context. A file with the same name as a
  default memory replaces the default.
- `skills/<skill>/SKILL.md`: DeepAgents skills. Project skills are loaded from
  these folders and may override bundled skills if MIRA adds any later.
- `subagents/*.py`: Python files that export `SUBAGENTS = [...]`. Project
  subagents are loaded from these files and may override bundled subagents if
  MIRA adds any later.
- `tools/*.py`: Python files with module-level LangChain `@tool` objects.
  Files can also define `get_tools(project_backend)` for tools that need
  workspace access. Project tools override defaults when the tool `name` is
  the same.

Use `/runtime` in the TUI to inspect the active model and connection. Use
`/tools`, `/memories`, `/skills`, and `/subagents` for their focused sections.
"""

EXAMPLE_MEMORY = """# Example Project Memory

This is example memory. Replace it with this project's commands, style,
architecture, and preferences.
MIRA loads this file instead of its default `AGENTS.md`.
"""

EXAMPLE_SKILL = """---
name: example-skill
description: Example skill placeholder. Rename this before using it for a real project workflow.
---

# Example Skill

This is an example skill. Rename the folder and frontmatter `name`, then
replace this text with a real project-specific workflow.
"""

EXAMPLE_SUBAGENT = '''"""Example project subagent placeholder.

Edit or delete this file when you know which project helpers you want. Rename
the subagent before using it for real work.
"""

SUBAGENTS = [
    {
        "name": "example-project-guide",
        "description": "Example subagent placeholder. Rename before using for real project guidance.",
        "system_prompt": (
            "You are an example project guide placeholder. Replace this prompt "
            "with concrete project guidance before relying on this subagent."
        ),
    }
]
'''

EXAMPLE_TOOL = '''"""Example project tool placeholder.

Edit or delete this file when you know which project helpers you want. Rename
the tool before using it for real work.
"""

from langchain.tools import tool


@tool
def example_project_note() -> str:
    """Return a short note proving example project tools are loaded."""
    return "Example project tool loaded."
'''
