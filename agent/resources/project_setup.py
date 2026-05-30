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
- `skills/<skill>/SKILL.md`: DeepAgents skills. Project skills override default
  skills when the frontmatter `name` is the same.
- `subagents/*.py`: Python files that export `SUBAGENTS = [...]`. Project
  subagents override default subagents when the `name` is the same.
- `tools/*.py`: Python files that export `TOOLS = [...]` or
  `get_tools(project_backend)`. Project tools override defaults when the tool
  `name` is the same.

Use `/memories`, `/skills`, `/subagents`, and `/tools` in the REPL to inspect
what MIRA loaded.
"""

EXAMPLE_MEMORY = """# Project Memory

Describe this project's commands, style, architecture, and preferences here.
MIRA loads this file instead of its default `AGENTS.md`.
"""

EXAMPLE_SKILL = """---
name: example-skill
description: Example project skill showing where local workflows belong.
---

# Example Skill

Use this file for a project-specific workflow. Rename the folder and `name`
when you turn it into a real skill.
"""

EXAMPLE_SUBAGENT = '''"""Example project subagent.

Edit or delete this file when you know which project helpers you want.
"""

SUBAGENTS = [
    {
        "name": "project-guide",
        "description": "Answer questions about this project's local conventions.",
        "system_prompt": (
            "You are a project guide. Inspect relevant files before answering "
            "and keep guidance specific to this workspace."
        ),
    }
]
'''

EXAMPLE_TOOL = '''"""Example project tool.

Edit or delete this file when you know which project helpers you want.
"""

from langchain.tools import tool


@tool
def project_note() -> str:
    """Return a short note proving project tools are loaded."""
    return "Project tool loaded."


TOOLS = [project_note]
'''
