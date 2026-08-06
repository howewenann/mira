"""Creation of editable `.mira` project examples."""

from __future__ import annotations

from pathlib import Path

from agent.resources.paths import (
    MCP_DIR,
    MEMORIES_DIR,
    PROJECT_DIR,
    SKILLS_DIR,
    SUBAGENTS_DIR,
    TOOLS_DIR,
)


def ensure_project_examples(workspace: Path) -> None:
    mira_dir = workspace / PROJECT_DIR
    mcp_dir = mira_dir / MCP_DIR
    memories_dir = mira_dir / MEMORIES_DIR
    skills_dir = mira_dir / SKILLS_DIR / "example-skill"
    subagents_dir = mira_dir / SUBAGENTS_DIR
    tools_dir = mira_dir / TOOLS_DIR
    tool_examples_dir = mira_dir / "examples" / TOOLS_DIR

    mcp_dir.mkdir(parents=True, exist_ok=True)
    memories_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    subagents_dir.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_examples_dir.mkdir(parents=True, exist_ok=True)

    write_example(mcp_dir / "mcp.json", EMPTY_MCP_CONFIGURATION)
    write_example(mcp_dir / "example.json", EXAMPLE_MCP_CONFIGURATION)
    write_example(mcp_dir / "schema.json", MCP_CONFIGURATION_SCHEMA)
    write_example(mira_dir / "README.md", PROJECT_README)
    write_example(memories_dir / "AGENTS.md", EXAMPLE_MEMORY)
    write_example(skills_dir / "SKILL.md", EXAMPLE_SKILL)
    write_example(subagents_dir / "example_subagent.py", EXAMPLE_SUBAGENT)
    write_example(tool_examples_dir / "mira_runtime_tool.py", MIRA_RUNTIME_TOOL_EXAMPLE)
    write_example(tool_examples_dir / "project_runtime_tool.py", PROJECT_RUNTIME_TOOL_EXAMPLE)


def write_example(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


EMPTY_MCP_CONFIGURATION = '''{
  "$schema": "./schema.json",
  "mcpServers": {}
}
'''


EXAMPLE_MCP_CONFIGURATION = '''{
  "$schema": "./schema.json",
  "mcpServers": {
    "local-server": {
      "type": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/server.py"],
      "env": {}
    },
    "remote-server": {
      "type": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${env:REMOTE_MCP_TOKEN}"
      }
    }
  }
}
'''


MCP_CONFIGURATION_SCHEMA = '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MIRA MCP configuration",
  "description": "The supported configuration format for MCP servers loaded by MIRA. String values may reference process environment variables with ${env:NAME}.",
  "type": "object",
  "properties": {
    "$schema": {
      "type": "string",
      "description": "A relative or absolute URI for this JSON Schema."
    },
    "mcpServers": {
      "type": "object",
      "description": "MCP servers keyed by the non-empty name shown in MIRA.",
      "propertyNames": {
        "type": "string",
        "minLength": 1
      },
      "additionalProperties": {
        "oneOf": [
          {
            "title": "stdio server",
            "description": "A local MCP server launched as a subprocess.",
            "type": "object",
            "properties": {
              "type": {
                "const": "stdio",
                "description": "The optional local subprocess transport type."
              },
              "command": {
                "type": "string",
                "minLength": 1,
                "description": "The executable used to launch the server."
              },
              "args": {
                "type": "array",
                "description": "Arguments passed to the server command.",
                "items": {
                  "type": "string"
                }
              },
              "env": {
                "type": "object",
                "description": "Environment variables passed to the server process. Values may contain ${env:NAME} references.",
                "additionalProperties": {
                  "type": "string"
                }
              }
            },
            "required": ["command"],
            "additionalProperties": false
          },
          {
            "title": "HTTP server",
            "description": "A remote MCP server reached over Streamable HTTP.",
            "type": "object",
            "properties": {
              "type": {
                "const": "http",
                "description": "The required remote HTTP transport type."
              },
              "url": {
                "type": "string",
                "minLength": 1,
                "description": "The remote MCP endpoint URL."
              },
              "headers": {
                "type": "object",
                "description": "HTTP headers sent with MCP requests. Values may contain ${env:NAME} references, such as Bearer ${env:MCP_TOKEN}.",
                "additionalProperties": {
                  "type": "string"
                }
              }
            },
            "required": ["type", "url"],
            "additionalProperties": false
          }
        ]
      }
    }
  },
  "required": ["mcpServers"],
  "additionalProperties": false
}
'''


PROJECT_README = """# MIRA Project Resources

MIRA loads project resources from this folder on top of its defaults.

- `memories/*.md`: always-on project context. A file with the same name as a
  default memory replaces the default.
- `skills/<skill>/SKILL.md`: DeepAgents skills. Project skills are loaded from
  these folders and may override bundled skills if MIRA adds any later.
- `subagents/*.py`: Python files that export `SUBAGENTS = [...]`. Project
  subagents are loaded from these files and may override bundled subagents if
  MIRA adds any later.
- `tools/*.py`: active Python tool files. Standard LangChain `@tool` runs in
  MIRA, while `mira_tool_api.project_tool` runs its function body in the
  configured project Execute Environment.
- `mcp/mcp.json`: active MCP configuration. `mcp/example.json` is inert, and
  `mcp/schema.json` documents the supported keys. MCP string values can use
  `${env:NAME}` to read explicit process or workspace `.env` values without
  storing the resolved value. Run `/reload` after changes.
- `examples/tools/*.py`: inert examples to copy into `tools/`; this folder is
  never scanned as active resources.
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

MIRA_RUNTIME_TOOL_EXAMPLE = '''# Standard MIRA-runtime tool.
#
# This file is imported and executed inside MIRA's Python environment.
# Packages imported here must be installed in MIRA's environment.
#
# To use:
#   1. Copy this file into .mira/tools/
#   2. Rename and edit the function.
#   3. Run /reload.
#
# If an imported package is missing, MIRA will offer to install it
# into MIRA's environment.

from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())
'''

PROJECT_RUNTIME_TOOL_EXAMPLE = '''# Project-runtime tool.
#
# The tool is exposed to the agent normally, but its function body runs
# in the configured project Execute Environment.
#
# Use this for:
#   - Packages installed only in the project's Conda env or venv
#   - Imports from the current project package
#
# Important:
#   Project-only imports must remain inside the function because MIRA
#   imports this file first to discover the tool.
#
# To use:
#   1. Copy this file into .mira/tools/
#   2. Rename and edit the function.
#   3. Configure the Execute Environment in /settings.
#   4. Run /reload.
#
# Arguments should be JSON-compatible. Return JSON-compatible data or text.
# Other return values are converted to a readable representation.

from mira_tool_api import project_tool


@project_tool
def inspect_csv(path: str) -> str:
    """Summarize a CSV using the project environment."""

    import pandas as pd

    dataframe = pd.read_csv(path)
    return dataframe.describe(include="all").to_string()
'''
