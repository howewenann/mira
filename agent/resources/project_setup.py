"""Bootstrap user-owned configuration and MIRA-managed project guidance."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from config.tracing import TRACING_REGISTRY_TEMPLATE

from agent.resources.paths import (
    MCP_DIR,
    MEMORIES_DIR,
    PROJECT_DIR,
    PROMPTS_DIR,
    SKILLS_DIR,
    SUBAGENTS_DIR,
    TOOLS_DIR,
)


PROJECT_KIT_ROOT = resources.files(__package__) / "project_kit"
_SOURCE_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"


def _examples_root() -> Traversable | Path:
    """Return packaged examples, falling back to their checkout source tree."""
    packaged = PROJECT_KIT_ROOT / "examples"
    return packaged if packaged.is_dir() else _SOURCE_EXAMPLES_ROOT


def _resource_text(path: Traversable | Path) -> str:
    return path.read_text(encoding="utf-8")


def _example_text(relative_path: str) -> str:
    return _resource_text(_examples_root() / relative_path)


def ensure_project_examples(workspace: Path) -> None:
    """Create the local project kit while preserving user-owned configuration."""
    mira_dir = Path(workspace) / PROJECT_DIR

    for relative_dir in (
        MCP_DIR,
        f"{MCP_DIR}/servers",
        MEMORIES_DIR,
        PROMPTS_DIR,
        SKILLS_DIR,
        SUBAGENTS_DIR,
        TOOLS_DIR,
    ):
        (mira_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    # These become project configuration as soon as they are created.
    write_user_template(mira_dir / MCP_DIR / "mcp.json", EMPTY_MCP_CONFIGURATION)
    write_user_template(mira_dir / "models.yml", MODEL_REGISTRY_TEMPLATE)
    write_user_template(mira_dir / "tracing.yml", TRACING_REGISTRY_TEMPLATE)

    # These document the installed MIRA version. Active resources never live
    # here, so upgrading them cannot replace project behavior or configuration.
    write_managed_resource(
        mira_dir / "README.md",
        _resource_text(PROJECT_KIT_ROOT / "README.md"),
    )
    write_managed_resource(
        mira_dir / ".env.example",
        _resource_text(PROJECT_KIT_ROOT / "env.example"),
    )
    write_managed_resource(mira_dir / MCP_DIR / "schema.json", MCP_CONFIGURATION_SCHEMA)

    for target, source in MANAGED_EXAMPLES.items():
        write_managed_resource(mira_dir / "examples" / target, _example_text(source))


def write_user_template(path: Path, content: str) -> None:
    """Create a user-owned template once and never overwrite it."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_managed_resource(path: Path, content: str) -> None:
    """Refresh documentation owned by the installed MIRA version."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


MANAGED_EXAMPLES = {
    "memories/AGENTS.md": "memories/AGENTS.md",
    "skills/example-skill/SKILL.md": "skills/example-skill/SKILL.md",
    "subagents/example_subagent.py": "subagents/example_subagent.py",
    "tools/mira_runtime_tool.py": "tools/mira_runtime_tool.py",
    "tools/project_runtime_tool.py": "tools/project_runtime_tool.py",
    "mcp/README.md": "mcp/README.md",
    "mcp/example.json": "mcp/example.json",
    "tracing/README.md": "tracing/README.md",
    "api/README.md": "mira_api/README.md",
    "api/minimal_frontend.py": "mira_api/minimal_frontend.py",
    "api/full_frontend.py": "mira_api/full_frontend.py",
    "acp/README.md": "acp/README.md",
    "acp/zed.md": "acp/zed.md",
    "acp/stdio/minimal_client.py": "acp/stdio/minimal_client.py",
    "acp/stdio/full_client.py": "acp/stdio/full_client.py",
    "acp/http/minimal_client.py": "acp/http/minimal_client.py",
    "acp/http/full_client.py": "acp/http/full_client.py",
}


EMPTY_MCP_CONFIGURATION = '''{
  "$schema": "./schema.json",
  "mcpServers": {}
}
'''

MCP_CONFIGURATION_SCHEMA = '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MIRA MCP configuration",
  "description": "The supported configuration format for MCP servers loaded by MIRA. String values may reference process environment variables with ${NAME}.",
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
                "description": "Environment variables passed to the server process. Values may contain ${NAME} references.",
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
                "description": "HTTP headers sent with MCP requests. Values may contain ${NAME} references, such as Bearer ${MCP_TOKEN}.",
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


MODEL_REGISTRY_TEMPLATE = '''# MIRA model profiles
#
# Add any number of named model profiles below.
#
# Required: provider, model
# Optional: api_key, api_base, temperature, max_tokens, top_p, model_kwargs
#
# Provider-specific arguments belong under model_kwargs.

models:

  # example-cloud:
  #   provider: <provider>          # Required. AnyLLM provider name.
  #   model: <model-id>             # Required. Exact model ID/name.
  #   api_key: ${API_KEY}           # Optional. Literal or environment reference.
  #   temperature: 0.2              # Optional. Sampling temperature.
  #   max_tokens: 4096              # Optional. Maximum generated output.
  #   top_p: 0.95                    # Optional. Nucleus sampling threshold.
  #   model_kwargs:                  # Optional. Provider-specific arguments.
  #     reasoning_effort: medium

  # example-endpoint:
  #   provider: <provider>          # Required. AnyLLM provider name.
  #   model: <model-id>             # Required. Exact model ID/name.
  #   api_base: http://localhost:1234/v1  # Optional. Compatible endpoint URL.
  #   api_key: <api-key>             # Optional. Literal or ${NAME} reference.
'''
