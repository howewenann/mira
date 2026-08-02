"""MIRA guidance for the DeepAgents execute tool."""

from __future__ import annotations

import copy
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from agent.tools.specs import tool_name as resource_tool_name

MIRA_EXECUTE_TOOL_DESCRIPTION = """Executes a shell command with proper handling and security measures.

Usage:
Executes a given command through MIRA's configured shell backend. In the normal
local execute mode, commands run in the host shell from the project workspace.

Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool
     to verify the parent directory exists and is the correct location.
   - For example, before running "mkdir foo/bar", first use ls to check that
     "foo" exists and is the intended parent directory.
2. MIRA Workspace Path Handling:
   - MIRA file tools use virtual workspace paths rooted at the project
     workspace. For example, write_file path `/tmp.py` creates `tmp.py` in the
     project workspace.
   - The shell does not see virtual workspace paths as host absolute paths. Do
     not pass file-tool paths like `/tmp.py` directly to shell commands.
   - Before running a file created or shown by a file tool, convert its virtual
     workspace path to a workspace-relative shell path.
   - To run a workspace file shown as `/tmp.py`, use `python tmp.py` or
     `python .\\tmp.py`, not `python /tmp.py`.
   - To run a workspace file shown as `/scripts/check_path.py`, use
     `python scripts/check_path.py` or `python .\\scripts\\check_path.py`, not
     `python /scripts/check_path.py`.
   - If a path is under a mounted virtual route such as `/mira-defaults/`, use
     the file tools unless an explicit host shell path is available.
3. Command Execution:
   - Always quote file paths that contain spaces with double quotes
     (e.g., cd "path with spaces/file.txt").
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct for a known host path)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "path with spaces/script.py" (correct)
     - python path with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting and workspace path handling, execute the
     command.
   - Capture the output of the command.

Usage notes:
  - Commands run through MIRA's configured shell backend.
  - Returns combined stdout/stderr output with exit code.
  - If the output is very large, it may be truncated.
  - For long-running commands, use the optional timeout parameter to override
    the default timeout (e.g., execute(command="make build", timeout=300)).
  - A timeout of 0 may disable timeouts on backends that support no-timeout
    execution.
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep.
    Instead use the grep, glob tools to search. You MUST avoid read tools like
    cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate
    them. DO NOT use newlines (newlines are ok in quoted strings).
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir").
    - Use ';' only when you need to run commands sequentially but don't care if
      earlier commands fail.
  - Try to maintain your current working directory throughout the session by
    using workspace-relative paths or known host absolute paths and avoiding
    usage of cd.

Examples:
  Good examples:
    - execute(command="python tmp.py")  # For /tmp.py
    - execute(command="python .\\tmp.py")  # Windows-friendly form for /tmp.py
    - execute(command="python scripts/check_path.py")  # For /scripts/check_path.py
    - execute(command="pytest tests")
    - execute(command="npm install && npm test")
    - execute(command="make build", timeout=300)

  Bad examples (avoid these):
    - execute(command="python /tmp.py")
    - execute(command="python /scripts/check_path.py")
    - execute(command="cd /foo/bar && pytest tests")
    - execute(command="cat file.txt")
    - execute(command="find . -name '*.py'")
    - execute(command="grep -r 'pattern' .")

Note: This tool is only available if the backend supports execution
(SandboxBackendProtocol). If execution is not supported, the tool will return
an error message."""


class ExecuteToolPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace the visible execute tool description with MIRA guidance."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._rewrite_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._rewrite_request(request))

    def _rewrite_request(self, request: Any) -> Any:
        tools = getattr(request, "tools", None)
        if not isinstance(tools, (list, tuple)):
            return request
        rewritten = [execute_tool_with_mira_description(tool) for tool in tools]
        if all(new is old for new, old in zip(rewritten, tools, strict=True)):
            return request
        return request.override(tools=rewritten)


def execute_tool_with_mira_description(tool: Any) -> Any:
    """Return a copy of the execute tool with MIRA's path guidance."""
    if resource_tool_name(tool) != "execute":
        return tool
    if isinstance(tool, dict):
        if tool.get("description") == MIRA_EXECUTE_TOOL_DESCRIPTION:
            return tool
        return {**tool, "description": MIRA_EXECUTE_TOOL_DESCRIPTION}
    model_copy = getattr(tool, "model_copy", None)
    if callable(model_copy):
        if getattr(tool, "description", None) == MIRA_EXECUTE_TOOL_DESCRIPTION:
            return tool
        return model_copy(update={"description": MIRA_EXECUTE_TOOL_DESCRIPTION})
    try:
        copied = copy.copy(tool)
        setattr(copied, "description", MIRA_EXECUTE_TOOL_DESCRIPTION)
        return copied
    except Exception:
        return tool


__all__ = [
    "ExecuteToolPromptMiddleware",
    "MIRA_EXECUTE_TOOL_DESCRIPTION",
    "execute_tool_with_mira_description",
]
