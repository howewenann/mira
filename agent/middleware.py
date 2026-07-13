"""MIRA-owned DeepAgents/LangChain middleware."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage
from langchain_quickjs import CodeInterpreterMiddleware

from agent.compaction import (
    create_mira_summarization_middleware,
    create_mira_summarization_tool_middleware,
)
from agent.context_overflow import ProviderContextOverflowMiddleware
from agent.tools.specs import tool_name as resource_tool_name
from config.settings import dynamic_subagents_enabled

QUICKJS_PTC_TOOLS = ("ls", "read_file", "glob", "grep")

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
    - execute(command="python tmp.py")  # For a workspace file shown by file tools as /tmp.py
    - execute(command="python .\\tmp.py")  # Windows-friendly form for /tmp.py
    - execute(command="python scripts/check_path.py")  # For /scripts/check_path.py
    - execute(command="pytest tests")
    - execute(command="npm install && npm test")
    - execute(command="make build", timeout=300)

  Bad examples (avoid these):
    - execute(command="python /tmp.py")  # /tmp.py is a virtual workspace path, not a host shell path
    - execute(command="python /scripts/check_path.py")  # Convert virtual paths to workspace-relative shell paths
    - execute(command="cd /foo/bar && pytest tests")  # Use an explicit path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution
(SandboxBackendProtocol). If execution is not supported, the tool will return an
error message.
"""


@dataclass(frozen=True)
class AgentMiddlewareStack:
    """Middleware items plus the summarization instance MIRA observes."""

    items: list[Any]
    summarization: Any


def build_agent_middleware(
    *,
    model: Any,
    backend: Any,
    workspace: Path,
    settings: dict[str, Any] | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> AgentMiddlewareStack:
    """Build MIRA's user middleware stack for DeepAgents."""
    summarization_middleware = create_mira_summarization_middleware(model=model, backend=backend)
    summarization_tool_middleware = create_mira_summarization_tool_middleware(model=model, backend=backend)
    middleware: list[Any] = [
        summarization_middleware,
        ModelResponseNormalizationMiddleware(Path(workspace)),
        ProviderContextOverflowMiddleware(),
        CodeInterpreterMiddleware(
            ptc=list(QUICKJS_PTC_TOOLS),
            subagents=dynamic_subagents_enabled(settings),
        ),
        summarization_tool_middleware,
        ExecuteToolPromptMiddleware(),
    ]
    middleware.extend(extra_middleware or [])
    return AgentMiddlewareStack(items=middleware, summarization=summarization_middleware)


class ExecuteToolPromptMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace the visible execute tool description with MIRA guidance."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Rewrite execute tool descriptions for synchronous model calls."""
        return handler(self._rewrite_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Rewrite execute tool descriptions for asynchronous model calls."""
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
        description = getattr(tool, "description", None)
        if description == MIRA_EXECUTE_TOOL_DESCRIPTION:
            return tool
        return model_copy(update={"description": MIRA_EXECUTE_TOOL_DESCRIPTION})

    try:
        copied = copy.copy(tool)
        setattr(copied, "description", MIRA_EXECUTE_TOOL_DESCRIPTION)
        return copied
    except Exception:
        return tool


class ModelToolVisibilityMiddleware(AgentMiddleware[Any, Any, Any]):
    """Middleware that hides selected tools from model calls."""

    def __init__(self, excluded_tools: tuple[str, ...]) -> None:
        """Store tool names that should be hidden from the model."""
        self.excluded_tools = set(excluded_tools)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter tools for synchronous LangChain model calls."""
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter tools for asynchronous LangChain model calls."""
        return await handler(self._filter_request(request))

    def _filter_request(self, request: Any) -> Any:
        """Return a request copy with excluded tools removed."""
        tools = [tool for tool in request.tools if resource_tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


class ModelResponseNormalizationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Correct known model-response incompatibilities before MIRA uses them.

    This fixes two issues:

    1. ChatAnyLLM omits the model provider metadata that DeepAgents needs
       to validate token usage before compacting the conversation.

    2. Models sometimes use ``path`` instead of ``file_path``, or return a
       host path that MIRA's filesystem tools cannot use directly.

    These values are corrected before tool approval and execution.
    """

    MODEL_PROVIDER = "anyllm"
    FILE_PATH_TOOLS = {"read_file", "write_file", "edit_file"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()

    def wrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        """Normalize the result of a synchronous model call."""
        response = handler(request)
        self._normalize_response(response)
        return response

    async def awrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        """Normalize the result of an asynchronous model call."""
        response = await handler(request)
        self._normalize_response(response)
        return response

    def _normalize_response(self, response: ModelResponse[Any]) -> None:
        """Apply compatibility corrections to every returned AI message."""
        for message in response.result:
            # Only assistant messages contain the metadata and model-made tool
            # calls that need these compatibility corrections.
            if not isinstance(message, AIMessage):
                continue
            self._normalize_metadata(message)
            self._normalize_file_tool_calls(message)

    def _normalize_metadata(self, message: AIMessage) -> None:
        """Fill response metadata omitted by ChatAnyLLM."""
        # ChatAnyLLM reports token usage but omits the matching provider name.
        # DeepAgents requires both before it trusts that usage for compaction.
        message.response_metadata.setdefault("model_provider", self.MODEL_PROVIDER)

    def _normalize_file_tool_calls(self, message: AIMessage) -> None:
        """Normalize file-tool arguments in canonical calls and content."""
        if not message.tool_calls:
            return

        changed = False
        normalized_calls = []
        for call in message.tool_calls:
            normalized, call_changed = self._normalize_tool_call(call)
            normalized_calls.append(normalized)
            changed = changed or call_changed

        # LangChain may store tool calls both in the canonical list and in
        # content blocks. Keep the two representations consistent.
        normalized_content, content_changed = self._normalize_content_blocks(message.content)
        changed = changed or content_changed

        if not changed:
            return

        message.tool_calls = normalized_calls
        if content_changed:
            message.content = normalized_content

    def _normalize_tool_call(self, call: Any) -> tuple[Any, bool]:
        if not isinstance(call, dict):
            return call, False
        if call.get("name") not in self.FILE_PATH_TOOLS:
            return call, False

        args = call.get("args")
        if not isinstance(args, dict):
            return call, False

        normalized_args = dict(args)
        changed = False
        # Models commonly guess `path`, while DeepAgents' filesystem tools
        # require the schema-defined argument name `file_path`.
        if "file_path" not in normalized_args and "path" in normalized_args:
            normalized_args["file_path"] = normalized_args.pop("path")
            changed = True

        file_path = normalized_args.get("file_path")
        if isinstance(file_path, str):
            normalized_path = self._normalize_workspace_path(file_path)
            if normalized_path != file_path:
                normalized_args["file_path"] = normalized_path
                changed = True

        if not changed:
            return call, False

        return {**call, "args": normalized_args}, True

    def _normalize_content_blocks(self, content: Any) -> tuple[Any, bool]:
        if not isinstance(content, list):
            return content, False

        changed = False
        normalized_blocks = []
        for block in content:
            normalized, block_changed = self._normalize_tool_call(block)
            normalized_blocks.append(normalized)
            changed = changed or block_changed
        return normalized_blocks, changed

    def _normalize_workspace_path(self, value: str) -> str:
        # Leading-slash paths are already MIRA/DeepAgents virtual paths.
        if value.startswith("/"):
            return value
        try:
            path = Path(value).expanduser()
        except (OSError, RuntimeError):
            return value
        # Relative paths are already usable and should remain unchanged.
        if not path.is_absolute():
            return value

        try:
            relative = path.resolve().relative_to(self.workspace)
        except (OSError, RuntimeError, ValueError):
            # Never redirect an absolute path from outside the workspace.
            return value
        # DeepAgents' filesystem backend expects a workspace-relative virtual
        # path rather than the host path returned by the model.
        return f"/{relative.as_posix()}"


__all__ = [
    "AgentMiddlewareStack",
    "ExecuteToolPromptMiddleware",
    "MIRA_EXECUTE_TOOL_DESCRIPTION",
    "ModelResponseNormalizationMiddleware",
    "ModelToolVisibilityMiddleware",
    "QUICKJS_PTC_TOOLS",
    "build_agent_middleware",
    "execute_tool_with_mira_description",
]
