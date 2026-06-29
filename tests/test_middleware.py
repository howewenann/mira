"""Tests for MIRA custom middleware."""

from __future__ import annotations

import unittest
from typing import Any

from agent.middleware import (
    ExecuteToolPromptMiddleware,
    MIRA_EXECUTE_TOOL_DESCRIPTION,
)


class FakeModelRequest:
    """Small model request double with overridable tools."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools

    def override(self, **kwargs: Any) -> "FakeModelRequest":
        return FakeModelRequest(kwargs.get("tools", self.tools))


class MiddlewareTests(unittest.TestCase):
    """Custom middleware behavior."""

    def test_mira_execute_description_keeps_deepagents_guidance(self) -> None:
        """MIRA's prompt should stay close to the original execute guidance."""
        self.assertIn("Executes a shell command", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("Before executing the command, please follow these steps:", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("1. Directory Verification:", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("3. Command Execution:", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('cd "/Users/name/My Documents" (correct', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("Usage notes:", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="make build", timeout=300)', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("Bad examples", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("cat file.txt", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("find . -name '*.py'", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("grep -r 'pattern' .", MIRA_EXECUTE_TOOL_DESCRIPTION)

    def test_mira_execute_description_hardens_virtual_workspace_paths(self) -> None:
        """The execute prompt should make virtual path conversion explicit."""
        self.assertIn("2. MIRA Workspace Path Handling:", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("file tools use virtual workspace paths", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("commands run in the host shell from the project workspace", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("use `python tmp.py` or", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn("not `python /tmp.py`", MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="python tmp.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="python .\\tmp.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="python /tmp.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="python scripts/check_path.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertIn('execute(command="python /scripts/check_path.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertNotIn('    - execute(command="python /path/to/script.py")', MIRA_EXECUTE_TOOL_DESCRIPTION)

    def test_execute_prompt_middleware_rewrites_only_execute_tool(self) -> None:
        """Only the visible execute tool description should be replaced."""
        middleware = ExecuteToolPromptMiddleware()
        execute_tool = {"name": "execute", "description": "old execute"}
        grep_tool = {"name": "grep", "description": "keep grep"}
        request = FakeModelRequest([execute_tool, grep_tool])
        captured: dict[str, Any] = {}

        def handler(updated: FakeModelRequest) -> str:
            captured["request"] = updated
            return "ok"

        self.assertEqual(middleware.wrap_model_call(request, handler), "ok")

        updated_tools = captured["request"].tools
        self.assertEqual(updated_tools[0]["description"], MIRA_EXECUTE_TOOL_DESCRIPTION)
        self.assertEqual(updated_tools[1], grep_tool)
        self.assertEqual(execute_tool["description"], "old execute")


if __name__ == "__main__":
    unittest.main()
