"""Tests for MIRA custom middleware."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from agent.middleware import (
    ExecuteToolPromptMiddleware,
    MIRA_EXECUTE_TOOL_DESCRIPTION,
    ModelResponseNormalizationMiddleware,
)


class FakeModelRequest:
    """Small model request double with overridable tools."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools

    def override(self, **kwargs: Any) -> "FakeModelRequest":
        return FakeModelRequest(kwargs.get("tools", self.tools))


class AnyLLMMetadataModel:
    """Minimal model identity used by LangChain's reported-token check."""

    def _get_ls_params(self) -> dict[str, str]:
        return {"ls_provider": "anyllm"}


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

    def test_model_response_normalizer_adds_missing_anyllm_provider(self) -> None:
        """ChatAnyLLM messages should gain the provider identity DeepAgents expects."""
        message = AIMessage(
            content="done",
            response_metadata={"model_name": "local-model"},
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        response = ModelResponse(result=[message])
        middleware = ModelResponseNormalizationMiddleware(Path("."))

        normalized = middleware.wrap_model_call(None, lambda _request: response)

        self.assertIs(normalized, response)
        self.assertEqual(message.response_metadata["model_provider"], "anyllm")
        self.assertEqual(message.response_metadata["model_name"], "local-model")
        self.assertEqual(message.usage_metadata["total_tokens"], 120)

    def test_model_response_normalizer_preserves_existing_provider_and_non_ai_messages(self) -> None:
        """Provider-owned metadata and non-AI messages must remain unchanged."""
        message = AIMessage(content="done", response_metadata={"model_provider": "openai"})
        human = HumanMessage(content="hello")
        response = ModelResponse(result=[human, message])
        middleware = ModelResponseNormalizationMiddleware(Path("."))

        middleware.wrap_model_call(None, lambda _request: response)

        self.assertEqual(message.response_metadata["model_provider"], "openai")
        self.assertEqual(human.response_metadata, {})

    def test_normalized_metadata_passes_reported_token_provider_check(self) -> None:
        """The compatibility field should unlock above-threshold reported usage."""
        message = AIMessage(
            content="done",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        response = ModelResponse(result=[message])
        ModelResponseNormalizationMiddleware(Path(".")).wrap_model_call(
            None,
            lambda _request: response,
        )
        summarization = SummarizationMiddleware(
            model=AnyLLMMetadataModel(),
            trigger=("tokens", 200),
            keep=("messages", 1),
            token_counter=lambda _messages: 0,
        )

        eligible = summarization._should_summarize_based_on_reported_tokens(  # noqa: SLF001
            [message],
            threshold=100,
        )

        self.assertTrue(eligible)


class AsyncMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """Asynchronous custom middleware behavior."""

    async def test_model_response_normalizer_handles_async_calls(self) -> None:
        """Async model responses should receive the same metadata correction."""
        message = AIMessage(content="done")
        response = ModelResponse(result=[message])
        middleware = ModelResponseNormalizationMiddleware(Path("."))

        async def handler(_request: Any) -> ModelResponse[Any]:
            return response

        normalized = await middleware.awrap_model_call(None, handler)

        self.assertIs(normalized, response)
        self.assertEqual(message.response_metadata["model_provider"], "anyllm")


if __name__ == "__main__":
    unittest.main()
