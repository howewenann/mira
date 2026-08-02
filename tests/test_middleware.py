"""Tests for MIRA custom middleware."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from typing import Any

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.middleware import (
    CORRECTION_SOURCE,
    CorrectionDecision,
    CorrectionMiddleware,
    ExecuteToolPromptMiddleware,
    MIRA_EXECUTE_TOOL_DESCRIPTION,
    ModelResponseNormalizationMiddleware,
    PlanningStageMiddleware,
)
from agent.planning.next_action import (
    PLANNING_NEXT_ACTION_ANSWER,
    PLANNING_NEXT_ACTION_FAILURE,
    PLANNING_NEXT_ACTION_PREPARE_GOAL,
    PLANNING_NEXT_ACTION_PREPARE_PLAN,
    PLANNING_NEXT_ACTION_RESEARCH,
    PlanningNextActionRule,
    planning_next_action,
    strip_planning_next_action,
)


class FakeModelRequest:
    """Small model request double with overridable tools."""

    def __init__(
        self,
        tools: list[Any],
        *,
        state: dict[str, Any] | None = None,
        tool_choice: Any = None,
        system_message: SystemMessage | None = None,
    ) -> None:
        self.tools = tools
        self.state = state or {}
        self.tool_choice = tool_choice
        self.system_message = system_message

    def override(self, **kwargs: Any) -> "FakeModelRequest":
        return FakeModelRequest(
            kwargs.get("tools", self.tools),
            state=kwargs.get("state", self.state),
            tool_choice=kwargs.get("tool_choice", self.tool_choice),
            system_message=kwargs.get("system_message", self.system_message),
        )


class FakeToolRequest:
    """Small tool request double carrying graph state."""

    def __init__(self, name: str, stage: str, *, call_id: str = "call-control") -> None:
        self.tool_call = {"name": name, "args": {}, "id": call_id}
        self.state = {"planning_stage": stage}


class AnyLLMMetadataModel:
    """Minimal model identity used by LangChain's reported-token check."""

    def _get_ls_params(self) -> dict[str, str]:
        return {"ls_provider": "anyllm"}


class ReviewCompletionRule:
    """Non-planning rule proving that correction owns no workflow vocabulary."""

    protocol_id = "review_completion"
    workflow_label = "Review"
    failure_text = "Review could not be completed."

    def applies(self, state: dict[str, Any]) -> bool:
        return state.get("review_active") is True

    def reminder(self, state: dict[str, Any]) -> str:  # noqa: ARG002
        return "End a completed review with REVIEW_COMPLETE."

    def inspect(self, message: AIMessage, state: dict[str, Any]) -> CorrectionDecision:  # noqa: ARG002
        if str(message.text).endswith("REVIEW_COMPLETE"):
            return CorrectionDecision(accepted=True)
        return CorrectionDecision(
            accepted=False,
            failed_check="The review was not classified as complete.",
            retry_prompt="Finish the review and append REVIEW_COMPLETE.",
        )


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

    def test_plan_research_stage_hides_present_plan(self) -> None:
        middleware = PlanningStageMiddleware()
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "ask_user"},
                {"name": "prepare_plan"},
                {"name": "present_plan"},
            ],
            state={"planning_stage": "plan_research"},
            tool_choice="previous",
        )

        updated = middleware._stage_request(request)

        self.assertEqual(
            [tool["name"] for tool in updated.tools],
            ["read_file", "ask_user", "prepare_plan"],
        )
        self.assertIsNone(updated.tool_choice)
        self.assertIsNone(updated.system_message)

    def test_goal_research_stage_exposes_only_goal_prepare_control(self) -> None:
        middleware = PlanningStageMiddleware()
        request = FakeModelRequest(
            [{"name": "prepare_goal"}, {"name": "prepare_plan"}],
            state={"planning_stage": "goal_research"},
            system_message=SystemMessage(content="base"),
        )

        updated = middleware._stage_request(request)

        self.assertEqual([tool["name"] for tool in updated.tools], ["prepare_goal"])
        self.assertIsNone(updated.tool_choice)
        self.assertEqual(updated.system_message.text, "base")
        self.assertEqual(request.system_message.text, "base")

    def test_async_correction_reminder_injection_matches_sync(self) -> None:
        middleware = CorrectionMiddleware(
            rules=(PlanningNextActionRule(workflow="plan"),),
            max_retries=2,
        )
        request = FakeModelRequest(
            [{"name": "read_file"}, {"name": "prepare_plan"}],
            state={"planning_stage": "plan_research"},
        )
        captured: dict[str, Any] = {}

        async def handler(updated: FakeModelRequest) -> str:
            captured["request"] = updated
            return "ok"

        result = asyncio.run(middleware.awrap_model_call(request, handler))

        self.assertEqual(result, "ok")
        self.assertIn(
            PLANNING_NEXT_ACTION_PREPARE_PLAN,
            str(captured["request"].system_message.text),
        )

    def test_generic_correction_selects_a_non_planning_rule(self) -> None:
        events: list[dict[str, Any]] = []
        runtime = type("Runtime", (), {"stream_writer": events.append})()
        middleware = CorrectionMiddleware(rules=(ReviewCompletionRule(),), max_retries=2)
        request = FakeModelRequest([], state={"review_active": True})
        captured: dict[str, Any] = {}

        def handler(updated: FakeModelRequest) -> str:
            captured["request"] = updated
            return "ok"

        self.assertEqual(middleware.wrap_model_call(request, handler), "ok")
        self.assertIn("REVIEW_COMPLETE", str(captured["request"].system_message.text))

        retry = middleware.after_agent(
            {"review_active": True, "messages": [AIMessage(content="Still reviewing.")]},
            runtime,
        )

        self.assertEqual(retry["_correction_retries"], {"review_completion": 1})
        self.assertEqual(events[-1]["workflow"], "Review")
        self.assertEqual(retry["messages"][-1].name, CORRECTION_SOURCE)

    def test_correction_reminders_are_stage_specific_transient_and_skip_finalization(self) -> None:
        middleware = CorrectionMiddleware(
            rules=(
                PlanningNextActionRule(workflow="plan"),
                PlanningNextActionRule(workflow="goal"),
            ),
            max_retries=2,
        )
        base = SystemMessage(content="base")

        plan = middleware._request_with_reminders(
            FakeModelRequest([], state={"planning_stage": "plan_research"}, system_message=base)
        )
        goal = middleware._request_with_reminders(
            FakeModelRequest([], state={"planning_stage": "goal_research"}, system_message=base)
        )
        final = middleware._request_with_reminders(
            FakeModelRequest([], state={"planning_stage": "plan_finalize"}, system_message=base)
        )
        plan_again = middleware._request_with_reminders(
            FakeModelRequest([], state={"planning_stage": "plan_research"}, system_message=base)
        )

        self.assertIn("plan_show", str(plan.system_message.text))
        self.assertNotIn("goal_show", str(plan.system_message.text))
        self.assertIn("goal_show", str(goal.system_message.text))
        self.assertNotIn("plan_show", str(goal.system_message.text))
        self.assertEqual(final.system_message.text, "base")
        self.assertEqual(base.text, "base")
        self.assertEqual(plan.system_message.text, plan_again.system_message.text)
        self.assertEqual(str(plan.system_message.text).count(PLANNING_NEXT_ACTION_PREPARE_PLAN), 2)

    def test_plan_finalize_stage_forces_present_plan_only(self) -> None:
        middleware = PlanningStageMiddleware()
        request = FakeModelRequest(
            [{"name": "read_file"}, {"name": "prepare_plan"}, {"name": "present_plan"}],
            state={"planning_stage": "plan_finalize"},
        )

        updated = middleware._stage_request(request)

        self.assertEqual([tool["name"] for tool in updated.tools], ["present_plan"])
        self.assertEqual(updated.tool_choice, "required")

    def test_plan_finalize_requires_registered_present_plan(self) -> None:
        request = FakeModelRequest(
            [{"name": "prepare_plan"}],
            state={"planning_stage": "plan_finalize"},
        )

        with self.assertRaisesRegex(RuntimeError, "requires present_plan"):
            PlanningStageMiddleware()._stage_request(request)

    def test_goal_finalize_stage_forces_present_goal_only(self) -> None:
        middleware = PlanningStageMiddleware()
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "present_goal"},
                {"name": "present_plan"},
            ],
            state={"planning_stage": "goal_finalize"},
        )

        updated = middleware._stage_request(request)

        self.assertEqual([tool["name"] for tool in updated.tools], ["present_goal"])
        self.assertEqual(updated.tool_choice, "required")
        self.assertIsNone(updated.system_message)

    def test_goal_finalize_requires_registered_present_goal(self) -> None:
        request = FakeModelRequest(
            [{"name": "present_plan"}],
            state={"planning_stage": "goal_finalize"},
        )

        with self.assertRaisesRegex(RuntimeError, "requires present_goal"):
            PlanningStageMiddleware()._stage_request(request)

    def test_wrong_stage_formal_controls_return_native_tool_errors(self) -> None:
        middleware = PlanningStageMiddleware()
        cases = (
            ("prepare_plan", "goal_research", "requires plan_research"),
            ("present_plan", "plan_research", "plan_show"),
            ("prepare_goal", "plan_research", "requires goal_research"),
            ("present_goal", "goal_research", "goal_show"),
        )

        for name, stage, expected in cases:
            with self.subTest(name=name, stage=stage):
                called = False

                def handler(_request: Any) -> str:
                    nonlocal called
                    called = True
                    return "executed"

                result = middleware.wrap_tool_call(FakeToolRequest(name, stage), handler)

                self.assertIsInstance(result, ToolMessage)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.name, name)
                self.assertIn(expected, str(result.content))
                self.assertFalse(called)

    def test_valid_stage_formal_control_executes_handler_sync_and_async(self) -> None:
        middleware = PlanningStageMiddleware()
        request = FakeToolRequest("present_plan", "plan_finalize")

        self.assertEqual(middleware.wrap_tool_call(request, lambda _request: "sync"), "sync")

        async def handler(_request: Any) -> str:
            return "async"

        self.assertEqual(asyncio.run(middleware.awrap_tool_call(request, handler)), "async")

    def test_async_wrong_stage_formal_control_returns_error_without_handler(self) -> None:
        middleware = PlanningStageMiddleware()
        called = False

        async def handler(_request: Any) -> str:
            nonlocal called
            called = True
            return "executed"

        result = asyncio.run(
            middleware.awrap_tool_call(
                FakeToolRequest("present_goal", "goal_research"),
                handler,
            )
        )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertIn("goal_show", str(result.content))
        self.assertFalse(called)

    def test_next_action_parser_requires_one_terminal_stage_valid_marker(self) -> None:
        plan = "plan_research"
        goal = "goal_research"
        self.assertEqual(
            planning_next_action(AIMessage(content="Complete.\nNEXT_ACTION: ANSWER"), plan),
            PLANNING_NEXT_ACTION_ANSWER,
        )
        self.assertEqual(
            planning_next_action(AIMessage(content="Need files.\nNEXT_ACTION: RESEARCH\n\n"), plan),
            PLANNING_NEXT_ACTION_RESEARCH,
        )
        self.assertIsNone(
            planning_next_action(AIMessage(content="NEXT_ACTION: PREPARE_GOAL"), plan)
        )
        self.assertEqual(
            planning_next_action(AIMessage(content="NEXT_ACTION: PREPARE_GOAL"), goal),
            PLANNING_NEXT_ACTION_PREPARE_GOAL,
        )
        for content in (
            "missing",
            "NEXT_ACTION: ANSWER\nmore",
            "NEXT_ACTION: ANSWER\nNEXT_ACTION: ANSWER",
            "NEXT_ACTION: ANSWER ",
            [{"type": "reasoning", "reasoning": "thinking"}],
        ):
            self.assertIsNone(planning_next_action(AIMessage(content=content), plan))

    def test_accepted_structured_answer_strips_only_marker_and_preserves_metadata(self) -> None:
        message = AIMessage(
            id="answer-1",
            content=[
                {"type": "reasoning", "reasoning": "internal"},
                {"type": "text", "text": "答案。\n"},
                {"type": "text", "text": "NEXT_ACTION: ANSWER\n\n"},
            ],
            response_metadata={"model_provider": "test"},
        )

        cleaned = strip_planning_next_action(message)

        self.assertEqual(cleaned.id, "answer-1")
        self.assertEqual(cleaned.response_metadata, {"model_provider": "test"})
        self.assertEqual(str(cleaned.text), "答案。\n\n")
        self.assertEqual(cleaned.content[0], message.content[0])
        self.assertNotIn("NEXT_ACTION", str(cleaned.content))

    def test_natural_stop_routes_retry_acceptance_and_exhaustion(self) -> None:
        events: list[dict[str, Any]] = []
        runtime = type("Runtime", (), {"stream_writer": events.append})()
        middleware = CorrectionMiddleware(
            rules=(PlanningNextActionRule(workflow="plan"),),
            max_retries=1,
        )
        invalid = AIMessage(
            id="invalid-1",
            content="I'll inspect next.\nNEXT_ACTION: RESEARCH",
        )

        retry = middleware.after_agent(
            {"planning_stage": "plan_research", "messages": [invalid]},
            runtime,
        )

        self.assertEqual(retry["jump_to"], "model")
        self.assertEqual(retry["_correction_retries"], {"plan_next_action": 1})
        self.assertEqual(retry["messages"][-1].additional_kwargs["lc_source"], CORRECTION_SOURCE)
        self.assertIn("Perform that research now", retry["messages"][-1].content)
        self.assertEqual(events[-1]["failed_check"], "NEXT_ACTION: RESEARCH was declared, but no research tool was called.")

        correction = retry["messages"][-1].model_copy(update={"id": "correction-1"})
        accepted = AIMessage(id="answer-1", content="Done.\nNEXT_ACTION: ANSWER")
        success = middleware.after_agent(
            {
                "planning_stage": "plan_research",
                "messages": [correction, accepted],
                "_correction_retries": {"plan_next_action": 1},
            },
            runtime,
        )
        self.assertEqual(success["_correction_retries"], {})
        self.assertEqual(success["messages"][-1].content, "Done.\n")

        exhausted = middleware.after_agent(
            {
                "planning_stage": "plan_research",
                "messages": [invalid],
                "_correction_retries": {"plan_next_action": 1},
            },
            runtime,
        )
        self.assertNotIn("jump_to", exhausted)
        self.assertIn("may be incomplete", exhausted["messages"][-1].content)
        self.assertTrue(events[-1]["exhausted"])
        self.assertEqual(exhausted["messages"][-1].content, PLANNING_NEXT_ACTION_FAILURE)

    def test_tool_call_bypasses_protocol_and_clears_feedback(self) -> None:
        correction = HumanMessage(
            id="correction-1",
            content="retry",
            additional_kwargs={"lc_source": CORRECTION_SOURCE},
        )
        tool_call = AIMessage(
            id="tool-1",
            content="I'll inspect.",
            tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}],
        )

        update = CorrectionMiddleware(
            rules=(PlanningNextActionRule(workflow="goal"),),
            max_retries=2,
        ).after_model(
            {
                "planning_stage": "goal_research",
                "messages": [correction, tool_call],
                "_correction_retries": {"goal_next_action": 1},
            },
            None,
        )

        self.assertEqual(update["_correction_retries"], {})
        self.assertNotIn("messages", update)

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
