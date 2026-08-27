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
    ExecuteToolDescriptionRewriteMiddleware,
    MIRA_EXECUTE_TOOL_DESCRIPTION,
    ModelResponseNormalizationMiddleware,
    PlanningStageEnforcementMiddleware,
)
from agent.planning.response_status import (
    PLANNING_RESPONSE_STATUS_COMPLETE,
    PLANNING_RESPONSE_STATUS_FAILURE,
    PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH,
    PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL,
    PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN,
    PlanningResponseStatusRule,
    planning_response_status,
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

    def with_retry(self) -> "AnyLLMMetadataModel":
        """Match the chat-model construction seam required by LangChain 1.3.17."""
        return self

    def _get_ls_params(self) -> dict[str, str]:
        return {"ls_provider": "anyllm"}


class ReviewCompletionRule:
    """Non-planning rule proving that correction owns no workflow vocabulary."""

    protocol_id = "review_completion"
    check_name = "Review"
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
        middleware = ExecuteToolDescriptionRewriteMiddleware()
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

    def test_plan_research_stage_hides_finalize_plan(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "ask_user"},
                {"name": "prepare_plan"},
                {"name": "finalize_plan"},
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
        middleware = PlanningStageEnforcementMiddleware()
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
            rules=(PlanningResponseStatusRule(workflow="plan"),),
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
            PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN,
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
        self.assertEqual(events[-1]["check_name"], "Review")
        self.assertEqual(events[-1]["workflow"], "Review")
        self.assertEqual(retry["messages"][-1].name, CORRECTION_SOURCE)

    def test_correction_reminders_are_stage_specific_transient_and_skip_finalization(self) -> None:
        middleware = CorrectionMiddleware(
            rules=(
                PlanningResponseStatusRule(workflow="plan"),
                PlanningResponseStatusRule(workflow="goal"),
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
            FakeModelRequest(
                [],
                state={
                    "planning_stage": "plan_research",
                    "messages": [
                        ToolMessage(
                            content="README contents",
                            tool_call_id="call-readme",
                        )
                    ],
                },
                system_message=base,
            )
        )

        self.assertIn("show_plan", str(plan.system_message.text))
        self.assertNotIn("show_goal", str(plan.system_message.text))
        self.assertIn("show_goal", str(goal.system_message.text))
        self.assertNotIn("show_plan", str(goal.system_message.text))
        self.assertIn("require an immediate\nshow_plan call", str(plan.system_message.text))
        self.assertIn("without research, prose reproduction", str(plan.system_message.text))
        self.assertEqual(final.system_message.text, "base")
        self.assertEqual(base.text, "base")
        self.assertEqual(plan.system_message.text, plan_again.system_message.text)
        self.assertEqual(
            str(plan.system_message.text).count(
                PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_PLAN
            ),
            2,
        )

    def test_plan_finalize_stage_forces_finalize_plan_only(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        request = FakeModelRequest(
            [{"name": "read_file"}, {"name": "prepare_plan"}, {"name": "finalize_plan"}],
            state={"planning_stage": "plan_finalize"},
        )

        updated = middleware._stage_request(request)

        self.assertEqual([tool["name"] for tool in updated.tools], ["finalize_plan"])
        self.assertEqual(updated.tool_choice, "required")

    def test_plan_finalize_requires_registered_finalize_plan(self) -> None:
        request = FakeModelRequest(
            [{"name": "prepare_plan"}],
            state={"planning_stage": "plan_finalize"},
        )

        with self.assertRaisesRegex(RuntimeError, "requires finalize_plan"):
            PlanningStageEnforcementMiddleware()._stage_request(request)

    def test_goal_finalize_stage_forces_finalize_goal_only(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "finalize_goal"},
                {"name": "finalize_plan"},
            ],
            state={"planning_stage": "goal_finalize"},
        )

        updated = middleware._stage_request(request)

        self.assertEqual([tool["name"] for tool in updated.tools], ["finalize_goal"])
        self.assertEqual(updated.tool_choice, "required")
        self.assertIsNone(updated.system_message)

    def test_goal_finalize_requires_registered_finalize_goal(self) -> None:
        request = FakeModelRequest(
            [{"name": "finalize_plan"}],
            state={"planning_stage": "goal_finalize"},
        )

        with self.assertRaisesRegex(RuntimeError, "requires finalize_goal"):
            PlanningStageEnforcementMiddleware()._stage_request(request)

    def test_wrong_stage_formal_controls_return_native_tool_errors(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        cases = (
            ("prepare_plan", "goal_research", "requires plan_research"),
            ("finalize_plan", "plan_research", "show_plan"),
            ("prepare_goal", "plan_research", "requires goal_research"),
            ("finalize_goal", "goal_research", "show_goal"),
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
        middleware = PlanningStageEnforcementMiddleware()
        request = FakeToolRequest("finalize_plan", "plan_finalize")

        self.assertEqual(middleware.wrap_tool_call(request, lambda _request: "sync"), "sync")

        async def handler(_request: Any) -> str:
            return "async"

        self.assertEqual(asyncio.run(middleware.awrap_tool_call(request, handler)), "async")

    def test_async_wrong_stage_formal_control_returns_error_without_handler(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        called = False

        async def handler(_request: Any) -> str:
            nonlocal called
            called = True
            return "executed"

        result = asyncio.run(
            middleware.awrap_tool_call(
                FakeToolRequest("finalize_goal", "goal_research"),
                handler,
            )
        )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertIn("show_goal", str(result.content))
        self.assertFalse(called)

    def test_finalization_rejects_every_name_except_matching_finalizer(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        cases = (
            ("plan_finalize", "finalize_plan", ("ask_user", "read_file", "finalize_goal", "stale_tool")),
            ("goal_finalize", "finalize_goal", ("ask_user", "read_file", "finalize_plan", "stale_tool")),
        )

        for stage, required, forbidden in cases:
            for name in forbidden:
                with self.subTest(stage=stage, name=name):
                    called = False

                    def handler(_request: Any) -> str:
                        nonlocal called
                        called = True
                        return "executed"

                    result = middleware.wrap_tool_call(
                        FakeToolRequest(name, stage, call_id=f"call-{stage}-{name}"),
                        handler,
                    )

                    self.assertIsInstance(result, ToolMessage)
                    self.assertEqual(result.status, "error")
                    self.assertEqual(result.name, name)
                    self.assertEqual(result.tool_call_id, f"call-{stage}-{name}")
                    self.assertIn(f"Only {required} can be called now.", str(result.content))
                    self.assertIn(f"Call {required}", str(result.content))
                    self.assertFalse(called)

    def test_async_finalization_backstop_rejects_hidden_tool_before_handler(self) -> None:
        middleware = PlanningStageEnforcementMiddleware()
        called = False

        async def handler(_request: Any) -> str:
            nonlocal called
            called = True
            return "executed"

        result = asyncio.run(
            middleware.awrap_tool_call(
                FakeToolRequest("ask_user", "plan_finalize", call_id="call-hidden"),
                handler,
            )
        )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.tool_call_id, "call-hidden")
        self.assertIn("Only finalize_plan can be called now.", str(result.content))
        self.assertFalse(called)

    def test_response_status_parser_requires_one_terminal_stage_valid_status(self) -> None:
        plan = "plan_research"
        goal = "goal_research"
        self.assertEqual(
            planning_response_status(
                AIMessage(content="Complete.\nRESPONSE_STATUS: COMPLETE"), plan
            ),
            PLANNING_RESPONSE_STATUS_COMPLETE,
        )
        self.assertEqual(
            planning_response_status(
                AIMessage(content="Need files.\nRESPONSE_STATUS: NEEDS_RESEARCH\n\n"), plan
            ),
            PLANNING_RESPONSE_STATUS_NEEDS_RESEARCH,
        )
        self.assertIsNone(
            planning_response_status(
                AIMessage(content="RESPONSE_STATUS: READY_TO_PREPARE_GOAL"), plan
            )
        )
        self.assertEqual(
            planning_response_status(
                AIMessage(content="RESPONSE_STATUS: READY_TO_PREPARE_GOAL"), goal
            ),
            PLANNING_RESPONSE_STATUS_READY_TO_PREPARE_GOAL,
        )
        for content in (
            "missing",
            "RESPONSE_STATUS: COMPLETE\nmore",
            "RESPONSE_STATUS: COMPLETE\nRESPONSE_STATUS: COMPLETE",
            "RESPONSE_STATUS: COMPLETE ",
            [{"type": "reasoning", "reasoning": "thinking"}],
        ):
            self.assertIsNone(planning_response_status(AIMessage(content=content), plan))

    def test_accepted_structured_answer_preserves_status_and_metadata(self) -> None:
        message = AIMessage(
            id="answer-1",
            content=[
                {"type": "reasoning", "reasoning": "internal"},
                {"type": "text", "text": "答案。\n"},
                {"type": "text", "text": "RESPONSE_STATUS: COMPLETE\n\n"},
            ],
            response_metadata={"model_provider": "test"},
        )

        decision = PlanningResponseStatusRule(workflow="plan").inspect(message, {})

        self.assertTrue(decision.accepted)
        self.assertEqual(message.id, "answer-1")
        self.assertEqual(message.response_metadata, {"model_provider": "test"})
        self.assertEqual(str(message.text), "答案。\nRESPONSE_STATUS: COMPLETE\n\n")
        self.assertEqual(message.content[0]["reasoning"], "internal")

    def test_non_complete_statuses_produce_action_specific_corrections(self) -> None:
        cases = (
            (
                "plan",
                "RESPONSE_STATUS: NEEDS_RESEARCH",
                "Perform that research now",
            ),
            (
                "plan",
                "RESPONSE_STATUS: NEEDS_USER_INPUT",
                "Call ask_user now",
            ),
            (
                "plan",
                "RESPONSE_STATUS: READY_TO_PREPARE_PLAN",
                "Call prepare_plan now",
            ),
            (
                "goal",
                "RESPONSE_STATUS: READY_TO_PREPARE_GOAL",
                "Call prepare_goal now",
            ),
        )

        for workflow, status, expected in cases:
            with self.subTest(workflow=workflow, status=status):
                decision = PlanningResponseStatusRule(workflow=workflow).inspect(
                    AIMessage(content=f"Draft.\n{status}"),
                    {},
                )
                self.assertFalse(decision.accepted)
                self.assertIn(expected, decision.retry_prompt)

    def test_natural_stop_routes_retry_acceptance_and_exhaustion(self) -> None:
        events: list[dict[str, Any]] = []
        runtime = type("Runtime", (), {"stream_writer": events.append})()
        middleware = CorrectionMiddleware(
            rules=(PlanningResponseStatusRule(workflow="plan"),),
            max_retries=1,
        )
        invalid = AIMessage(
            id="invalid-1",
            content="I'll inspect next.\nRESPONSE_STATUS: NEEDS_RESEARCH",
        )

        retry = middleware.after_agent(
            {"planning_stage": "plan_research", "messages": [invalid]},
            runtime,
        )

        self.assertEqual(retry["jump_to"], "model")
        self.assertEqual(retry["_correction_retries"], {"plan_response_status": 1})
        self.assertEqual(retry["messages"][-1].additional_kwargs["lc_source"], CORRECTION_SOURCE)
        self.assertIn("Perform that research now", retry["messages"][-1].content)
        self.assertEqual(
            events[-1]["failed_check"],
            "RESPONSE_STATUS: NEEDS_RESEARCH was declared, but no research tool was called.",
        )
        self.assertEqual(events[-1]["check_name"], "Response")
        self.assertEqual(events[-1]["workflow"], "Plan")

        correction = retry["messages"][-1].model_copy(update={"id": "correction-1"})
        accepted = AIMessage(id="answer-1", content="Done.\nRESPONSE_STATUS: COMPLETE")
        success = middleware.after_agent(
            {
                "planning_stage": "plan_research",
                "messages": [correction, accepted],
                "_correction_retries": {"plan_response_status": 1},
            },
            runtime,
        )
        self.assertEqual(success["_correction_retries"], {})
        self.assertNotIn("messages", success)
        self.assertEqual(accepted.content, "Done.\nRESPONSE_STATUS: COMPLETE")

        exhausted = middleware.after_agent(
            {
                "planning_stage": "plan_research",
                "messages": [invalid],
                "_correction_retries": {"plan_response_status": 1},
            },
            runtime,
        )
        self.assertNotIn("jump_to", exhausted)
        self.assertIn("may be incomplete", exhausted["messages"][-1].content)
        self.assertTrue(events[-1]["exhausted"])
        self.assertEqual(exhausted["messages"][-1].content, PLANNING_RESPONSE_STATUS_FAILURE)

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
            rules=(PlanningResponseStatusRule(workflow="goal"),),
            max_retries=2,
        ).after_model(
            {
                "planning_stage": "goal_research",
                "messages": [correction, tool_call],
                "_correction_retries": {"goal_response_status": 1},
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
