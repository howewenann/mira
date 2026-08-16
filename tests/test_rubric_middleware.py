"""Focused tests for MIRA rubric tools and nested approvals."""

from __future__ import annotations

import unittest
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import Field

from agent.middleware.rubric import MiraRubricMiddleware


class FixedFakeChatModel(GenericFakeChatModel):
    """Deterministic model whose structured-output binding returns itself."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def grader_response() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GraderResponse",
                "args": {
                    "result": "satisfied",
                    "explanation": "current state verified",
                    "criteria": [{"name": "state current", "passed": True}],
                },
                "id": "grader-response",
                "type": "tool_call",
            }
        ],
    )


class RubricMiddlewareTests(unittest.TestCase):
    """Nested rubric HITL should behave like normal MIRA tool approval."""

    def _agent(self, observed: list[str], *, require_approval: bool) -> Any:
        @tool
        def inspect_external(resource_id: str) -> str:
            """Inspect an external resource."""
            observed.append(resource_id)
            return "current"

        main_model = FixedFakeChatModel(messages=iter([AIMessage(content="done")]))
        grader_model = FixedFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "inspect_external",
                                "args": {"resource_id": "page-123"},
                                "id": "inspect-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    grader_response(),
                ]
            )
        )
        nested = (
            [
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "inspect_external": {"allowed_decisions": ["approve", "edit", "reject"]}
                    }
                )
            ]
            if require_approval
            else []
        )
        rubric = MiraRubricMiddleware(
            model=grader_model,
            tools=[inspect_external],
            grader_middleware=nested,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return create_deep_agent(
                model=main_model,
                middleware=[rubric],
                checkpointer=InMemorySaver(),
            )

    def test_always_allow_tools_bypass_nested_approval(self) -> None:
        observed: list[str] = []
        agent = self._agent(observed, require_approval=False)
        config = {"configurable": {"thread_id": "rubric-direct"}}
        result = agent.invoke(
            {"messages": [HumanMessage(content="finish")], "rubric": "state current"},
            config=config,
        )

        self.assertNotIn("__interrupt__", result)
        self.assertEqual(observed, ["page-123"])
        self.assertEqual(agent.get_state(config).values["_rubric_status"], "satisfied")

    def test_approval_resumes_nested_grader_without_grader_error(self) -> None:
        observed: list[str] = []
        agent = self._agent(observed, require_approval=True)
        config = {"configurable": {"thread_id": "rubric-approve"}}
        first = agent.invoke(
            {"messages": [HumanMessage(content="finish")], "rubric": "state current"},
            config=config,
        )
        interrupt = first["__interrupt__"][0]
        self.assertEqual(interrupt.value["action_requests"][0]["name"], "inspect_external")
        self.assertEqual(
            interrupt.value["review_configs"][0]["allowed_decisions"],
            ["approve", "edit", "reject"],
        )

        result = agent.invoke(
            Command(resume={interrupt.id: {"decisions": [{"type": "approve"}]}}),
            config=config,
        )

        self.assertEqual(observed, ["page-123"])
        state = agent.get_state(config).values
        self.assertEqual(state["_rubric_status"], "satisfied")
        self.assertNotEqual(state["_rubric_evaluations"][-1]["result"], "grader_error")

    def test_rejection_returns_to_grader_without_executing_tool(self) -> None:
        observed: list[str] = []
        agent = self._agent(observed, require_approval=True)
        config = {"configurable": {"thread_id": "rubric-reject"}}
        first = agent.invoke(
            {"messages": [HumanMessage(content="finish")], "rubric": "state current"},
            config=config,
        )
        interrupt = first["__interrupt__"][0]

        result = agent.invoke(
            Command(
                resume={interrupt.id: {"decisions": [{"type": "reject", "message": "do not inspect"}]}},
            ),
            config=config,
        )

        self.assertEqual(observed, [])
        state = agent.get_state(config).values
        self.assertEqual(state["_rubric_status"], "satisfied")
        self.assertNotEqual(state["_rubric_evaluations"][-1]["result"], "grader_error")

    def test_graph_interrupt_is_not_converted_to_grader_error(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        interrupt = GraphInterrupt(())
        with self.assertRaises(GraphInterrupt):
            middleware._handle_grader_exception(None, {}, "run", 0, interrupt)
