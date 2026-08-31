"""Focused tests for MIRA's isolated Rubric verifier and final grader."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import rubric as deepagents_rubric
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import Field

from agent.rubric import middleware as mira_rubric
from agent.rubric.middleware import (
    FINAL_GRADER_SYSTEM_PROMPT,
    VERIFICATION_EVIDENCE_MESSAGE,
    VERIFIER_SYSTEM_PROMPT,
    MiraRubricMiddleware,
)
from core.execution.runner import run_turn


def tool_names(tools: Sequence[dict[str, Any] | type | Callable | BaseTool]) -> list[str]:
    return [
        str(getattr(candidate, "name", getattr(candidate, "__name__", "tool")))
        for candidate in tools
    ]


class FixedFakeChatModel(GenericFakeChatModel):
    """Deterministic model that records every provider-facing tool binding."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)
    bindings: list[dict[str, Any]] = Field(default_factory=list, exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bindings.append({"tools": tool_names(tools), "tool_choice": tool_choice})
        return self

    def _stream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream fake tool calls in the normalized chunks real providers emit."""
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        tool_chunks = [
            {
                "name": call.get("name"),
                "args": json.dumps(call.get("args", {})),
                "id": call.get("id"),
                "index": index,
                "type": "tool_call_chunk",
            }
            for index, call in enumerate(message.tool_calls)
        ]
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                id=message.id,
                tool_call_chunks=tool_chunks,
                chunk_position="last",
            )
        )
        if run_manager is not None:
            run_manager.on_llm_new_token(str(message.content or ""), chunk=chunk)
        yield chunk


def grader_message(
    *,
    result: str = "satisfied",
    criteria: Sequence[dict[str, Any]] | None = None,
    explanation: str = "current state verified",
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "result": result,
                "explanation": explanation,
                "criteria": list(
                    criteria
                    if criteria is not None
                    else [{"name": "state current", "passed": True}]
                ),
            }
        )
    )


def graded_result(
    *,
    result: str = "satisfied",
    criteria: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "structured_response": deepagents_rubric.GraderResponse(
            result=result,
            explanation="graded",
            criteria=list(criteria or []),
        )
    }


class ProbeGraderModel(FixedFakeChatModel):
    """Read the real probe, then grade only the evidence passed to the grader."""

    active_tools: list[str] = Field(default_factory=list, exclude=True)
    grader_observations: list[str] = Field(default_factory=list, exclude=True)
    provider_output: bool = Field(default=False, exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.active_tools = tool_names(tools)
        self.provider_output = "response_format" in kwargs
        self.bindings.append({"tools": list(self.active_tools), "tool_choice": tool_choice})
        return self

    def _generate(self, messages: list[Any], **kwargs: Any) -> ChatResult:  # noqa: ARG002
        if self.active_tools == ["read_file"]:
            results = [message for message in messages if isinstance(message, ToolMessage)]
            if not results:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "/rubric_probe.txt"},
                            "id": "probe-read",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                response = AIMessage(content="VERIFICATION_COMPLETE")
        elif not self.active_tools and self.provider_output:
            observations = [
                str(message.content)
                for message in messages
                if isinstance(message, ToolMessage)
            ]
            observed = "\n".join(observations)
            self.grader_observations.append(observed)
            satisfied = "MIRA_RUBRIC_PROBE_7F3A" in observed
            probe_criterion: dict[str, Any] = {
                "name": "rubric_probe.txt exists and contains exactly MIRA_RUBRIC_PROBE_7F3A",
                "passed": satisfied,
            }
            if not satisfied:
                probe_criterion["gap"] = "Current content is WRONG."
            response = grader_message(
                result="satisfied" if satisfied else "needs_revision",
                explanation="Fresh probe evidence evaluated.",
                criteria=[
                    {
                        "name": "rubric_done.txt exists and contains exactly DONE",
                        "passed": True,
                    },
                    probe_criterion,
                ],
            )
        else:
            raise AssertionError(f"Unexpected tool surface: {self.active_tools!r}")
        return ChatResult(generations=[ChatGeneration(message=response)])


class TranscriptStoryGraderModel(FixedFakeChatModel):
    """Finish verification without tools, then grade the stock transcript."""

    active_tools: list[str] = Field(default_factory=list, exclude=True)
    final_payloads: list[str] = Field(default_factory=list, exclude=True)
    provider_output: bool = Field(default=False, exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.active_tools = tool_names(tools)
        self.provider_output = "response_format" in kwargs
        self.bindings.append({"tools": list(self.active_tools), "tool_choice": tool_choice})
        return self

    def _generate(self, messages: list[Any], **kwargs: Any) -> ChatResult:  # noqa: ARG002
        if self.active_tools == ["inspect_reference"]:
            response = AIMessage(content="VERIFICATION_COMPLETE")
        elif not self.active_tools and self.provider_output:
            payload = "\n".join(
                str(message.content)
                for message in messages
                if isinstance(message, HumanMessage)
            )
            self.final_payloads.append(payload)
            story_is_visible = all(
                theme in payload.lower()
                for theme in ("lighthouse", "storm", "dog")
            )
            response = grader_message(
                result="satisfied" if story_is_visible else "needs_revision",
                explanation="The transcript contains the requested story.",
                criteria=[
                    {
                        "name": "The final response is a story of approximately 200 words.",
                        "passed": story_is_visible,
                    },
                    {
                        "name": "The story includes a lighthouse, a storm, and a dog.",
                        "passed": story_is_visible,
                    },
                ],
            )
        else:
            raise AssertionError(f"Unexpected tool surface: {self.active_tools!r}")
        return ChatResult(generations=[ChatGeneration(message=response)])


class RubricMiddlewareTests(unittest.TestCase):
    def _agent(
        self,
        observed: list[str],
        *,
        require_approval: bool,
    ) -> tuple[Any, FixedFakeChatModel]:
        @tool
        def inspect_external(resource_id: str) -> str:
            """Inspect an external resource."""
            observed.append(resource_id)
            return "current"

        verifier_and_grader_model = FixedFakeChatModel(
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
                    AIMessage(content="VERIFICATION_COMPLETE"),
                    grader_message(),
                ]
            )
        )
        nested_middleware = (
            [
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "inspect_external": {
                            "allowed_decisions": ["approve", "edit", "reject"]
                        }
                    }
                )
            ]
            if require_approval
            else []
        )
        rubric = MiraRubricMiddleware(
            model=verifier_and_grader_model,
            verifier_tools=[inspect_external],
            verifier_middleware=nested_middleware,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = create_deep_agent(
                model=FixedFakeChatModel(messages=iter([AIMessage(content="done")])),
                middleware=[rubric],
                checkpointer=InMemorySaver(),
            )
        return agent, verifier_and_grader_model

    def test_agents_are_separate_static_surfaces_with_one_resolved_model(self) -> None:
        verification_tool = Mock(name="verification_tool")
        hitl = Mock(name="hitl")
        resolved_model = Mock(name="resolved_model")
        verifier = Mock(name="verifier")
        final_grader = Mock(name="final_grader")
        middleware = MiraRubricMiddleware(
            model="provider:model",
            verifier_tools=[verification_tool],
            verifier_middleware=[hitl],
        )

        self.assertEqual(middleware._verifier_tools, [verification_tool])
        self.assertEqual(middleware._verifier_middleware, [hitl])
        self.assertEqual(middleware._tools, [])
        self.assertEqual(middleware._grader_middleware, ())
        with patch.object(
            deepagents_rubric,
            "_strategy_from_model",
            return_value="provider",
        ) as strategy:
            metadata = middleware._grader_trace_metadata()
        strategy.assert_called_once_with("provider:model", has_tools=False)
        self.assertEqual(metadata["rubric_grader_effective_strategy"], "provider")

        with (
            patch("deepagents._models.resolve_model", return_value=resolved_model) as resolve,
            patch(
                "agent.rubric.middleware.create_agent",
                side_effect=[verifier, final_grader],
            ) as create,
        ):
            self.assertIs(middleware._ensure_verifier(), verifier)
            self.assertIs(middleware._ensure_final_grader(), final_grader)
            self.assertIs(middleware._ensure_verifier(), verifier)
            self.assertIs(middleware._ensure_final_grader(), final_grader)

        resolve.assert_called_once_with("provider:model")
        self.assertEqual(create.call_count, 2)
        verifier_kwargs = create.call_args_list[0].kwargs
        self.assertIs(verifier_kwargs["model"], resolved_model)
        self.assertEqual(verifier_kwargs["tools"], [verification_tool])
        self.assertIs(verifier_kwargs["middleware"][0], hitl)
        self.assertEqual(
            type(verifier_kwargs["middleware"][1]).__name__,
            "_VerifierToolObserver",
        )
        self.assertIsNone(verifier_kwargs["response_format"])
        self.assertEqual(verifier_kwargs["system_prompt"], VERIFIER_SYSTEM_PROMPT)
        final_kwargs = create.call_args_list[1].kwargs
        self.assertIs(final_kwargs["model"], resolved_model)
        self.assertEqual(final_kwargs["tools"], [])
        self.assertNotIn("middleware", final_kwargs)
        self.assertIsInstance(final_kwargs["response_format"], ProviderStrategy)
        self.assertIs(final_kwargs["response_format"].schema, deepagents_rubric.GraderResponse)
        self.assertEqual(final_kwargs["system_prompt"], FINAL_GRADER_SYSTEM_PROMPT)

    def test_real_agents_leave_verifier_tool_choice_unforced_and_outer_state_private(self) -> None:
        observed: list[str] = []
        agent, model = self._agent(observed, require_approval=False)
        config = {"configurable": {"thread_id": "rubric-static-agents"}}

        result = agent.invoke(
            {"messages": [HumanMessage(content="finish")], "rubric": "state current"},
            config=config,
        )

        self.assertEqual(observed, ["page-123"])
        self.assertEqual([message.content for message in result["messages"]], ["finish", "done"])
        self.assertEqual(
            model.bindings,
            [
                {"tools": ["inspect_external"], "tool_choice": None},
                {"tools": ["inspect_external"], "tool_choice": None},
                {"tools": [], "tool_choice": None},
            ],
        )
        self.assertEqual(agent.get_state(config).values["_rubric_status"], "satisfied")

    def test_real_nested_agent_projects_live_custom_tool_events(self) -> None:
        observed: list[str] = []
        agent, _model = self._agent(observed, require_approval=False)

        async def collect() -> list[dict[str, Any]]:
            events: list[dict[str, Any]] = []
            async for event in agent.astream(
                {
                    "messages": [HumanMessage(content="finish")],
                    "rubric": "state current",
                },
                config={"configurable": {"thread_id": "rubric-live-custom"}},
                stream_mode="custom",
            ):
                if isinstance(event, dict):
                    events.append(event)
            return events

        events = asyncio.run(collect())
        types = [event.get("type") for event in events]

        self.assertLess(types.index("rubric_verification_start"), types.index("rubric_tool_start"))
        self.assertIn("rubric_tool_call_delta", types)
        self.assertLess(types.index("rubric_tool_start"), types.index("rubric_tool_end"))
        self.assertLess(types.index("rubric_tool_end"), types.index("rubric_verification_end"))
        self.assertLess(types.index("rubric_verification_end"), types.index("rubric_grading_start"))
        start = next(event for event in events if event.get("type") == "rubric_tool_start")
        end = next(event for event in events if event.get("type") == "rubric_tool_end")
        self.assertEqual(start["tool_call_id"], "inspect-call")
        self.assertEqual(start["tool_args"], {"resource_id": "page-123"})
        self.assertEqual(end["tool_call_id"], "inspect-call")
        self.assertEqual(end["output"], "current")
        self.assertFalse(end["is_error"])

    def test_runner_contains_verifier_tools_and_internal_graphs_in_rubric_ui(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.root_tools: list[str] = []
                self.subagents: list[str] = []
                self.rubric_events: list[dict[str, Any]] = []

            def waiting_started(self) -> None:
                return None

            def waiting_finished(self) -> None:
                return None

            def finish_main(self) -> None:
                return None

            def text_delta(self, text: str) -> None:  # noqa: ARG002
                return None

            def reasoning_delta(self, text: str) -> None:  # noqa: ARG002
                return None

            def tool_call(self, name: str, args: Any, call_id: str = "") -> None:  # noqa: ARG002
                self.root_tools.append(name)

            def rubric_evaluation_started(self, *args: Any, **kwargs: Any) -> None:
                return None

            def rubric_lifecycle_event(self, event: dict[str, Any]) -> None:
                self.rubric_events.append(dict(event))

            def rubric_evaluation_finished(self, *args: Any, **kwargs: Any) -> None:
                return None

            def subagent_label(self, subagent: Any) -> str:
                return str(getattr(subagent, "graph_name", "subagent"))

            def subagent_started(self, subagent: str, *args: Any, **kwargs: Any) -> None:
                self.subagents.append(subagent)

            def subagent_finished(self, *args: Any, **kwargs: Any) -> None:
                return None

        observed: list[str] = []
        agent, _model = self._agent(observed, require_approval=False)
        renderer = Renderer()

        result = asyncio.run(
            run_turn(
                agent,
                "finish",
                renderer,
                "rubric-runner-containment",
                rubric="state current",
                include_rubric_state=True,
            )
        )

        self.assertEqual(result.rubric_status, "satisfied")
        self.assertEqual(renderer.root_tools, [])
        self.assertEqual(renderer.subagents, [])
        lifecycle_types = [event.get("type") for event in renderer.rubric_events]
        self.assertEqual(lifecycle_types.count("rubric_tool_start"), 1)
        self.assertEqual(lifecycle_types.count("rubric_tool_end"), 1)

    def test_verifier_uses_existing_hitl_and_resumes_without_grader_error(self) -> None:
        observed: list[str] = []
        agent, _model = self._agent(observed, require_approval=True)
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

        agent.invoke(
            Command(resume={interrupt.id: {"decisions": [{"type": "approve"}]}}),
            config=config,
        )

        self.assertEqual(observed, ["page-123"])
        self.assertEqual(agent.get_state(config).values["_rubric_status"], "satisfied")

    def test_graph_bubble_up_behavior_is_inherited_sync_and_async(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        interrupt = GraphInterrupt(())
        state = {
            "rubric": "state current",
            "messages": [],
            "_current_grading_run_id": "run",
            "_rubric_iterations": 0,
        }
        runtime = SimpleNamespace(context=None, stream_writer=lambda _event: None)
        with patch.object(middleware, "_grade", side_effect=interrupt), self.assertRaises(
            GraphInterrupt
        ):
            middleware.after_agent(state, runtime)

        async def invoke() -> None:
            with patch.object(
                middleware,
                "_agrade",
                new=AsyncMock(side_effect=interrupt),
            ), self.assertRaises(GraphInterrupt):
                await middleware.aafter_agent(state, runtime)

        asyncio.run(invoke())

    def test_only_per_call_seams_replace_the_stock_lifecycle(self) -> None:
        parent = deepagents_rubric.RubricMiddleware
        for name in (
            "_grade",
            "_agrade",
            "_grader_input",
            "_build_grader_payload",
            "_extract_graded",
            "after_agent",
            "aafter_agent",
            "_handle_grader_exception",
        ):
            self.assertIs(getattr(MiraRubricMiddleware, name), getattr(parent, name))
        self.assertIsNot(MiraRubricMiddleware._invoke_grader, parent._invoke_grader)
        self.assertIsNot(MiraRubricMiddleware._ainvoke_grader, parent._ainvoke_grader)

        source = inspect.getsource(MiraRubricMiddleware)
        for abandoned_name in (
            "_GraderPhaseMiddleware",
            "_rubric_verification_complete",
            "jump_to",
            "task(",
        ):
            self.assertNotIn(abandoned_name, source)

    def test_evidence_keeps_complete_tool_pairs_and_discards_all_verifier_prose(self) -> None:
        large_result = "x" * 100_000
        success = ToolMessage(
            content=large_result,
            tool_call_id="read-1",
            name="read_file",
        )
        failure = ToolMessage(
            content="permission denied",
            tool_call_id="read-2",
            name="read_file",
            status="error",
        )
        result = {
            "messages": [
                HumanMessage(content="stock payload"),
                AIMessage(content="private verifier reasoning"),
                AIMessage(
                    content="commentary to discard",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "/one"},
                            "id": "read-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "read_file",
                            "args": {"file_path": "/two"},
                            "id": "read-2",
                            "type": "tool_call",
                        },
                        {
                            "name": "read_file",
                            "args": {"file_path": "/unfinished"},
                            "id": "read-3",
                            "type": "tool_call",
                        },
                    ],
                ),
                success,
                failure,
                AIMessage(content="VERIFICATION_COMPLETE"),
            ]
        }

        evidence = MiraRubricMiddleware._verification_evidence(result)

        self.assertEqual([type(message) for message in evidence], [AIMessage, ToolMessage, ToolMessage])
        tool_call_message = evidence[0]
        self.assertEqual(tool_call_message.content, "")
        self.assertEqual(
            [(call["name"], call["args"], call["id"]) for call in tool_call_message.tool_calls],
            [
                ("read_file", {"file_path": "/one"}, "read-1"),
                ("read_file", {"file_path": "/two"}, "read-2"),
            ],
        )
        self.assertIs(evidence[1], success)
        self.assertEqual(evidence[1].content, large_result)
        self.assertIs(evidence[2], failure)
        self.assertEqual(evidence[2].status, "error")
        self.assertNotIn("VERIFICATION_COMPLETE", repr(evidence))
        self.assertNotIn("reasoning", repr(evidence))

    def test_verifier_input_reuses_bounded_transcript_without_grader_instructions(
        self,
    ) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        messages = [HumanMessage(content="original user request")]
        messages.extend(AIMessage(content=f"assistant-{index}") for index in range(35))
        state = {
            "rubric": "The delivered report contains a conclusion.",
            "_rubric_criteria": [
                "The report exists.",
                "The report contains a conclusion.",
            ],
            "messages": messages,
        }
        correction = "A previous attempt returned only 1 of the 2 criteria in the rubric."

        with patch("agent.rubric.middleware.secrets.token_hex", return_value="fixed"):
            verifier_input = middleware._verifier_input(state, 3)

        payload = verifier_input["messages"][0].content
        self.assertIn(
            "<rubric-fixed>\nThe delivered report contains a conclusion.\n</rubric-fixed>",
            payload,
        )
        self.assertIn(
            "<criteria-fixed>\n1. The report exists.\n"
            "2. The report contains a conclusion.\n</criteria-fixed>",
            payload,
        )
        self.assertIn("[user] original user request", payload)
        self.assertIn("[assistant] assistant-34", payload)
        self.assertNotIn("[assistant] assistant-0", payload)
        self.assertIn("The transcript is valid historical evidence", payload)
        self.assertIn("return only VERIFICATION_COMPLETE", payload)
        for grader_instruction in (
            "Evaluate whether the agent transcript below satisfies",
            "Return a GraderResponse",
            "Break the rubric into its individual criteria and return one entry",
            correction,
            "satisfied",
            "needs_revision",
            "failed",
        ):
            self.assertNotIn(grader_instruction, payload)

    def test_sync_call_uses_stock_inputs_and_separate_evidence_channel(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        tool_call = AIMessage(
            content="verifier prose",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/result"},
                    "id": "read-1",
                    "type": "tool_call",
                }
            ],
        )
        tool_result = ToolMessage(content="RAW", tool_call_id="read-1", name="read_file")
        verifier = Mock()
        verifier.invoke.return_value = {
            "messages": [HumanMessage(content="input"), tool_call, tool_result, AIMessage(content="VERIFICATION_COMPLETE")]
        }
        final_grader = Mock()
        final_grader.invoke.return_value = graded_result(
            criteria=[{"name": "criterion", "passed": True}]
        )
        state = {
            "rubric": "criterion",
            "_rubric_criteria": ["criterion"],
            "messages": [HumanMessage(content="original"), AIMessage(content="done")],
        }
        correction = "A previous attempt returned only 0 of the 1 criteria in the rubric."
        context = object()

        with (
            patch.object(middleware, "_ensure_verifier", return_value=verifier),
            patch.object(middleware, "_ensure_final_grader", return_value=final_grader),
            patch.object(
                middleware,
                "_grader_input",
                wraps=middleware._grader_input,
            ) as stock_input,
            patch.object(middleware, "_extract_graded", wraps=middleware._extract_graded) as extract,
            patch("agent.rubric.middleware.secrets.token_hex", return_value="fixed"),
        ):
            graded = middleware._invoke_grader(
                state,
                2,
                correction,
                context=context,
            )

        self.assertEqual(graded.result, "satisfied")
        extract.assert_called_once_with(final_grader.invoke.return_value)
        stock_input.assert_called_once_with(state, 2, correction)
        verifier_input = verifier.invoke.call_args.args[0]
        verifier_payload = verifier_input["messages"][0].content
        self.assertIn("<rubric-fixed>\ncriterion", verifier_payload)
        self.assertIn("<criteria-fixed>\n1. criterion", verifier_payload)
        self.assertIn("[user] original", verifier_payload)
        self.assertIn("[assistant] done", verifier_payload)
        self.assertNotIn(correction, verifier_payload)
        self.assertNotIn(
            "Evaluate whether the agent transcript below satisfies",
            verifier_payload,
        )
        self.assertNotIn("Return a GraderResponse", verifier_payload)
        final_input = final_grader.invoke.call_args.args[0]
        self.assertIn(
            "Evaluate whether the agent transcript below satisfies",
            final_input["messages"][0].content,
        )
        self.assertIn("Return a GraderResponse", final_input["messages"][0].content)
        self.assertIn(correction, final_input["messages"][0].content)
        self.assertEqual(final_input["messages"][1].content, VERIFICATION_EVIDENCE_MESSAGE)
        self.assertEqual(final_input["messages"][2].content, "")
        self.assertEqual(final_input["messages"][2].tool_calls, tool_call.tool_calls)
        self.assertIs(final_input["messages"][3], tool_result)
        self.assertEqual(state["messages"], [HumanMessage(content="original"), AIMessage(content="done")])
        verifier.invoke.assert_called_once()
        final_grader.invoke.assert_called_once()
        self.assertIs(verifier.invoke.call_args.kwargs["context"], context)
        self.assertIs(final_grader.invoke.call_args.kwargs["context"], context)

    def test_phase_events_wrap_the_isolated_verifier_before_the_final_grader(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        verifier = Mock()
        verifier.invoke.return_value = {"messages": [AIMessage(content="VERIFICATION_COMPLETE")]}
        grader = Mock()
        grader.invoke.return_value = graded_result()
        events: list[dict[str, Any]] = []
        state = {
            "rubric": "criterion",
            "messages": [],
            "_current_grading_run_id": "grade-phases",
            "_rubric_iterations": 0,
        }
        middleware._prepare_evaluation(
            state,
            SimpleNamespace(stream_writer=events.append),
        )

        with (
            patch.object(middleware, "_ensure_verifier", return_value=verifier),
            patch.object(middleware, "_ensure_final_grader", return_value=grader),
        ):
            middleware._invoke_grader(state, 0)

        self.assertEqual(
            [event["type"] for event in events],
            [
                "rubric_evaluation_start",
                "rubric_verification_start",
                "rubric_verification_end",
                "rubric_grading_start",
                "rubric_grading_end",
            ],
        )
        self.assertTrue(events[2]["succeeded"])
        self.assertTrue(events[4]["succeeded"])

    def test_verifier_observer_preserves_ids_raw_results_and_tool_errors(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        events: list[dict[str, Any]] = []
        middleware._prepare_evaluation(
            {
                "rubric": "criterion",
                "messages": [],
                "_current_grading_run_id": "grade-tools",
                "_rubric_iterations": 0,
            },
            SimpleNamespace(stream_writer=events.append),
        )
        observer = mira_rubric._VerifierToolObserver()
        raw = "line one\n" + "x" * 10_000
        request = SimpleNamespace(
            tool_call={
                "id": "read-17",
                "name": "read_file",
                "args": {"file_path": "/full"},
            }
        )

        returned = observer.wrap_tool_call(
            request,
            lambda _request: ToolMessage(
                content=raw,
                tool_call_id="read-17",
                name="read_file",
                status="error",
            ),
        )

        self.assertIsInstance(returned, ToolMessage)
        start, end = events[-2:]
        self.assertEqual(start["tool_call_id"], "read-17")
        self.assertEqual(start["tool_args"], {"file_path": "/full"})
        self.assertEqual(end["tool_call_id"], "read-17")
        self.assertEqual(end["output"], raw)
        self.assertTrue(end["is_error"])

    def test_verifier_failure_emits_failed_phase_without_starting_grader(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        verifier = Mock()
        verifier.invoke.side_effect = RuntimeError("verifier broke")
        events: list[dict[str, Any]] = []
        state = {
            "rubric": "criterion",
            "messages": [],
            "_current_grading_run_id": "grade-verifier-error",
            "_rubric_iterations": 0,
        }
        middleware._prepare_evaluation(state, SimpleNamespace(stream_writer=events.append))

        with (
            patch.object(middleware, "_ensure_verifier", return_value=verifier),
            self.assertRaisesRegex(RuntimeError, "verifier broke"),
        ):
            middleware._invoke_grader(state, 0)

        types = [event["type"] for event in events]
        self.assertEqual(types[-2:], ["rubric_verification_start", "rubric_verification_end"])
        self.assertFalse(events[-1]["succeeded"])
        self.assertNotIn("rubric_grading_start", types)

    def test_no_verifier_tool_call_still_grades_transcript_only_story(self) -> None:
        @tool
        def inspect_reference(reference: str) -> str:
            """Inspect an external reference when one is relevant."""
            raise AssertionError(f"Transcript-only rubric should not inspect {reference}")

        story = " ".join(
            [
                "At the lighthouse, a dog watched the storm gather beyond the harbor.",
                *[
                    "Wind pressed salt against the windows while the keeper tended the lamp."
                    for _ in range(15)
                ],
                "By dawn, the dog barked at calm water and guided the tired keeper home.",
            ]
        )
        model = TranscriptStoryGraderModel(messages=iter(()))
        middleware = MiraRubricMiddleware(model=model, verifier_tools=[inspect_reference])
        rubric = (
            "- The final response is a story of approximately 200 words.\n"
            "- The story includes a lighthouse, a storm, and a dog."
        )

        graded = middleware._invoke_grader(
            {
                "rubric": rubric,
                "messages": [
                    HumanMessage(content="Write a 200-word story with three requested themes."),
                    AIMessage(content=story),
                ],
            },
            0,
        )

        self.assertEqual(graded.result, "satisfied")
        self.assertTrue(all(criterion["passed"] for criterion in graded.criteria))
        self.assertEqual(
            model.bindings,
            [
                {"tools": ["inspect_reference"], "tool_choice": None},
                {"tools": [], "tool_choice": None},
            ],
        )
        self.assertEqual(len(model.final_payloads), 1)
        self.assertIn(story, model.final_payloads[0])
        self.assertIn(
            "Evaluate whether the agent transcript below satisfies",
            model.final_payloads[0],
        )

    def test_async_call_matches_sync_evidence_and_context_behavior(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "inspect",
                    "args": {"key": "value"},
                    "id": "inspect-1",
                    "type": "tool_call",
                }
            ],
        )
        result = ToolMessage(content="current", tool_call_id="inspect-1", status="error")
        verifier = Mock()
        verifier.ainvoke = AsyncMock(return_value={"messages": [call, result]})
        grader = Mock()
        grader.ainvoke = AsyncMock(
            return_value=graded_result(criteria=[{"name": "criterion", "passed": True}])
        )
        context = object()

        async def invoke() -> deepagents_rubric.GraderResponse:
            with (
                patch.object(middleware, "_ensure_verifier", return_value=verifier),
                patch.object(middleware, "_ensure_final_grader", return_value=grader),
            ):
                return await middleware._ainvoke_grader(
                    {"rubric": "criterion", "messages": []},
                    0,
                    context=context,
                )

        graded = asyncio.run(invoke())

        self.assertEqual(graded.result, "satisfied")
        verifier_payload = verifier.ainvoke.await_args.args[0]["messages"][0].content
        self.assertIn("Gather useful current-state evidence", verifier_payload)
        self.assertNotIn(
            "Evaluate whether the agent transcript below satisfies",
            verifier_payload,
        )
        self.assertNotIn("Return a GraderResponse", verifier_payload)
        grader_input = grader.ainvoke.await_args.args[0]
        self.assertIn(
            "Evaluate whether the agent transcript below satisfies",
            grader_input["messages"][0].content,
        )
        self.assertIn("Return a GraderResponse", grader_input["messages"][0].content)
        self.assertEqual(grader_input["messages"][-1].status, "error")
        self.assertIs(verifier.ainvoke.await_args.kwargs["context"], context)
        self.assertIs(grader.ainvoke.await_args.kwargs["context"], context)

    def test_stock_coverage_retry_reruns_both_agents_and_corrects_only_final_grader(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        verifier = Mock()
        verifier.invoke.side_effect = [
            {"messages": [AIMessage(content="VERIFICATION_COMPLETE")]},
            {"messages": [AIMessage(content="VERIFICATION_COMPLETE")]},
        ]
        final_grader = Mock()
        final_grader.invoke.side_effect = [
            graded_result(criteria=[]),
            graded_result(criteria=[{"name": "criterion", "passed": True}]),
        ]
        state = {"rubric": "criterion", "messages": [], "_rubric_criteria": []}

        with (
            patch.object(middleware, "_ensure_verifier", return_value=verifier),
            patch.object(middleware, "_ensure_final_grader", return_value=final_grader),
        ):
            graded = middleware._grade(state, 0)

        self.assertEqual(graded.criteria[0]["name"], "criterion")
        self.assertEqual(verifier.invoke.call_count, 2)
        self.assertEqual(final_grader.invoke.call_count, 2)
        correction = "A previous attempt returned no per-criterion verdicts at all."
        self.assertNotIn(correction, verifier.invoke.call_args_list[1].args[0]["messages"][0].content)
        self.assertIn(correction, final_grader.invoke.call_args_list[1].args[0]["messages"][0].content)

    def test_stock_async_coverage_retry_reruns_both_agents(self) -> None:
        middleware = MiraRubricMiddleware(model="fake-model")
        verifier = Mock()
        verifier.ainvoke = AsyncMock(
            side_effect=[
                {"messages": [AIMessage(content="VERIFICATION_COMPLETE")]},
                {"messages": [AIMessage(content="VERIFICATION_COMPLETE")]},
            ]
        )
        grader = Mock()
        grader.ainvoke = AsyncMock(
            side_effect=[
                graded_result(criteria=[]),
                graded_result(criteria=[{"name": "criterion", "passed": True}]),
            ]
        )

        async def invoke() -> deepagents_rubric.GraderResponse:
            with (
                patch.object(middleware, "_ensure_verifier", return_value=verifier),
                patch.object(middleware, "_ensure_final_grader", return_value=grader),
            ):
                return await middleware._agrade(
                    {"rubric": "criterion", "messages": [], "_rubric_criteria": []},
                    0,
                )

        graded = asyncio.run(invoke())
        self.assertEqual(graded.criteria[0]["name"], "criterion")
        self.assertEqual(verifier.ainvoke.await_count, 2)
        self.assertEqual(grader.ainvoke.await_count, 2)

    def test_golden_probe_positive_wrong_and_false_transcript_conflict(self) -> None:
        rubric = (
            "- rubric_done.txt exists and contains exactly DONE\n"
            "- rubric_probe.txt exists and contains exactly MIRA_RUBRIC_PROBE_7F3A"
        )
        cases = (
            ("MIRA_RUBRIC_PROBE_7F3A", "No claim about the probe.", "satisfied"),
            ("WRONG", "No claim about the probe.", "needs_revision"),
            ("WRONG", "rubric_probe.txt contains MIRA_RUBRIC_PROBE_7F3A.", "needs_revision"),
        )
        for index, (probe_content, transcript_claim, expected) in enumerate(cases):
            with self.subTest(probe_content=probe_content, claim=transcript_claim), tempfile.TemporaryDirectory() as root:
                Path(root, "rubric_done.txt").write_text("DONE", encoding="utf-8")
                Path(root, "rubric_probe.txt").write_text(probe_content, encoding="utf-8")
                backend = FilesystemBackend(root_dir=root, virtual_mode=True)
                read_file = FilesystemMiddleware(backend=backend, tools=["read_file"]).tools[0]
                model = ProbeGraderModel(messages=iter(()))
                middleware = MiraRubricMiddleware(model=model, verifier_tools=[read_file])

                graded = middleware._invoke_grader(
                    {
                        "rubric": rubric,
                        "messages": [
                            HumanMessage(
                                content="Create only rubric_done.txt; do not inspect rubric_probe.txt."
                            ),
                            AIMessage(content=f"Created rubric_done.txt. {transcript_claim}"),
                        ],
                    },
                    index,
                )

                self.assertEqual(graded.result, expected)
                self.assertEqual(
                    model.bindings,
                    [
                        {"tools": ["read_file"], "tool_choice": None},
                        {"tools": ["read_file"], "tool_choice": None},
                        {"tools": [], "tool_choice": None},
                    ],
                )
                self.assertEqual(len(model.grader_observations), 1)
                self.assertIn(probe_content, model.grader_observations[0])

    def test_prompts_define_verification_only_and_two_evidence_channel_roles(self) -> None:
        self.assertIn("Do not return GraderResponse", VERIFIER_SYSTEM_PROMPT)
        self.assertIn("VERIFICATION_COMPLETE", VERIFIER_SYSTEM_PROMPT)
        self.assertIn("Do not assign satisfied or needs_revision", VERIFIER_SYSTEM_PROMPT)
        self.assertIn(deepagents_rubric.GRADER_SYSTEM_PROMPT, FINAL_GRADER_SYSTEM_PROMPT)
        self.assertIn("either channel may be sufficient", FINAL_GRADER_SYSTEM_PROMPT)
        self.assertIn("not a mandatory proof gate", FINAL_GRADER_SYSTEM_PROMPT)
        self.assertIn("current-state observation", FINAL_GRADER_SYSTEM_PROMPT)
        self.assertIn("untrusted observations", FINAL_GRADER_SYSTEM_PROMPT)
