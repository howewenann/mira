"""Public native-LangGraph workflow API tests."""

from __future__ import annotations

import inspect
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, NotRequired, TypedDict
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, PrivateAttr

from config.metadata import ModelMetadata
from config.settings import normalize_settings
from core.application.app import MiraApplication
from core.interface import NullFrontend
from mira import INHERIT, MiraContext, MiraWorkflowAPI


class WorkflowModel(BaseChatModel):
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)
    bound_history: ClassVar[list[list[str]]] = []

    @property
    def _llm_type(self) -> str:
        return "mira-workflow-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "WorkflowModel":
        del kwargs
        clone = self.model_copy(deep=True)
        clone._bound_tool_names = [
            str(
                getattr(item, "name", None)
                or getattr(item, "__name__", None)
                or (item.get("name") if isinstance(item, dict) else "")
                or (
                    item.get("function", {}).get("name", "")
                    if isinstance(item, dict)
                    else ""
                )
            )
            for item in tools
        ]
        type(self).bound_history.append(list(clone._bound_tool_names))
        return clone

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        text = next(
            (
                str(message.content)
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "done",
        )
        if "Findings" in self._bound_tool_names:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Findings",
                        "args": {"summary": f"structured:{text}", "confidence": 0.9},
                        "id": "findings-call",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(f"answer:{text}")
        return ChatResult(generations=[ChatGeneration(message=message)])


class Findings(BaseModel):
    summary: str
    confidence: float


class AgentState(TypedDict):
    messages: list[Any]
    structured_response: NotRequired[Findings]


class WorkflowAPITests(unittest.IsolatedAsyncioTestCase):
    async def start_application(
        self,
        workspace: Path,
        settings: dict[str, Any] | None = None,
    ) -> MiraApplication:
        WorkflowModel.bound_history = []
        config = {
            "session_dir": str(workspace / ".mira" / "_sessions"),
            "settings": normalize_settings(settings or {}),
            "settings_valid": True,
        }
        metadata = ModelMetadata(context_tokens=32768, context_source="test")
        with (
            patch("agent.llm.active_model_issues", return_value=[]),
            patch("agent.llm.get_llm", return_value=WorkflowModel()),
            patch("agent.factory.get_llm", return_value=WorkflowModel()),
            patch("agent.llm.get_model_name", return_value="workflow-test"),
            patch(
                "config.metadata.infer_model_metadata",
                new=AsyncMock(return_value=metadata),
            ),
        ):
            return await MiraApplication.start(workspace=workspace, config=config)

    async def test_application_exposes_facade_and_native_graph_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            application = await self.start_application(Path(directory))
            try:
                self.assertIsInstance(application.frontend, NullFrontend)
                self.assertIsInstance(application.workflows, MiraWorkflowAPI)
                context = application.workflows.context
                self.assertIsInstance(context, MiraContext)
                self.assertIn("write_file", context.tools)
                self.assertIn("general-purpose", context.agents)

                graph = StateGraph(AgentState, context_schema=MiraContext)
                graph.add_node("worker", application.workflows.agent())
                graph.add_edge(START, "worker")
                graph.add_edge("worker", END)
                compiled = graph.compile()
                with warnings.catch_warnings(record=True) as caught:
                    result = await compiled.ainvoke(
                        {"messages": [HumanMessage("topic")]},
                        context=context,
                    )
                self.assertEqual(result["messages"][-1].content, "answer:topic")
                self.assertFalse(
                    any("Pydantic serializer warnings" in str(item.message) for item in caught)
                )
                sync_result = compiled.invoke(
                    {"messages": [HumanMessage("sync topic")]},
                    context=application.workflows.context,
                )
                self.assertEqual(
                    sync_result["messages"][-1].content,
                    "answer:sync topic",
                )

                structured = application.workflows.agent(response_format=Findings)
                structured_graph = StateGraph(AgentState, context_schema=MiraContext)
                structured_graph.add_node("worker", structured)
                structured_graph.add_edge(START, "worker")
                structured_graph.add_edge("worker", END)
                structured_result = await structured_graph.compile().ainvoke(
                    {"messages": [HumanMessage("topic")]},
                    context=application.workflows.context,
                )
                self.assertTrue(WorkflowModel.bound_history, structured_result)
                self.assertIn(
                    "structured_response",
                    structured_result,
                    WorkflowModel.bound_history,
                )
                self.assertEqual(
                    structured_result["structured_response"].summary,
                    "structured:topic",
                )
            finally:
                await application.shutdown()

    async def test_specializations_are_independent_and_validate_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            subagent_dir = workspace / ".mira" / "subagents"
            subagent_dir.mkdir(parents=True)
            (subagent_dir / "researcher.py").write_text(
                "SUBAGENTS = [{\n"
                "    'name': 'researcher',\n"
                "    'description': 'Configured researcher',\n"
                "    'system_prompt': 'Research carefully.',\n"
                "}]\n",
                encoding="utf-8",
            )
            application = await self.start_application(
                workspace,
                {
                    "models": {
                        "subagents": {
                            "researcher": {"enabled": True, "model": None}
                        }
                    }
                },
            )
            try:
                mira = application.workflows
                first = mira.agent(name="scanner", tools=["ls"])
                second = mira.agent(tools=[])
                inherited = mira.agent(system_prompt=INHERIT, response_format=INHERIT)
                configured_researcher = mira.agent("researcher")

                self.assertIsNot(first, second)
                self.assertIsNot(first, inherited)
                researcher_graph = StateGraph(AgentState, context_schema=MiraContext)
                researcher_graph.add_node("researcher", configured_researcher)
                researcher_graph.add_edge(START, "researcher")
                researcher_graph.add_edge("researcher", END)
                researcher_result = await researcher_graph.compile().ainvoke(
                    {"messages": [HumanMessage("configured topic")]},
                    context=mira.context,
                )
                self.assertEqual(
                    researcher_result["messages"][-1].content,
                    "answer:configured topic",
                )
                self.assertEqual(
                    inspect.signature(mira.agent).parameters["base"].default,
                    "general-purpose",
                )
                self.assertNotIn("model", inspect.signature(mira.agent).parameters)
                with self.assertRaisesRegex(KeyError, "Unknown MIRA subagent"):
                    mira.agent("missing")
                with self.assertRaisesRegex(ValueError, "Unavailable references"):
                    mira.agent(tools=["missing-tool"])

                registry = application.agent.mira_workflow_agents
                base_spec = {
                    "name": "researcher",
                    "description": "Researcher",
                    "system_prompt": "base prompt",
                    "tools": [],
                    "response_format": {"type": "object"},
                }
                application.agent.mira_workflow_agents = replace(
                    registry,
                    specs=(base_spec,),
                )
                with patch(
                    "agent.workflows.agents.compile_raw_subagent",
                    side_effect=lambda spec, **_kwargs: {"runnable": spec},
                ):
                    inherited_spec = mira.agent("researcher")
                    cleared_spec = mira.agent(
                        "researcher",
                        name="local-researcher",
                        tools=[],
                        system_prompt=None,
                        response_format=None,
                    )
                self.assertEqual(inherited_spec["system_prompt"], "base prompt")
                self.assertIn("response_format", inherited_spec)
                self.assertEqual(cleared_spec["name"], "local-researcher")
                self.assertIsNone(cleared_spec["system_prompt"])
                self.assertNotIn("response_format", cleared_spec)
                self.assertNotEqual(inherited_spec, cleared_spec)
                self.assertEqual(base_spec["name"], "researcher")
                self.assertEqual(base_spec["system_prompt"], "base prompt")

                opaque = replace(
                    registry,
                    specs=(
                        {
                            "name": "opaque",
                            "description": "opaque",
                            "runnable": inherited,
                        },
                    ),
                )
                application.agent.mira_workflow_agents = opaque
                self.assertIs(mira.agent("opaque"), inherited)
                with self.assertRaisesRegex(TypeError, "already compiled"):
                    mira.agent("opaque", tools=[])

                application.agent.mira_workflow_agents = replace(
                    registry,
                    specs=(
                        {
                            "name": "remote",
                            "description": "remote",
                            "graph_id": "deployment/graph",
                        },
                    ),
                )
                with self.assertRaisesRegex(TypeError, "remote/opaque"):
                    mira.agent("remote")
            finally:
                await application.shutdown()


if __name__ == "__main__":
    unittest.main()
