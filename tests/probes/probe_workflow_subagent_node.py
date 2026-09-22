"""Probe workflow-local specialization of MIRA/DeepAgents subagents.

Questions answered
------------------
1. Can one reusable base subagent be specialized into independent workflow
   variants with different tool allowlists without mutating the base spec?
2. Can `response_format` be overridden for one workflow-local variant?
3. Can the resulting DeepAgents runnable be inserted directly into a LangGraph
   `StateGraph`?
4. If the workflow uses ordinary domain state rather than AgentState/messages,
   what is the smallest adapter required?

This deliberately does NOT re-test MIRA's existing tool-name resolution. The
probe resolves "ls"/"glob" to exact BaseTool objects locally, then tests the
DeepAgents/LangGraph boundary after resolution.

Run from repository root:
    python tests/probes/probe_workflow_subagent_node.py

Run from tests/probes:
    python probe_workflow_subagent_node.py
"""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepagents.middleware.subagents import create_sub_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, PrivateAttr


# ---------------------------------------------------------------------------
# Probe tools
# ---------------------------------------------------------------------------


@tool
def ls(path: str = "/") -> str:
    """List a directory for the probe."""
    return f"LS:{path}"


@tool
def glob(pattern: str) -> str:
    """Match paths for the probe."""
    return f"GLOB:{pattern}"


TOOL_REGISTRY: dict[str, BaseTool] = {
    "ls": ls,
    "glob": glob,
}


# ---------------------------------------------------------------------------
# Probe response schema + deterministic model
# ---------------------------------------------------------------------------


class Findings(BaseModel):
    summary: str
    confidence: float


class ProbeChatModel(BaseChatModel):
    """Deterministic model supporting both text and ToolStrategy output.

    If LangChain binds a structured-output tool named `Findings`, the model
    calls it. Otherwise it returns ordinary text.

    The model also requires a HumanMessage so a direct insertion into a
    domain-only workflow cannot accidentally appear to work.
    """

    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "workflow-subagent-probe"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "ProbeChatModel":
        del tool_choice, kwargs
        self._bound_tool_names = [_tool_name(item) for item in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        human = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        if human is None:
            raise RuntimeError(
                "PROBE_EXPECTED_HUMAN_MESSAGE: direct domain-state node supplied "
                "no AgentState/messages input."
            )

        if "Findings" in self._bound_tool_names:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Findings",
                        "args": {
                            "summary": f"structured:{human.content}",
                            "confidence": 0.91,
                        },
                        "id": "findings-call",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content=f"text:{human.content}")

        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_name(item: Any) -> str:
    if isinstance(item, BaseTool):
        return item.name
    if isinstance(item, dict):
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        if item.get("name"):
            return str(item["name"])
    name = getattr(item, "__name__", None)
    return str(name) if name else type(item).__name__


# ---------------------------------------------------------------------------
# Probe-only specialization helper
# ---------------------------------------------------------------------------


_UNSET = object()


def specialize(
    base: dict[str, Any],
    *,
    tools: list[str | BaseTool] | object = _UNSET,
    response_format: Any = _UNSET,
) -> dict[str, Any]:
    """Mimic the intended `mira.agent(... overrides ...)` specialization."""

    spec = copy.copy(base)

    if tools is not _UNSET:
        resolved: list[BaseTool] = []
        for entry in tools:
            if isinstance(entry, str):
                resolved.append(TOOL_REGISTRY[entry])
            else:
                resolved.append(entry)
        spec["tools"] = resolved

    if response_format is not _UNSET:
        spec["response_format"] = response_format

    # Each workflow-local specialization gets its own deterministic model
    # instance so bound tool state cannot bleed between variants.
    spec["model"] = ProbeChatModel()
    return spec


def execution_tool_names(agent: Any) -> list[str]:
    try:
        node = agent.nodes["tools"].bound
    except (AttributeError, KeyError):
        return []
    if not isinstance(node, ToolNode):
        return []
    return sorted(node.tools_by_name)


# ---------------------------------------------------------------------------
# Workflow states
# ---------------------------------------------------------------------------


class AgentLikeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    structured_response: NotRequired[Findings]


class DomainState(TypedDict):
    topic: str
    findings: NotRequired[Findings]
    text: NotRequired[str]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(label: str, passed: bool, detail: Any = None) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")
    if detail is not None:
        print(detail)
    return passed


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


async def main() -> int:
    print("=" * 88)
    print("MIRA workflow subagent-node probe")
    print("=" * 88)
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Probe cwd:       {Path.cwd()}")

    failures = 0

    base_model = ProbeChatModel()
    base_spec: dict[str, Any] = {
        "name": "researcher",
        "description": "Probe researcher",
        "system_prompt": "Return the requested result.",
        "model": base_model,
        "tools": [ls, glob],
    }

    original_tools = list(base_spec["tools"])
    original_keys = set(base_spec)

    scanner_ls_spec = specialize(base_spec, tools=["ls"])
    scanner_ls_glob_spec = specialize(base_spec, tools=["ls", "glob"])
    structured_spec = specialize(
        base_spec,
        tools=["ls"],
        response_format=Findings,
    )
    text_spec = specialize(base_spec, tools=["ls"])

    failures += not report(
        "Workflow-local specialization does not mutate the reusable base spec",
        base_spec["tools"] == original_tools and set(base_spec) == original_keys,
        {
            "base_tools": [tool.name for tool in base_spec["tools"]],
            "base_has_response_format": "response_format" in base_spec,
        },
    )

    scanner_ls = create_sub_agent(scanner_ls_spec)
    scanner_ls_glob = create_sub_agent(scanner_ls_glob_spec)
    structured = create_sub_agent(structured_spec)
    text_agent = create_sub_agent(text_spec)

    failures += not report(
        "Two variants from the same base compile with independent exact tool surfaces",
        execution_tool_names(scanner_ls) == ["ls"]
        and execution_tool_names(scanner_ls_glob) == ["glob", "ls"],
        {
            "ls_only": execution_tool_names(scanner_ls),
            "ls_glob": execution_tool_names(scanner_ls_glob),
        },
    )

    # ------------------------------------------------------------------
    # Case A: direct node insertion when workflow state is AgentState-like.
    # ------------------------------------------------------------------

    direct_graph = StateGraph(AgentLikeState)
    direct_graph.add_node("researcher", structured)
    direct_graph.add_edge(START, "researcher")
    direct_graph.add_edge("researcher", END)
    direct_app = direct_graph.compile()

    direct_result = await direct_app.ainvoke(
        {
            "messages": [
                HumanMessage(content="pineapple")
            ]
        }
    )
    direct_structured = direct_result.get("structured_response")

    failures += not report(
        "Specialized DeepAgents runnable works directly as a LangGraph node when state is AgentState-like",
        isinstance(direct_structured, Findings)
        and direct_structured.summary == "structured:pineapple",
        {
            "result_keys": sorted(direct_result),
            "structured_response": direct_structured,
        },
    )

    # ------------------------------------------------------------------
    # Case B: direct insertion into ordinary domain state.
    # This should NOT satisfy the desired workflow contract because the
    # subagent runnable expects `messages`.
    # ------------------------------------------------------------------

    domain_direct_graph = StateGraph(DomainState)
    domain_direct_graph.add_node("researcher", structured)
    domain_direct_graph.add_edge(START, "researcher")
    domain_direct_graph.add_edge("researcher", END)
    domain_direct_app = domain_direct_graph.compile()

    direct_domain_failed_as_expected = False
    direct_domain_error = ""
    try:
        await domain_direct_app.ainvoke({"topic": "pineapple"})
    except Exception as exc:
        direct_domain_error = f"{type(exc).__name__}: {exc}"
        direct_domain_failed_as_expected = (
            "PROBE_EXPECTED_HUMAN_MESSAGE" in str(exc)
            or "messages" in str(exc).lower()
        )

    failures += not report(
        "Raw subagent runnable is NOT sufficient for arbitrary domain-shaped workflow state",
        direct_domain_failed_as_expected,
        direct_domain_error or "Unexpectedly succeeded",
    )

    # ------------------------------------------------------------------
    # Case C: smallest useful MIRA workflow-node adapter.
    # Input mapping: domain state -> HumanMessage
    # Output mapping: structured_response -> workflow field
    # ------------------------------------------------------------------

    async def structured_node(state: DomainState) -> dict[str, Any]:
        result = await structured.ainvoke(
            {
                "messages": [
                    HumanMessage(content=state["topic"])
                ]
            }
        )
        return {
            "findings": result["structured_response"],
        }

    adapter_graph = StateGraph(DomainState)
    adapter_graph.add_node("researcher", structured_node)
    adapter_graph.add_edge(START, "researcher")
    adapter_graph.add_edge("researcher", END)
    adapter_app = adapter_graph.compile()

    adapter_result = await adapter_app.ainvoke({"topic": "pineapple"})
    findings = adapter_result.get("findings")

    failures += not report(
        "Tiny adapter makes structured MIRA agent natural in domain-shaped LangGraph state",
        isinstance(findings, Findings)
        and findings.summary == "structured:pineapple"
        and findings.confidence == 0.91,
        adapter_result,
    )

    # ------------------------------------------------------------------
    # Case D: same base agent without response_format returns text.
    # ------------------------------------------------------------------

    async def text_node(state: DomainState) -> dict[str, Any]:
        result = await text_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=state["topic"])
                ]
            }
        )
        messages = result["messages"]
        final = next(
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage) and message.text
        )
        return {"text": final.text}

    text_graph = StateGraph(DomainState)
    text_graph.add_node("researcher", text_node)
    text_graph.add_edge(START, "researcher")
    text_graph.add_edge("researcher", END)
    text_app = text_graph.compile()

    text_result = await text_app.ainvoke({"topic": "pineapple"})

    failures += not report(
        "Same base subagent can remain text-returning in another workflow-local specialization",
        text_result.get("text") == "text:pineapple",
        text_result,
    )

    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)

    if failures == 0:
        print(
            "PASS: workflow-local tools/response_format specialization is mechanically "
            "sound and does not mutate the reusable base definition."
        )
        print(
            "Direct StateGraph.add_node(subagent_runnable) works only when the workflow "
            "state already follows the agent/messages contract."
        )
        print(
            "For normal domain-shaped workflow state, MIRA needs only a thin node adapter: "
            "map workflow state -> subagent messages, invoke the existing DeepAgents runnable, "
            "then map text/structured_response -> workflow state update."
        )
    else:
        print(f"FAILED: {failures} check(s)")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    if failures:
        print("FAILED")
        return 1

    print("PASS: workflow subagent specialization/node boundary proven")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
