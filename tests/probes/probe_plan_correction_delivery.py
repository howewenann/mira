"""Deterministic probe: production transient Plan/Goal correction delivery.

Run from the repository root with the MIRA environment active:

    python tests/probes/probe_plan_correction_delivery.py

Uses the installed MIRA/DeepAgents/LangChain stack and two scripted fake-model
responses. No actual LLM is invoked. No production files are changed.

Tests actual native LangGraph after_agent -> model retries, model-request
content, checkpointed messages, and effective conversation used at handoff.
Checks production middleware directly, without a second correction implementation.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.middleware.correction import (  # noqa: E402
    CORRECTION_SOURCE,
    CorrectionMiddleware,
)
from agent.middleware.file_references import FileReferenceMiddleware  # noqa: E402
from agent.middleware.planning_stage_enforcement import PlanningStageEnforcementMiddleware  # noqa: E402
from agent.planning.response_status import PlanningResponseStatusRule  # noqa: E402
from core.execution.handoff import effective_conversation, synchronize_conversation  # noqa: E402


class RecordingFakeModel(FakeMessagesListChatModel):
    """Fake model which records exact provider-visible requests."""

    requests: list[list[Any]] = Field(default_factory=list, exclude=True)

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def system_text(request: list[Any]) -> str:
    return "\n".join(str(m.content) for m in request if isinstance(m, SystemMessage))


async def run_case(workflow: str) -> None:
    label = workflow.upper()
    model = RecordingFakeModel(responses=[
        AIMessage(content="I still need to inspect it.\nRESPONSE_STATUS: NEEDS_RESEARCH"),
        AIMessage(content="Completed the protocol retry.\nRESPONSE_STATUS: COMPLETE"),
    ])
    correction = CorrectionMiddleware(
        rules=(PlanningResponseStatusRule(workflow=workflow),), max_retries=2,
    )
    graph = create_agent(
        model=model,
        tools=[],
        system_prompt="Preserve this original system instruction.",
        middleware=[FileReferenceMiddleware(), PlanningStageEnforcementMiddleware(), correction],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": f"probe-correction-{label}"}}
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Please inspect @config.py before answering")],
        "planning_stage": f"{workflow}_research",
    }, config=config)
    checkpoint = (await graph.aget_state(config)).values
    assert len(model.requests) == 2, f"Expected exactly two model calls: {len(model.requests)}"
    first, second = model.requests
    assert "/config.py" in system_text(first), "File-reference system guidance missing on first request"
    assert "Your previous response classified" in (
        system_text(second) + "\n".join(str(m.content) for m in second if isinstance(m, HumanMessage))
    ), "Correction feedback missing on second invocation"

    correction_messages = [
        m for m in checkpoint["messages"]
        if isinstance(m, HumanMessage) and getattr(m, "name", None) == CORRECTION_SOURCE
    ]
    handoff_messages = effective_conversation(graph, checkpoint)
    handoff_corrections = [
        m for m in handoff_messages
        if isinstance(m, HumanMessage) and getattr(m, "name", None) == CORRECTION_SOURCE
    ]
    # Exercise the real MIRA state-copy operation, not just a simulated handoff.
    target_graph = create_agent(
        model=RecordingFakeModel(responses=[AIMessage(content="target not invoked")]),
        tools=[],
        middleware=[FileReferenceMiddleware()],
        checkpointer=InMemorySaver(),
    )
    target_id = f"probe-handoff-{label}"
    assert await synchronize_conversation(
        graph,
        config["configurable"]["thread_id"],
        target_graph,
        target_id,
    ), "Expected MIRA handoff to copy source history"
    target_state = (await target_graph.aget_state(
        {"configurable": {"thread_id": target_id}}
    )).values
    actual_handoff_corrections = [
        m for m in target_state["messages"]
        if isinstance(m, HumanMessage) and getattr(m, "name", None) == CORRECTION_SOURCE
    ]
    assert len(actual_handoff_corrections) == len(handoff_corrections), (
        "Actual Plan-to-Act-style handoff differs from effective transcript projection"
    )

    assert not correction_messages, "Correction leaked into checkpoint"
    assert not handoff_corrections, "Correction leaked into handoff projection"
    assert not actual_handoff_corrections, "Correction leaked into actual handoff"
    assert "/config.py" in system_text(second), "User @file reference was not reinjected on retry"
    assert system_text(second).count("Your previous response classified") == 1
    assert not checkpoint.get("_correction_pending"), "Pending correction not cleared"
    for request in (first, second):
        assert "Preserve this original system instruction." in system_text(request)
        assert [m.content for m in request if isinstance(m, HumanMessage)] == [
            "Please inspect @config.py before answering"
        ], "Original user request changed"
    assert len(checkpoint["messages"]) == 3, "Unexpected checkpoint message"

    print(f"\n[{label}] PASS")
    print(f"  model calls: {len(model.requests)}")
    print(f"  @file guidance: first={('/config.py' in system_text(first))}, "
          f"retry={('/config.py' in system_text(second))}")
    print(f"  correction HumanMessages: checkpoint={len(correction_messages)}, "
          f"actual handoff={len(actual_handoff_corrections)}")
    print(f"  final: {result['messages'][-1].content!r}")


async def main() -> None:
    await run_case("plan")
    await run_case("goal")
    print("\nPASS: production Plan/Goal retries use transient guidance.")
    print("NOTE: This proves graph mechanics, not whether real LLMs follow one nudge better.")


if __name__ == "__main__":
    asyncio.run(main())
