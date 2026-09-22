"""Deterministic LangGraph nodes using MIRA tools and subagents."""

import asyncio
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from mira import MiraApplication, MiraContext


class State(TypedDict):
    topic: str
    report: NotRequired[str]
    saved_to: NotRequired[str]


async def research(state: State, runtime: Runtime[MiraContext]) -> State:
    # `general-purpose` is MIRA's default configured subagent. A workspace may
    # select another configured subagent by its configured name, for example:
    #
    #     runtime.context.agents["my-researcher"]
    #
    # `mira.agent(name="researcher")` is different: it returns a workflow-local
    # runnable and does not add that specialization to this context mapping.
    researcher = runtime.context.agents["general-purpose"]
    report = await researcher.ainvoke(f"Research this topic: {state['topic']}")
    return {"report": str(report)}


async def save(state: State, runtime: Runtime[MiraContext]) -> State:
    destination = "/workflow-report.md"
    write = runtime.context.tools["write_file"]
    await write.ainvoke({"file_path": destination, "content": state["report"]})
    return {"saved_to": destination}


def workflow(_mira):
    graph = StateGraph(State, context_schema=MiraContext)
    graph.add_node("research", research)
    graph.add_node("save", save)
    graph.add_edge(START, "research")
    graph.add_edge("research", "save")
    graph.add_edge("save", END)
    return graph.compile()


async def main() -> None:
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = workflow(mira)

        # Native LangGraph `updates` streaming reports completed node/state
        # updates rather than token-level output. For typed LangChain/DeepAgents
        # V3 events from an individual agent, use
        # `agent.astream_events(..., version="v3")` and consume projections such
        # as `run.messages`, `run.tool_calls`, or `run.subagents`.
        async for update in graph.astream(
            {
                "topic": (
                    "Explain native LangGraph context in one sentence from existing "
                    "knowledge. Do not use tools."
                )
            },
            context=mira.context,
            stream_mode="updates",
        ):
            for node_name, values in update.items():
                if node_name == "__interrupt__":
                    print("[interrupt] The workflow paused for tool approval.")
                    continue
                if "report" in values:
                    print(f"[{node_name}] {values['report']}")
                if "saved_to" in values:
                    print(f"[{node_name}] {values['saved_to']}")
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
