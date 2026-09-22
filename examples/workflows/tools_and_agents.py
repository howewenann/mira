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
    researcher = runtime.context.agents["researcher"]
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
        result = await workflow(mira).ainvoke(
            {"topic": "native LangGraph context"},
            context=mira.context,
        )
        print(result["saved_to"])
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
