"""A small LangGraph workflow with a MIRA agent called from a node."""

import asyncio
from typing import NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from mira import MiraApplication, MiraContext


class InputState(TypedDict):
    topic: str


class State(InputState):
    summary: NotRequired[str]


def workflow(mira):
    # Standalone: construct this worker with DeepAgents and your own model/tools.
    worker = mira.agent(name="worker", system_prompt="Summarize the topic briefly.")

    async def summarize(state: State) -> dict[str, str]:
        result = await worker.ainvoke(
            {"messages": [HumanMessage(state["topic"])]}
        )
        return {"summary": result["messages"][-1].text}

    # Standalone: omit MIRA's context schema when nodes need no runtime context.
    graph = StateGraph(State, input_schema=InputState, context_schema=MiraContext)
    graph.add_node("worker", summarize)
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    return graph.compile()


async def main() -> None:
    # Standalone: run in your own host; omit MIRA startup and unused context.
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = workflow(mira)

        result = await graph.ainvoke({"topic": "this workspace"}, context=mira.context)
        if "__interrupt__" in result:
            print(f"[interrupt] Partial run at tool approval: {result['__interrupt__']}")
            return
        if "summary" not in result:
            raise RuntimeError("The final summary was not observed.")
        print(f"[complete] Final state: {result}")
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
