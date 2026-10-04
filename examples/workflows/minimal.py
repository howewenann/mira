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
    worker = mira.agent(name="worker", system_prompt="Summarize the topic briefly.")

    async def summarize(state: State) -> dict[str, str]:
        result = await worker.ainvoke(
            {"messages": [HumanMessage(state["topic"])]}
        )
        return {"summary": result["messages"][-1].text}

    graph = StateGraph(State, input_schema=InputState, context_schema=MiraContext)
    graph.add_node("worker", summarize)
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    return graph.compile()


async def main() -> None:
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = workflow(mira)

        async for update in graph.astream(
            {"topic": "this workspace"},
            context=mira.context,
            stream_mode="updates",
        ):
            for node_name, values in update.items():
                if node_name == "__interrupt__":
                    print("[interrupt] The workflow paused for tool approval.")
                    continue
                if "summary" in values:
                    print(f"[{node_name}] {values['summary']}")
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
