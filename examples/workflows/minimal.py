"""The smallest native LangGraph workflow using a MIRA subagent."""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from mira import MiraApplication, MiraContext


def workflow(mira):
    graph = StateGraph(MessagesState, context_schema=MiraContext)
    graph.add_node("worker", mira.agent())
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    return graph.compile()


async def main() -> None:
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = workflow(mira)
        result = await graph.ainvoke(
            {"messages": [HumanMessage("Summarize this workspace.")]},
            context=mira.context,
        )
        print(result["messages"][-1].text)
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
