"""The smallest native LangGraph workflow using a MIRA subagent."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
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

        # `graph.astream(..., stream_mode="updates")` is native LangGraph
        # streaming for completed node/state updates, not token-level output.
        #
        # For LangChain/DeepAgents V3 event streams from an agent itself, use:
        #
        #     worker = mira.agent(name="worker")
        #     run = await worker.astream_events(
        #         {"messages": [HumanMessage("Research pineapples.")]},
        #         version="v3",
        #     )
        #     async for message in run.messages:
        #         ...
        async for update in graph.astream(
            {"messages": [HumanMessage("Summarize this workspace.")]},
            context=mira.context,
            stream_mode="updates",
        ):
            for node_name, values in update.items():
                if node_name == "__interrupt__":
                    print("[interrupt] The workflow paused for tool approval.")
                    continue
                for message in values.get("messages", []):
                    if isinstance(message, AIMessage) and message.text:
                        print(f"[{node_name}] {message.text}")
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
