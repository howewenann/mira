"""Ordinary LangChain tools may consume MIRA's trusted runtime context."""

import asyncio

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from mira import MiraApplication, MiraContext


@tool
async def convert_pdf(
    source: str,
    destination: str,
    runtime: ToolRuntime,
) -> str:
    """Convert a PDF to Markdown and save the result."""
    convert = runtime.context.tools["mcp__converter__pdf_to_markdown"]
    write = runtime.context.tools["write_file"]
    markdown = await convert.ainvoke({"path": source})
    await write.ainvoke({"file_path": destination, "content": markdown})
    return destination


@tool
async def research_topic(topic: str, runtime: ToolRuntime) -> str:
    """Ask the configured researcher subagent to investigate a topic."""
    # `general-purpose` is MIRA's default configured subagent. If the workspace
    # defines another subagent, use its configured name instead:
    #
    #     runtime.context.agents["my-researcher"]
    #
    # This differs from `mira.agent(name="researcher")`, which returns a
    # workflow-local runnable and does not add it to `runtime.context.agents`.
    researcher = runtime.context.agents["general-purpose"]
    return str(await researcher.ainvoke(f"Research this topic: {topic}"))


async def main() -> None:
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows

        graph_builder = StateGraph(MessagesState, context_schema=MiraContext)
        graph_builder.add_node("tools", ToolNode([research_topic]))
        graph_builder.add_edge(START, "tools")
        graph_builder.add_edge("tools", END)
        graph = graph_builder.compile()

        result = await graph.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": research_topic.name,
                                "args": {
                                    "topic": (
                                        "In one sentence, explain native LangGraph "
                                        "context from existing knowledge. Do not use tools."
                                    )
                                },
                                "id": "example-research",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            context=mira.context,
        )
        if result.get("__interrupt__"):
            print("[interrupt] The workflow paused for tool approval.")
        else:
            print(result["messages"][-1].content)
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
