"""Independent text and structured specializations of MIRA subagents."""

import asyncio
from typing import NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mira import MiraApplication, MiraContext


class Findings(BaseModel):
    summary: str
    confidence: float


class State(TypedDict):
    topic: str
    scan: NotRequired[str]
    findings: NotRequired[Findings]
    plain: NotRequired[str]


def workflow(mira):
    scanner = mira.agent(name="scanner", tools=["ls", "glob"])
    researcher = mira.agent("researcher", response_format=Findings)
    text_researcher = mira.agent("researcher", response_format=None)

    async def scan(state: State) -> State:
        result = await scanner.ainvoke(
            {"messages": [HumanMessage(f"Find files related to {state['topic']}.")]}
        )
        return {"scan": result["messages"][-1].text}

    async def research(state: State) -> State:
        result = await researcher.ainvoke(
            {"messages": [HumanMessage(state["topic"])]}
        )
        return {"findings": result["structured_response"]}

    async def plain_research(state: State) -> State:
        result = await text_researcher.ainvoke(
            {"messages": [HumanMessage(state["topic"])]}
        )
        return {"plain": result["messages"][-1].text}

    graph = StateGraph(State, context_schema=MiraContext)
    graph.add_node("scan", scan)
    graph.add_node("research", research)
    graph.add_node("plain_research", plain_research)
    graph.add_edge(START, "scan")
    graph.add_edge("scan", "research")
    graph.add_edge("research", "plain_research")
    graph.add_edge("plain_research", END)
    return graph.compile()


async def main() -> None:
    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        result = await workflow(mira).ainvoke(
            {"topic": "MIRA execution context"},
            context=mira.context,
        )
        print(result["findings"])
    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
