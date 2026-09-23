"""Deterministic launchable Workflow for the Phase 3 TUI walkthrough."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    topic: str
    repeat: NotRequired[int]


class State(InputState):
    lines: list[str]
    result: str


def workflow(mira):
    """Construct the graph; actual work remains inside its nodes."""
    _ = mira
    graph = StateGraph(State, input_schema=InputState)

    async def prepare(state: State) -> dict[str, object]:
        repeat = max(1, state.get("repeat", 1))
        return {
            "repeat": repeat,
            "lines": [f"{index + 1}. {state['topic']}" for index in range(repeat)],
        }

    async def finish(state: State) -> dict[str, str]:
        return {"result": "\n".join(state["lines"])}

    graph.add_node("prepare", prepare)
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


if __name__ == "__main__":
    raise SystemExit(
        "Copy this file to .mira/workflows/demo.py and launch it with /workflow__demo."
    )
