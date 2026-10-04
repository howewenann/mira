# Native LangGraph workflows with MIRA

MIRA discovers only direct Python files under `.mira/workflows/`; nested
directories are not scanned. A file named:

```text
.mira/workflows/research.py
        ↓
/workflow__research
```

must expose one synchronous, side-effect-free graph factory:

```python
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    topic: str
    depth: NotRequired[int]


class State(InputState):
    results: list[str]
    report: str


def workflow(mira):
    graph = StateGraph(
        State,
        input_schema=InputState,
    )

    async def research(state: State):
        return {"results": [], "report": f"Report about {state['topic']}"}

    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    return graph.compile()
```

The explicit `input_schema` is the public launch contract; the internal State
may contain additional fields. Launch values always use `name=value`:

```text
/workflow__research topic="Cubaris isopods" depth=3
```

JSON scalars, lists, and objects retain their types. MIRA owns discovery,
validation, observation, HITL resume, and presentation, while your code owns the
native `StateGraph`, topology, reducers, nodes, loops, `Send()` calls,
parallelism, and compilation.

`workflow(mira)` should only construct and compile the graph. Put actual work
inside graph nodes. Each invocation renders as one Workflow execution tree;
its **Final state** action and native task or nested-agent rows open the existing
Inspector. Completed and interrupted Workflow history is stored with the MIRA
session, so its values and nested agent execution remain inspectable after a
restart.

Live execution remains process-owned. If MIRA exits while a Workflow is active,
that invocation becomes a frozen historical `INTERRUPTED` record. It is not
resumed or reconnected to a new graph. Run the `/workflow__...` command again to
create a new invocation with a new identity.

Read the examples in this order:

1. `demo.py` - deterministic launch and final-state inspection.
2. `minimal.py` - a workflow-local agent called from a domain-state node.
3. `structured_agents.py` - independent text and structured variants.
4. `tools_and_agents.py` - deterministic nodes using MIRA runtime capabilities.

The latter examples can also be run directly. They start a headless
`MiraApplication`; configure the workspace's models and subagents, then use
`python <example>.py`.
