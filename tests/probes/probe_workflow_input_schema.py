# tests/probes/probe_workflow_input_schema.py

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, MessagesState, StateGraph


class InputState(TypedDict):
    topic: str
    depth: NotRequired[int]


class State(InputState):
    results: list[str]
    current_step: int
    report: str


class MessageInput(MessagesState):
    depth: NotRequired[int]


class MessageState(MessageInput):
    report: str


def build_graph(state_type, *, input_schema=None):
    graph = StateGraph(state_type, input_schema=input_schema)

    async def noop(state):
        return {}

    graph.add_node("noop", noop)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return graph.compile()


def dump(label: str, graph) -> None:
    print(f"\n{'=' * 80}")
    print(label)
    print("=" * 80)

    print("\nTYPE")
    print(type(graph))

    print("\nPUBLIC ATTRIBUTES THAT LOOK RELEVANT")
    for name in sorted(dir(graph)):
        lowered = name.lower()
        if any(token in lowered for token in ("schema", "input", "state")):
            if name.startswith("__"):
                continue
            try:
                value = getattr(graph, name)
            except Exception as exc:
                value = f"<error: {exc}>"
            print(f"{name}: {value!r}")

    print("\nget_input_schema()")
    schema = graph.get_input_schema()
    print(schema)
    print("type:", type(schema))

    print("\nget_input_jsonschema()")
    try:
        print(graph.get_input_jsonschema())
    except Exception as exc:
        print("ERROR:", repr(exc))

    print("\ninput_schema.model_json_schema()")
    try:
        print(schema.model_json_schema())
    except Exception as exc:
        print("ERROR:", repr(exc))

    print("\n__dict__")
    try:
        for key, value in graph.__dict__.items():
            lowered = key.lower()
            if any(token in lowered for token in ("schema", "input", "state")):
                print(f"{key}: {value!r}")
    except Exception as exc:
        print("ERROR:", repr(exc))


def main() -> None:
    explicit = build_graph(
        State,
        input_schema=InputState,
    )

    implicit = build_graph(
        State,
    )

    messages = build_graph(
        MessageState,
        input_schema=MessageInput,
    )

    dump("A — EXPLICIT input_schema=InputState", explicit)
    dump("B — IMPLICIT StateGraph(State)", implicit)
    dump("C — EXPLICIT MessagesState-based input schema", messages)

    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print("=" * 80)

    explicit_json = explicit.get_input_jsonschema()
    implicit_json = implicit.get_input_jsonschema()
    messages_json = messages.get_input_jsonschema()

    print("\nA input schema:")
    print(explicit_json)

    print("\nB input schema:")
    print(implicit_json)

    print("\nC input schema:")
    print(messages_json)

    print("\nA != B:", explicit_json != implicit_json)

    print("\nA properties:")
    print(explicit_json.get("properties"))

    print("\nA required:")
    print(explicit_json.get("required"))

    print("\nB properties:")
    print(implicit_json.get("properties"))

    print("\nB required:")
    print(implicit_json.get("required"))

    print("\nC properties:")
    print(messages_json.get("properties"))

    print("\nC required:")
    print(messages_json.get("required"))


if __name__ == "__main__":
    main()