"""
Probe: can MIRA persist native LangGraph InMemorySaver state as JSON and restore it?

What this proves
----------------
1. Real LangGraph checkpoint state is created using MIRA's current MiraMemorySaver.
2. A nested subgraph interrupts, so the saver must retain resumable graph state.
3. The message history contains an offload-style ToolMessage breadcrumb.
4. The saver's native storage/writes/blobs structures are flattened into JSON.
5. A brand-new MiraMemorySaver is hydrated from that JSON.
6. The restored graph state is compared with the original.
7. The restored graph is resumed through the interrupt.
8. The completed state is persisted/reloaded again and compared.

No LLM, network, MCP server, or external model is required.

Run directly from the probe directory (or anywhere else inside the MIRA repo):

    python probe_checkpoint_json_roundtrip.py

The probe locates the MIRA repository root automatically and adds it to
`sys.path`, so the current working directory does not need to be the repo root.

Expected result:

    PASS: interrupted checkpoint round-trip matched
    PASS: restored checkpoint resumed successfully
    PASS: completed checkpoint round-trip matched
    PASS: probe succeeded

The JSON files are written under:

    .mira/_probe_checkpoint_json/

Important
---------
This is a probe, NOT a proposed production file format.

For exact round-tripping, native serialized checkpoint bytes are stored as base64.
A decoded "preview" is also written beside each record so you can inspect what is
inside with an ordinary text editor/read_file.

If this probe succeeds, it demonstrates that MIRA can preserve LangGraph's native
InMemorySaver semantics while making the backing state durable in JSON. It does
NOT yet prove that directly serializing InMemorySaver internals is the best long-
term API boundary.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    """Find the MIRA repository root from the probe file location."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "session").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the MIRA repository root. "
        "Place this probe somewhere inside the MIRA repository."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, interrupt

from session.checkpoint import (
    MiraJsonPlusSerializer,
    MiraMemorySaver,
    sanitize_checkpoint_value,
)


OUT_DIR = REPO_ROOT / ".mira" / "_probe_checkpoint_json"
INTERRUPTED_JSON = OUT_DIR / "interrupted.json"
COMPLETED_JSON = OUT_DIR / "completed.json"

THREAD_ID = "probe-json-checkpointer"

OFFLOAD_BREADCRUMB = """Tool result too large, the result of this tool call call_big was saved in the filesystem at this path: /large_tool_results/call_big

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

Here is a preview of the result:

1: alpha
2: beta
... [9990 lines truncated] ...
9993: omega
"""


def new_saver() -> MiraMemorySaver:
    """Construct the same saver+serializer pairing MIRA uses today."""
    return MiraMemorySaver(serde=MiraJsonPlusSerializer())


def build_graph(checkpointer: MiraMemorySaver):
    """Build a deterministic parent graph with a nested interrupting subgraph."""

    def seed(state: MessagesState) -> dict[str, Any]:
        del state
        return {
            "messages": [
                AIMessage(
                    content="",
                    id="ai-tool-call",
                    tool_calls=[
                        {
                            "name": "synthetic_big_tool",
                            "args": {"query": "probe"},
                            "id": "call_big",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=OFFLOAD_BREADCRUMB,
                    tool_call_id="call_big",
                    name="synthetic_big_tool",
                    id="tool-big-result",
                ),
            ]
        }

    def pause_inside_subgraph(state: MessagesState) -> dict[str, Any]:
        del state
        answer = interrupt("approve probe?")
        return {
            "messages": [
                AIMessage(
                    content=f"subgraph resumed with: {answer}",
                    id="ai-subgraph-resumed",
                )
            ]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("pause", pause_inside_subgraph)
        .add_edge(START, "pause")
        .add_edge("pause", END)
        .compile()
    )

    def finish(state: MessagesState) -> dict[str, Any]:
        del state
        return {
            "messages": [
                AIMessage(
                    content="probe complete",
                    id="ai-final",
                )
            ]
        }

    return (
        StateGraph(MessagesState)
        .add_node("seed", seed)
        .add_node("inner", inner)
        .add_node("finish", finish)
        .add_edge(START, "seed")
        .add_edge("seed", "inner")
        .add_edge("inner", "finish")
        .add_edge("finish", END)
        .compile(checkpointer=checkpointer)
    )


def encode_typed(value: tuple[str, bytes]) -> dict[str, str]:
    """Encode LangGraph's native typed serialized value losslessly for JSON."""
    type_name, payload = value
    return {
        "type": type_name,
        "data_b64": base64.b64encode(payload).decode("ascii"),
    }


def decode_typed(value: dict[str, str]) -> tuple[str, bytes]:
    """Decode one typed serialized value produced by encode_typed()."""
    return (
        value["type"],
        base64.b64decode(value["data_b64"].encode("ascii")),
    )


def json_safe(value: Any) -> Any:
    """Convert a normalized value into plain JSON-safe data for inspection."""
    value = sanitize_checkpoint_value(value)

    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, bytes | bytearray):
        return {
            "__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii"),
        }

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple | set | frozenset):
        return [json_safe(item) for item in value]

    return repr(value)


def decoded_preview(saver: MiraMemorySaver, value: tuple[str, bytes]) -> Any:
    """Decode a native saver payload into a human-readable JSON projection.

    LangGraph's InMemorySaver uses ("empty", b"") as a sentinel for a channel
    version that intentionally has no stored blob. That sentinel is saver
    bookkeeping, not a serializer payload, so JsonPlusSerializer correctly
    refuses to deserialize it.
    """
    type_name, _payload = value
    if type_name == "empty":
        return {"__langgraph_empty_blob__": True}
    return json_safe(saver.serde.loads_typed(value))


def export_saver(saver: MiraMemorySaver, path: Path) -> None:
    """Flatten native InMemorySaver internals into one JSON document."""
    document: dict[str, Any] = {
        "format": "mira-inmemory-saver-probe-v1",
        "storage": [],
        "writes": [],
        "blobs": [],
    }

    for thread_id, namespaces in saver.storage.items():
        for checkpoint_ns, checkpoints in namespaces.items():
            for checkpoint_id, (
                checkpoint,
                metadata,
                parent_checkpoint_id,
            ) in checkpoints.items():
                document["storage"].append(
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "parent_checkpoint_id": parent_checkpoint_id,
                        "checkpoint": encode_typed(checkpoint),
                        "metadata": encode_typed(metadata),
                        "decoded_preview": {
                            "checkpoint": decoded_preview(saver, checkpoint),
                            "metadata": decoded_preview(saver, metadata),
                        },
                    }
                )

    for (
        thread_id,
        checkpoint_ns,
        checkpoint_id,
    ), rows in saver.writes.items():
        for (task_id, idx), (
            stored_task_id,
            channel,
            value,
            task_path,
        ) in rows.items():
            document["writes"].append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "idx": idx,
                    "stored_task_id": stored_task_id,
                    "channel": channel,
                    "task_path": task_path,
                    "value": encode_typed(value),
                    "decoded_preview": decoded_preview(saver, value),
                }
            )

    for (
        thread_id,
        checkpoint_ns,
        channel,
        version,
    ), value in saver.blobs.items():
        document["blobs"].append(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "channel": channel,
                "version": version,
                "value": encode_typed(value),
                "decoded_preview": decoded_preview(saver, value),
            }
        )

    # Preserve the native InMemorySaver insertion order exactly.
    #
    # This matters for `writes`: InMemorySaver exposes pending writes using the
    # underlying dict insertion order. Re-sorting them for prettier JSON changes
    # the observable CheckpointTuple history after reload even though resume may
    # still work. JSON arrays already preserve order, so write the rows exactly
    # as the saver holds them.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def import_saver(path: Path) -> MiraMemorySaver:
    """Create a fresh saver and hydrate native InMemorySaver internals."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "mira-inmemory-saver-probe-v1":
        raise AssertionError(f"unexpected probe format: {document.get('format')!r}")

    saver = new_saver()

    for row in document["storage"]:
        saver.storage[row["thread_id"]][row["checkpoint_ns"]][row["checkpoint_id"]] = (
            decode_typed(row["checkpoint"]),
            decode_typed(row["metadata"]),
            row["parent_checkpoint_id"],
        )

    for row in document["writes"]:
        key = (
            row["thread_id"],
            row["checkpoint_ns"],
            row["checkpoint_id"],
        )
        saver.writes[key][(row["task_id"], row["idx"])] = (
            row["stored_task_id"],
            row["channel"],
            decode_typed(row["value"]),
            row["task_path"],
        )

    for row in document["blobs"]:
        key = (
            row["thread_id"],
            row["checkpoint_ns"],
            row["channel"],
            row["version"],
        )
        saver.blobs[key] = decode_typed(row["value"])

    return saver


def canonical_checkpoint_tuple(saver: MiraMemorySaver, config: dict[str, Any]) -> Any:
    """Return the latest CheckpointTuple in a stable inspectable representation."""
    item = saver.get_tuple(config)
    if item is None:
        return None

    return json_safe(
        {
            "config": item.config,
            "checkpoint": item.checkpoint,
            "metadata": item.metadata,
            "pending_writes": item.pending_writes,
            "parent_config": item.parent_config,
        }
    )


def canonical_history(saver: MiraMemorySaver, config: dict[str, Any]) -> list[Any]:
    """Return the complete checkpoint history in stable representation."""
    return [
        json_safe(
            {
                "config": item.config,
                "checkpoint": item.checkpoint,
                "metadata": item.metadata,
                "pending_writes": item.pending_writes,
                "parent_config": item.parent_config,
            }
        )
        for item in saver.list(config)
    ]


def checkpoint_namespaces(saver: MiraMemorySaver) -> list[str]:
    """Return all checkpoint namespaces captured for the probe thread."""
    return sorted(saver.storage.get(THREAD_ID, {}).keys())


def assert_same_checkpoint_state(
    left: MiraMemorySaver,
    right: MiraMemorySaver,
    config: dict[str, Any],
) -> None:
    """Assert latest checkpoint and complete history are identical."""
    left_latest = canonical_checkpoint_tuple(left, config)
    right_latest = canonical_checkpoint_tuple(right, config)
    assert left_latest == right_latest, "latest CheckpointTuple changed after JSON round-trip"

    left_history = canonical_history(left, config)
    right_history = canonical_history(right, config)
    if left_history != right_history:
        for index, (left_item, right_item) in enumerate(
            zip(left_history, right_history, strict=False)
        ):
            if left_item != right_item:
                print(f"first checkpoint-history mismatch at index {index}")
                print("LEFT:")
                print(json.dumps(left_item, indent=2, ensure_ascii=False))
                print("RIGHT:")
                print(json.dumps(right_item, indent=2, ensure_ascii=False))
                break
        if len(left_history) != len(right_history):
            print(
                "history lengths differ: "
                f"left={len(left_history)} right={len(right_history)}"
            )
        raise AssertionError("checkpoint history changed after JSON round-trip")

    assert checkpoint_namespaces(left) == checkpoint_namespaces(right), (
        "checkpoint namespaces changed after JSON round-trip"
    )


def message_summary(graph: Any, config: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Return compact message identity/type/content summary from graph state."""
    snapshot = graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    return [
        (
            message.__class__.__name__,
            str(getattr(message, "content", "")),
            getattr(message, "id", None),
        )
        for message in messages
    ]


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {"configurable": {"thread_id": THREAD_ID}}

    # ------------------------------------------------------------------
    # 1. Create real checkpoint state and stop at a nested interrupt.
    # ------------------------------------------------------------------
    original_saver = new_saver()
    original_graph = build_graph(original_saver)

    interrupted_result = original_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="start checkpoint probe",
                    id="human-start",
                )
            ]
        },
        config,
    )

    assert "__interrupt__" in interrupted_result, (
        "probe graph did not interrupt; cannot test resumable checkpoint state"
    )

    original_messages = message_summary(original_graph, config)
    assert any(
        message_id == "tool-big-result" and "/large_tool_results/call_big" in content
        for _, content, message_id in original_messages
    ), "offload-style ToolMessage breadcrumb was not checkpointed"

    export_saver(original_saver, INTERRUPTED_JSON)

    # ------------------------------------------------------------------
    # 2. Simulate process restart: brand-new saver, loaded only from JSON.
    # ------------------------------------------------------------------
    restored_saver = import_saver(INTERRUPTED_JSON)
    restored_graph = build_graph(restored_saver)

    assert_same_checkpoint_state(original_saver, restored_saver, config)
    assert message_summary(original_graph, config) == message_summary(restored_graph, config)

    namespaces = checkpoint_namespaces(restored_saver)
    print(f"checkpoint namespaces after restore: {namespaces!r}")
    print(
        "native row counts after restore: "
        f"storage={sum(len(cps) for ns in restored_saver.storage.values() for cps in ns.values())}, "
        f"writes={sum(len(rows) for rows in restored_saver.writes.values())}, "
        f"blobs={len(restored_saver.blobs)}"
    )
    print("PASS: interrupted checkpoint round-trip matched")

    # ------------------------------------------------------------------
    # 3. Resume ONLY the restored graph. No original process state is used.
    # ------------------------------------------------------------------
    completed_result = restored_graph.invoke(
        Command(resume="approved-after-restart"),
        config,
    )

    assert "__interrupt__" not in completed_result, "restored graph interrupted again unexpectedly"

    completed_messages = message_summary(restored_graph, config)
    assert any(
        message_id == "ai-subgraph-resumed"
        and "approved-after-restart" in content
        for _, content, message_id in completed_messages
    ), "resumed subgraph did not receive the post-restart resume value"

    assert completed_messages[-1] == (
        "AIMessage",
        "probe complete",
        "ai-final",
    ), f"unexpected final message after restore/resume: {completed_messages[-1]!r}"

    print("PASS: restored checkpoint resumed successfully")

    # ------------------------------------------------------------------
    # 4. Persist the completed state and reload it one more time.
    # ------------------------------------------------------------------
    export_saver(restored_saver, COMPLETED_JSON)

    completed_reloaded_saver = import_saver(COMPLETED_JSON)
    completed_reloaded_graph = build_graph(completed_reloaded_saver)

    assert_same_checkpoint_state(restored_saver, completed_reloaded_saver, config)
    assert message_summary(restored_graph, config) == message_summary(
        completed_reloaded_graph,
        config,
    )

    print("PASS: completed checkpoint round-trip matched")
    print(f"interrupted JSON: {INTERRUPTED_JSON}")
    print(f"completed JSON:   {COMPLETED_JSON}")
    print("PASS: probe succeeded")


if __name__ == "__main__":
    main()
