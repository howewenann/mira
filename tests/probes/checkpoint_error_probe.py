"""Probe whether LangGraph persists failed-task errors in MIRA SQLite checkpoints.

From the MIRA repository root:
    python tests/probes/checkpoint_error_probe.py

Optional:
    python tests/probes/checkpoint_error_probe.py --output checkpoint_probe.json

This deliberately uses a fake graph node, not an LLM. It isolates LangGraph's
native persistence behavior with MIRA's REAL session_checkpointer. The node
streams tokens via LangGraph's native custom stream and then raises an error.
The SQLite connection is closed and reopened before inspecting its contents.

Runs twice: with ``durability='async'`` (MIRA's normal/default-like behavior)
and ``durability='exit'`` (DCode's explicit mode).

No modifications to MIRA's real sessions. No API keys or running model required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages

from session.checkpoint import session_checkpointer


class ProbeState(TypedDict):
    messages: Annotated[list, add_messages]
    fail: bool


class DeliberateModelFailure(RuntimeError):
    pass


async def mock_model(state: ProbeState) -> dict[str, Any]:
    """Emit display-only chunks, then either commit an AIMessage or fail."""
    write = get_stream_writer()
    parts = [f"token-{i:03d}" for i in range(1, 121)]
    chunks_to_emit = 100 if state["fail"] else 120

    for i, word in enumerate(parts[:chunks_to_emit], start=1):
        write({"type": "text_delta", "text": word + " "})
        await asyncio.sleep(0)  # yield to the stream consumer
        if state["fail"] and i == chunks_to_emit:
            raise DeliberateModelFailure("PROBE: model connection dropped after 100 tokens")

    return {"messages": [AIMessage(content=" ".join(parts[:chunks_to_emit]))]}


def make_graph(saver: Any) -> Any:
    builder = StateGraph(ProbeState)
    builder.add_node("model", mock_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    return builder.compile(checkpointer=saver)


def extract_messages(snapshot: Any) -> list[dict[str, str]]:
    result = []
    for msg in (snapshot.values or {}).get("messages", []):
        result.append({"type": str(getattr(msg, "type", type(msg).__name__)),
                       "content": str(getattr(msg, "content", msg))})
    return result


def extract_task_errors(snapshot: Any) -> list[dict[str, str]]:
    errors = []
    for task in getattr(snapshot, "tasks", ()) or ():
        err = getattr(task, "error", None)
        if err is not None:
            errors.append({"task_name": str(getattr(task, "name", "")),
                           "error": str(err)})
    return errors


async def extract_pending_errors(saver: Any, config: dict[str, Any]) -> list[dict[str, str]]:
    """Also check the raw LangGraph pending-writes channel (__error__)."""
    tup = await saver.aget_tuple(config)
    errors = []
    if tup is None:
        return errors
    for task_id, channel, value in tup.pending_writes or []:
        if channel == "__error__":
            errors.append({"task_id": str(task_id), "error": str(value)})
    return errors


async def describe(graph: Any, saver: Any, config: dict[str, Any]) -> dict[str, Any]:
    snapshot = await graph.aget_state(config)
    history = [s async for s in graph.aget_state_history(config, limit=12)]
    return {
        "messages": extract_messages(snapshot),
        "latest_tasks": extract_task_errors(snapshot),
        "history_tasks": [
            {"checkpoint_id": str((s.config.get("configurable") or {}).get("checkpoint_id")),
             "errors": extract_task_errors(s)}
            for s in history if extract_task_errors(s)
        ],
        "raw_pending_error_writes": await extract_pending_errors(saver, config),
        "checkpoint_count_in_sample": len(history),
        "next": list(snapshot.next),
    }


async def stream_turn(graph: Any, config: dict[str, Any], *, fail: bool,
                      durability: str) -> dict[str, Any]:
    observed: list[str] = []
    caught = None
    try:
        async for mode, payload in graph.astream(
            {"messages": [HumanMessage(content="Please explain checkpoints")], "fail": fail},
            config=config,
            stream_mode=["custom", "updates"],
            durability=durability,
        ):
            if mode == "custom" and isinstance(payload, dict) and payload.get("type") == "text_delta":
                observed.append(str(payload["text"]))
    except DeliberateModelFailure as exc:
        caught = f"{type(exc).__name__}: {exc}"
    return {"streamed_chunks": len(observed),
            "streamed_text": "".join(observed).strip(),
            "caught_exception": caught}


async def run_mode(directory: Path, durability: str) -> dict[str, Any]:
    path = directory / f"probe_{durability}.sqlite"
    config = {"configurable": {"thread_id": f"probe-{durability}"}}

    async with session_checkpointer(path) as saver:
        graph = make_graph(saver)
        success = await stream_turn(graph, config, fail=False, durability=durability)
        failure = await stream_turn(graph, config, fail=True, durability=durability)
        before_restart = await describe(graph, saver, config)

    # Real SQLite close/reopen; no in-memory graph or saver reused.
    async with session_checkpointer(path) as saver:
        graph = make_graph(saver)
        after_restart = await describe(graph, saver, config)

    msgs = after_restart["messages"]
    completed_output = any("token-120" in x["content"] for x in msgs)
    incomplete_output = any("token-100" in x["content"] and "token-120" not in x["content"]
                            for x in msgs)
    errors = (
        after_restart["latest_tasks"]
        + [e for item in after_restart["history_tasks"] for e in item["errors"]]
        + after_restart["raw_pending_error_writes"]
    )
    return {
        "durability": durability,
        "successful_turn": success,
        "failed_turn": failure,
        "sqlite_before_restart": before_restart,
        "sqlite_after_restart": after_restart,
        "checks": {
            "exception_was_raised": failure["caught_exception"] is not None,
            "partial_text_was_streamed": failure["streamed_chunks"] == 100,
            "completed_answer_restored": completed_output,
            "failed_partial_answer_not_restored": not incomplete_output,
            "error_found_after_restart": bool(errors),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mira_checkpoint_error_probe_") as tmp:
        results = [await run_mode(Path(tmp), mode) for mode in ("async", "exit")]

    print("\nMIRA NATIVE CHECKPOINT ERROR PROBE")
    print("=" * 45)
    for item in results:
        print(f"\nDurability: {item['durability']}")
        print("Live streamed text (tail):", item["failed_turn"]["streamed_text"][-90:])
        print("Raised exception:", item["failed_turn"]["caught_exception"])
        print("Restored task errors:", item["sqlite_after_restart"]["latest_tasks"])
        print("Historical errors:", item["sqlite_after_restart"]["history_tasks"])
        print("Raw __error__ pending writes:",
              item["sqlite_after_restart"]["raw_pending_error_writes"])
        for name, ok in item["checks"].items():
            print(f"  {'YES' if ok else 'NO '}  {name}")
        print("IMPORTANT RESULT:",
              "native error REPLAY POSSIBLE" if item["checks"]["error_found_after_restart"]
              else "native error NOT found on restart")

    report = {"results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("\nJSON report:", args.output.resolve())
    print("\nNote: This probes a handled node exception, not an OS process kill or"
          " exceptions thrown by the TUI outside graph execution.")


if __name__ == "__main__":
    asyncio.run(main())
