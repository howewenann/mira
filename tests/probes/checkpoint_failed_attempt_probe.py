"""Probe saving a FAILED streamed attempt in native LangGraph state.

Copy this file into MIRA/tests/probes/ and run from the MIRA repository root:

    python tests/probes/checkpoint_failed_attempt_probe.py
    python tests/probes/checkpoint_failed_attempt_probe.py --output failed_attempt_probe.json

No LLM, LM Studio, network, or real MIRA session is used. This probe uses
MIRA's actual session.checkpoint.session_checkpointer (SQLite + serializer).

It tests two ways to commit ONE accumulated diagnostic after a node exception:
  * automatic: graph.aupdate_state(config, {"failed_attempts": [record]})
  * explicit:  graph.aupdate_state(..., as_node="prepare")

Both variants deliberately operate on a SIMPLE graph, START -> prepare -> model
-> END. They must not be assumed safe for MIRA's full DeepAgents graph until
validated against that graph's middleware, interrupts, and branching.

Important: aupdate_state creates a real checkpoint and can change the graph's
next tasks. This probe *checks* for such changes rather than presuming safety.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages

from session.checkpoint import session_checkpointer


class State(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    failed_attempts: Annotated[list[dict[str, Any]], operator.add]
    prepare_count: int


class ProbeModelError(RuntimeError):
    """An intentional failure after partial text/reasoning was streamed."""


def build_graph(saver: Any, *, model_fails: bool) -> Any:
    async def prepare(state: State) -> dict[str, Any]:
        # Detect if a post-failure state update causes a predecessor to re-run.
        return {"prepare_count": int(state.get("prepare_count", 0)) + 1}

    async def model(state: State) -> dict[str, Any]:
        if not model_fails:
            return {"messages": [AIMessage(content="RECOVERED: model completed on retry.")]}

        write = get_stream_writer()
        for i in range(1, 41):
            write({"kind": "reasoning_delta", "text": f"thought-{i:02d} "})
            await asyncio.sleep(0)
        for i in range(1, 41):
            write({"kind": "text_delta", "text": f"word-{i:02d} "})
            await asyncio.sleep(0)
        raise ProbeModelError("PROBE: provider disconnected after partial reasoning and text")

    builder = StateGraph(State)
    builder.add_node("prepare", prepare)
    builder.add_node("model", model)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "model")
    builder.add_edge("model", END)
    return builder.compile(checkpointer=saver)


def get_messages(snapshot: Any) -> list[dict[str, str]]:
    return [{"role": str(getattr(m, "type", type(m).__name__)),
             "text": str(getattr(m, "content", m))}
            for m in (snapshot.values or {}).get("messages", [])]


def error_tasks(snapshot: Any) -> list[dict[str, str]]:
    result = []
    for task in getattr(snapshot, "tasks", ()) or ():
        error = getattr(task, "error", None)
        if error is not None:
            result.append({"name": str(getattr(task, "name", "")), "error": str(error)})
    return result


async def checkpoint_view(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    snap = await graph.aget_state(config)
    history = [s async for s in graph.aget_state_history(config, limit=30)]
    return {
        "next": list(snap.next),
        "prepare_count": int((snap.values or {}).get("prepare_count", 0)),
        "messages": get_messages(snap),
        "failed_attempts": list((snap.values or {}).get("failed_attempts", [])),
        "latest_errors": error_tasks(snap),
        "historical_errors": [
            {"checkpoint_id": str((s.config.get("configurable") or {}).get("checkpoint_id", "")),
             "tasks": error_tasks(s)}
            for s in history if error_tasks(s)
        ],
        "history_count": len(history),
    }


async def collect_failing_stream(graph: Any, config: dict[str, Any], durability: str) -> dict[str, Any]:
    stream = {"reasoning": [], "text": [], "exception": None}
    try:
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(content="Debug MIRA checkpoint restoration")]},
            config=config,
            stream_mode=["custom", "updates"],
            durability=durability,
        ):
            if mode == "custom" and isinstance(chunk, dict):
                if chunk.get("kind") == "reasoning_delta":
                    stream["reasoning"].append(str(chunk.get("text", "")))
                elif chunk.get("kind") == "text_delta":
                    stream["text"].append(str(chunk.get("text", "")))
    except ProbeModelError as exc:
        stream["exception"] = f"{type(exc).__name__}: {exc}"
    stream["reasoning"] = "".join(stream["reasoning"]).strip()
    stream["text"] = "".join(stream["text"]).strip()
    return stream


async def experiment(directory: Path, durability: str, strategy: str) -> dict[str, Any]:
    db = directory / f"failed_attempt_{durability}_{strategy}.sqlite"
    config = {"configurable": {"thread_id": f"probe-{durability}-{strategy}"}}
    update_exception: str | None = None
    retry_exception: str | None = None
    after_update: dict[str, Any] | None = None
    after_restart: dict[str, Any] | None = None
    after_retry: dict[str, Any] | None = None

    async with session_checkpointer(db) as saver:
        failing_graph = build_graph(saver, model_fails=True)
        emitted = await collect_failing_stream(failing_graph, config, durability)
        before_update = await checkpoint_view(failing_graph, config)
        # This is the ONLY persistence of partial streamed output.
        # It happens once after the failed graph invocation has exited.
        attempt = {
            "attempt_id": "failed-model-001",
            "node": "model",
            "status": "error",
            "reasoning": emitted["reasoning"],
            "partial_text": emitted["text"],
            "error": emitted["exception"],
        }
        if emitted["exception"]:
            try:
                await failing_graph.aupdate_state(
                    config,
                    {"failed_attempts": [attempt]},
                    **({"as_node": "prepare"} if strategy == "explicit" else {}),
                )
                after_update = await checkpoint_view(failing_graph, config)
            except Exception as exc:
                update_exception = f"{type(exc).__name__}: {exc}"

    # Close the saver and graph, then actually reopen the DB as MIRA does.
    async with session_checkpointer(db) as saver:
        resumed_graph = build_graph(saver, model_fails=False)
        after_restart = await checkpoint_view(resumed_graph, config)
        if after_update is not None:
            try:
                await resumed_graph.ainvoke(None, config=config, durability=durability)
                after_retry = await checkpoint_view(resumed_graph, config)
            except Exception as exc:
                retry_exception = f"{type(exc).__name__}: {exc}"

    baseline_messages = before_update["messages"]
    persisted = after_restart["failed_attempts"]
    errors = after_restart["historical_errors"] or after_restart["latest_errors"]
    checks = {
        "native_error_detected_before_update": bool(before_update["latest_errors"]),
        "partial_reasoning_and_text_streamed": (
            "thought-40" in emitted["reasoning"] and "word-40" in emitted["text"]
        ),
        "diagnostic_committed_once": len(persisted) == 1 and persisted[0].get("reasoning") == emitted["reasoning"]
            and persisted[0].get("partial_text") == emitted["text"],
        "next_tasks_unchanged_by_update": (
            after_update is not None and before_update["next"] == after_update["next"]
        ),
        "next_tasks_survive_restart": before_update["next"] == after_restart["next"],
        "prior_messages_unchanged": baseline_messages == after_restart["messages"],
        "original_exception_still_in_history": bool(errors),
        "resumed_at_model_without_rerunning_prepare": (
            after_retry is not None and after_retry["prepare_count"] == 1
        ),
        "normal_model_completion_after_resume": (
            after_retry is not None
            and any("RECOVERED: model completed" in m["text"] for m in after_retry["messages"])
        ),
        "diagnostic_survives_successful_resume": (
            after_retry is not None and after_retry["failed_attempts"] == persisted
        ),
    }
    return {
        "durability": durability,
        "strategy": strategy,
        "failed_stream": emitted,
        "before_update": before_update,
        "after_update": after_update,
        "after_restart": after_restart,
        "after_retry": after_retry,
        "update_exception": update_exception,
        "retry_exception": retry_exception,
        "checks": checks,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write complete JSON evidence here")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mira_failed_attempt_probe_") as temp:
        results = [
            await experiment(Path(temp), durability, strategy)
            for durability in ("async", "exit")
            for strategy in ("automatic", "explicit")
        ]

    print("\nMIRA FAILED-ATTEMPT CHECKPOINT PROBE")
    print("=" * 45)
    for result in results:
        print(f"\nDurability: {result['durability']} | as_node: {result['strategy']}")
        print("Failed stream: reasoning + text chars =",
              len(result["failed_stream"]["reasoning"]), "+",
              len(result["failed_stream"]["text"]))
        print("Pending tasks before -> after update -> restart:",
              result["before_update"]["next"], "->",
              (result["after_update"] or {}).get("next"), "->",
              result["after_restart"]["next"])
        print("Original native error still recorded in checkpoint history:",
              result["after_restart"]["historical_errors"])
        if result["update_exception"]:
            print("STATE UPDATE FAILED:", result["update_exception"])
        if result["retry_exception"]:
            print("RETRY FAILED:", result["retry_exception"])
        for title, ok in result["checks"].items():
            print(f"  {'PASS' if ok else 'FAIL'}  {title}")
        print("VERDICT:", "PASS (simple graph only)" if all(result["checks"].values())
              else "NOT SAFE / NOT PROVEN")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print("\nFull report:", args.output.resolve())
    print("\nWARNING: A passing toy graph does NOT establish safety in MIRA's full")
    print("DeepAgents graph. as_node can change pending tasks and interrupts.")


if __name__ == "__main__":
    asyncio.run(main())
