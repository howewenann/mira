"""Read-only probe: stock LangGraph checkpoint serialization in MIRA.

Run with MIRA's Python environment, from the MIRA repo root:
    python probe_default_checkpoint_serde.py --repo .

Or when the probe is outside the repository:
    python probe_default_checkpoint_serde.py --repo D:\\Projects\\mira

The probe NEVER edits MIRA source or existing sessions. It uses a disposable
workspace.

Two investigations:
  1. Does stock JsonPlusSerializer handle normal LangGraph values and reject
     historical problematic values (Pydantic serializer internals, classes)?
  2. Can real MIRA Act/Plan/retained Plan state survive a *process restart*
     when using stock AsyncSqliteSaver and stock JsonPlusSerializer?

It deliberately does not claim full subagent/rubric/HITL/compaction coverage.
The comprehensive probe covers those paths.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from contextlib import nullcontext


def boot_repo(repo: Path) -> None:
    if not (repo / "session" / "checkpoint.py").is_file():
        raise FileNotFoundError(f"Not a MIRA checkout: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def run_typed_value_checks() -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.serde.types import _DeltaSnapshot
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from pydantic import BaseModel
    from pydantic_core import SchemaSerializer, SchemaValidator

    from tests.test_checkpoint import Finding

    default = JsonPlusSerializer()
    supported = {
        "plain graph data": {"count": 2, "status": "ready"},
        "AIMessage": AIMessage(content="hello"),
        "DeltaSnapshot": _DeltaSnapshot([
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            ToolMessage(content="ok", tool_call_id="tool-1"),
        ]),
        "Pydantic instance": Finding(summary="ok"),
    }
    unusual = {
        "Python class object": AIMessage,
        "Pydantic MockValSer internal": BaseModel.__pydantic_serializer__,
        "Pydantic SchemaSerializer": SchemaSerializer({"type": "any"}),
        "Pydantic SchemaValidator": SchemaValidator({"type": "any"}),
    }
    print("\n=== A. STOCK SERIALIZER DIRECT ROUND TRIPS ===", flush=True)
    for name, obj in supported.items():
        kind, raw = default.dumps_typed(obj)
        restored = default.loads_typed((kind, raw))
        assert restored is not None, name
        if name == "AIMessage":
            assert isinstance(restored, AIMessage) and restored.content == "hello"
        elif name == "DeltaSnapshot":
            assert isinstance(restored, _DeltaSnapshot)
            assert [msg.content for msg in restored.value] == ["hello", "hi", "ok"]
        elif name == "Pydantic instance":
            assert getattr(restored, "summary", None) == "ok", (
                "Stock serializer did not preserve the Pydantic model instance", restored
            )
        else:
            assert restored == obj
        print(f"  PASS  {name}: {kind}, type={type(restored).__name__}", flush=True)

    print("\n=== B. HISTORICALLY PROBLEMATIC VALUES (INFORMATIONAL) ===", flush=True)
    print("  Failure here is EXPECTED and not a failure of normal graph state.", flush=True)
    for name, obj in unusual.items():
        for label, serde in (("stock", default),):
            try:
                kind, raw = serde.dumps_typed({"value": obj})
                restored = serde.loads_typed((kind, raw))
                value = restored.get("value") if isinstance(restored, dict) else restored
                print(f"  {label:5s} {name}: OK -> {str(value)[:115]}", flush=True)
            except Exception as exc:
                print(f"  {label:5s} {name}: {type(exc).__name__}: {str(exc)[:110]}", flush=True)


async def run_mira_phase(repo: Path, workspace: Path, phase: str) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langchain_core.messages import HumanMessage

    # Use MIRA's established fake-model fixture; it exercises MIRA's real
    # application, agent factory, graph and checkpoint wiring offline.
    from tests.test_sqlite_sessions import SqliteSessionTests
    from session.plans import plan_artifact, replace_current_plan

    test_case = SqliteSessionTests("test_cold_restart_restores_messages_and_session_state")

    # On app.open_session(), the constructed saver/serializer must be stock.
    with nullcontext():
        app = await test_case.start(workspace)
        try:
            session = await app.open_session("stock-serializer-probe")
            if type(session.checkpointer) is not AsyncSqliteSaver:
                raise AssertionError("MIRA did not construct stock AsyncSqliteSaver")
            if type(session.checkpointer.serde) is not JsonPlusSerializer:
                raise AssertionError("MIRA did not construct stock JsonPlusSerializer")
            print("  PASS  Actual session uses stock saver + serializer", flush=True)

            if phase == "write":
                await session.prompt("first message")
                print("  PASS  Actual MIRA Act turn checkpointed", flush=True)
                await session.set_mode("plan")
                await session.prompt("plan something")
                print("  PASS  Actual MIRA Plan turn checkpointed", flush=True)

                plan = plan_artifact(
                    plan_id="probe-plan-1",
                    title="Serialization probe plan",
                    objective="Run serialization checks",
                    context_and_constraints="Isolated fake-model probe only",
                    key_changes=["Check default serializer"],
                    test_plan=["Reopen checkpoint in another process"],
                    assumptions=["No external side effects"],
                    success_criteria="Both processes read correct messages",
                    rubric_enabled=True,
                    rubric_iterations=2,
                )
                replace_current_plan(session.record, plan)
                await session.persist_graph_state()
                app.store.save(session.record)
                started = await session.start_artifact("plan")
                assert started is not None and started["attempts"] == 1
                print("  PASS  Retained Plan + lifecycle checkpointed", flush=True)
                db = app.store.database_path(session.id)
                assert db.is_file(), db
                print(f"  DB    {db}", flush=True)

            elif phase == "read":
                act_state = await session.agent.aget_state(
                    {"configurable": {"thread_id": session.id}}
                )
                messages = act_state.values.get("messages", ())
                assert any(isinstance(x, HumanMessage) and x.content == "first message" for x in messages), (
                    "Act checkpoint lost the original HumanMessage"
                )
                plan_state = await session.plan_agent.aget_state(
                    {"configurable": {"thread_id": f"{session.id}:plan"}}
                )
                assert plan_state.values.get("messages"), "Plan checkpoint missing messages"
                assert session.record["current_plan"] is not None, "Retained Plan missing"
                assert session.record["current_plan"]["id"] == "probe-plan-1"
                assert session.record["current_plan"]["status"] == "paused", (
                    "Active Plan should be paused on reopened completed graph"
                )
                assert session.record["current_plan"]["attempts"] == 1
                print("  PASS  Cold process restored Act + Plan + retained Plan", flush=True)

                await session.set_mode("act")
                await session.prompt("second message")
                continued = await session.agent.aget_state(
                    {"configurable": {"thread_id": session.id}}
                )
                humans = [x.content for x in continued.values.get("messages", ()) if isinstance(x, HumanMessage)]
                assert humans.count("first message") == 1, humans
                assert humans.count("second message") == 1, humans
                print("  PASS  New turn continues native state with no duplicate history", flush=True)
            else:
                raise ValueError(f"Unknown worker phase: {phase}")
        finally:
            await test_case.stop(app)


def run_subprocess(repo: Path, workspace: Path, phase: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo", str(repo),
        "--workspace", str(workspace),
        "--phase", phase,
    ]
    print(f"\n=== C. MIRA {phase.upper()} (FRESH PYTHON PROCESS) ===", flush=True)
    result = subprocess.run(cmd, cwd=repo, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"MIRA {phase} subprocess failed, exit={result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="MIRA repository root")
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("write", "read"), help=argparse.SUPPRESS)
    parser.add_argument("--direct-only", action="store_true", help="Only test serializer values, no MIRA app")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    boot_repo(repo)

    if args.phase:
        if args.workspace is None:
            parser.error("--workspace required for internal worker")
        asyncio.run(run_mira_phase(repo, args.workspace.resolve(), args.phase))
        return 0

    print("MIRA STOCK SERIALIZER PROBE (no source edits / no external model calls)", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Repo:   {repo}", flush=True)
    run_typed_value_checks()
    if args.direct_only:
        print("\nDirect serialization checks complete; MIRA integration NOT tested.", flush=True)
        return 0

    with tempfile.TemporaryDirectory(prefix="mira_stock_serde_probe_") as temp:
        workspace = Path(temp) / "disposable_workspace"
        workspace.mkdir()
        run_subprocess(repo, workspace, "write")
        run_subprocess(repo, workspace, "read")

    print("\nPASS: stock serializer handled this MIRA smoke path across 2 processes.", flush=True)
    print("LIMIT: not proof for all subagent, HITL, rubric evaluation or /compact paths.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        print("\nFAIL: see traceback. No MIRA source or existing session was modified.", file=sys.stderr)
        raise SystemExit(1)
