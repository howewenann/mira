"""MIRA stock serializer integration/cold-restart probe.

Read-only to the MIRA checkout and existing sessions: all databases and MIRA
session files are created under a temporary *external* disposable workspace.
This script does NOT call an external model; it uses deterministic fake models.

Examples (inside the MIRA conda environment):
    python probe_mira_stock_serde_full.py --repo D:\\Projects\\mira
    python probe_mira_stock_serde_full.py --repo . --keep
    python probe_mira_stock_serde_full.py --repo . --quick

The default test runs both actual real-graph cold restart scenarios in separate
Python processes and selected existing integration regressions with the stock
AsyncSqliteSaver, JsonPlusSerializer, and InMemorySaver.

Limit: no finite probe can prove *all* future graph states/model outputs safe.
A PASS is evidence for the executed paths, NOT a guarantee. Intentional failures
for unsupported Python classes and Pydantic serializer internals are recorded
as known limitations, not as failures of legitimate graph-state paths.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest
import warnings
from unittest.mock import patch

PROBE_ID = "stock-serde-integration"

# These are real MIRA tests, not substitutes for graph execution. Selected
# cases intentionally avoid tests whose purpose is to assert Mira's *custom*
# marker-based fallback, which would be inapplicable to stock serialization.
INTEGRATION_TESTS = [
    "tests.test_sqlite_sessions.SqliteSessionTests.test_cold_restart_restores_messages_and_session_state",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_two_sessions_use_distinct_databases",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_native_rubric_channels_survive_cold_restart",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_formal_artifact_lifecycle_survives_restart",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_saved_tool_approval_is_redisplayed_before_native_resume",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_application_redisplays_saved_approval_after_cold_restart",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_saved_ask_user_interrupt_uses_native_resume",
    "tests.test_sqlite_sessions.SqliteSessionTests.test_incomplete_non_hitl_checkpoint_requires_explicit_continuation",
    "tests.test_checkpoint.CheckpointHitlTests.test_native_hitl_resume_preserves_ai_tool_and_final_messages",
    "tests.test_compaction_native.NativeCompactionTests.test_manual_compaction_on_mira_graph_survives_reload_and_repeats",
    "tests.test_compaction_native.NativeCompactionTests.test_automatic_and_tool_paths_append_to_real_checkpoint_archive",
    "tests.test_compaction_native.NativeCompactionTests.test_checkpoint_write_failure_rolls_back_archive",
    "tests.test_rubric_middleware.RubricMiddlewareTests.test_real_agents_leave_verifier_tool_choice_unforced_and_outer_state_private",
    "tests.test_rubric_middleware.RubricMiddlewareTests.test_verifier_uses_existing_hitl_and_resumes_without_grader_error",
    "tests.test_rubric_middleware.RubricMiddlewareTests.test_no_verifier_tool_call_still_grades_transcript_only_story",
    "tests.test_subagent_compilation.SubagentCompilationTests.test_deepagents_native_fork_inherits_messages_while_default_isolated_does_not",
    "tests.test_subagent_compilation.SubagentCompilationTests.test_deepagents_invokes_compiled_worker_without_schema",
    "tests.test_subagent_persistence.PersistentSubagentRunsTests.test_native_standalone_consumer_projects_through_observer",
    "tests.test_subagent_persistence.PersistentSubagentRunsTests.test_hitl_reattach_keeps_one_durable_run",
    "tests.test_workflows.WorkflowAPITests.test_application_exposes_facade_and_native_graph_context",
    "tests.test_workflows.WorkflowAPITests.test_specializations_are_independent_and_validate_overrides",
    "tests.test_plan_mode.PlanModeTests.test_explicit_goal_request_uses_plan_agent_without_persistent_plan_mode",
    "tests.test_plan_mode.PlanModeTests.test_goal_executes_only_in_explicit_attempt_scope",
]


def setup_repo(repo: Path) -> None:
    repo = repo.resolve()
    if not (repo / "session" / "checkpoint.py").is_file():
        raise RuntimeError(f"Not a MIRA checkout: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)


@contextlib.contextmanager
def stock_persistence():
    """Keep the existing probe phases grouped under stock persistence."""
    yield


def check_stock(session: object) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    cp = session.checkpointer
    assert type(cp) is AsyncSqliteSaver, f"Nonstock saver: {type(cp)}"
    assert type(cp.serde) is JsonPlusSerializer, f"Nonstock serializer: {type(cp.serde)}"


def direct_checks() -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.serde.types import _DeltaSnapshot
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from pydantic import BaseModel
    from pydantic_core import SchemaSerializer, SchemaValidator
    from tests.test_checkpoint import Finding
    serde = JsonPlusSerializer()
    supported = [
        ("dict", {"foo": [1, 2], "bar": "text"}),
        ("message", AIMessage(content="hi")),
        ("delta snapshot", _DeltaSnapshot([HumanMessage(content="ask"), AIMessage(content="answer"), ToolMessage(content="result", tool_call_id="t1")])),
        ("Pydantic model", Finding(summary="ok")),
    ]
    print("\n[DIRECT SERIALIZER ROUND-TRIPS]", flush=True)
    for label, value in supported:
        kind, data = serde.dumps_typed(value)
        restored = serde.loads_typed((kind, data))
        if label == "delta snapshot":
            assert isinstance(restored, _DeltaSnapshot) and [m.content for m in restored.value] == ["ask", "answer", "result"]
        elif label == "Pydantic model":
            assert getattr(restored, "summary", None) == "ok"
        elif label == "message":
            assert isinstance(restored, AIMessage) and restored.content == "hi"
        else:
            assert restored == value
        print(f" PASS {label}: {kind} -> {type(restored).__name__}", flush=True)
    unusual = [
        ("Python class", AIMessage),
        ("Pydantic MockValSer", BaseModel.__pydantic_serializer__),
        ("Pydantic SchemaSerializer", SchemaSerializer({"type": "any"})),
        ("Pydantic SchemaValidator", SchemaValidator({"type": "any"})),
    ]
    print("[HISTORICAL UNSUPPORTED VALUES — informational, expected to fail]", flush=True)
    for label, value in unusual:
        try:
            serde.dumps_typed({"value": value})
        except (TypeError, ValueError) as e:
            print(f" EXPECTED {label}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        else:
            print(f" NOTE {label}: now supported by stock serializer", flush=True)


async def core_phase(workspace: Path, phase: str) -> None:
    from langchain_core.messages import HumanMessage
    from session.goals import goal_artifact, replace_current_goal
    from session.plans import plan_artifact, replace_current_plan
    from tests.test_sqlite_sessions import SqliteSessionTests
    t = SqliteSessionTests("test_cold_restart_restores_messages_and_session_state")
    app = await t.start(workspace, rubric=True)
    try:
        s = await app.open_session(PROBE_ID)
        check_stock(s)
        db = app.store.database_path(s.id)
        assert db.parent.is_dir(), db
        if phase == "write":
            await s.prompt("SERDE_ACT_FIRST")
            await s.set_mode("plan")
            await s.prompt("SERDE_PLAN_FIRST")
            plan = plan_artifact(
                plan_id="probe-plan", title="Stock plan", objective="Safe checks",
                context_and_constraints="Disposable workspace", key_changes=["check"],
                test_plan=["restart"], assumptions=["fake only"],
                success_criteria="Successful restart", rubric_enabled=True, rubric_iterations=2,
            )
            replace_current_plan(s.record, plan)
            await s.persist_graph_state()
            app.store.save(s.record)
            await s.start_artifact("plan")
            await s.agent.aupdate_state({"configurable": {"thread_id": s.id}}, {
                "_rubric_status": "max_iterations_reached", "_rubric_iterations": 2,
                "_rubric_criteria": ["probe criterion"]
            })
            # Separate Goal session: Plan and Goal are exclusive.
            g = await app.open_session(PROBE_ID + "-goal")
            check_stock(g)
            await g.begin_goal("SERDE_GOAL_OBJECTIVE")
            goal = goal_artifact(
                goal_id="probe-goal", title="Stock goal", objective="Check goal",
                success_criteria="Goal restored", rubric_enabled=False, rubric_iterations=3,
            )
            replace_current_goal(g.record, goal)
            await g.persist_graph_state()
            app.store.save(g.record)
            print(f" PASS wrote MIRA Act, Plan, Goal, native rubric channels: {db}", flush=True)
        else:
            assert s.mode["planning"] is False or s.mode["planning"] is True
            act = await s.agent.aget_state({"configurable": {"thread_id": s.id}})
            plan = await s.plan_agent.aget_state({"configurable": {"thread_id": f"{s.id}:plan"}})
            assert any(isinstance(m, HumanMessage) and m.content == "SERDE_ACT_FIRST" for m in act.values["messages"])
            assert plan.values["messages"]
            assert s.record["current_plan"]["id"] == "probe-plan"
            assert s.record["current_plan"]["status"] == "paused"
            assert s.record["current_plan"]["attempts"] == 1
            assert act.values["_rubric_iterations"] == 2
            assert act.values["_rubric_criteria"] == ["probe criterion"]
            g = await app.open_session(PROBE_ID + "-goal", resume=True)
            check_stock(g)
            assert g.record["current_goal"]["id"] == "probe-goal"
            assert g.mode["goal_staging"]["thread_id"].startswith(g.id + ":plan:")
            await s.set_mode("act")
            await s.prompt("SERDE_ACT_SECOND")
            messages = (await s.agent.aget_state({"configurable": {"thread_id": s.id}})).values["messages"]
            humans = [m.content for m in messages if isinstance(m, HumanMessage)]
            assert humans.count("SERDE_ACT_FIRST") == 1, humans
            assert humans.count("SERDE_ACT_SECOND") == 1, humans
            print(" PASS cold restart restored Act, Plan, Goal, rubric state, new turn without duplicate history", flush=True)
    finally:
        await t.stop(app)


async def hitl_phase(workspace: Path, phase: str) -> None:
    """Real LangGraph HITL, handled via MIRA's own runner after cold restart."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from tests.test_checkpoint import BindableFakeMessagesListChatModel
    from tests.test_runner import RunTurnRenderer
    from core.execution.runner import run_turn

    effects = workspace / "approved_tool_effects.txt"
    db = workspace / "hitl.db"
    @tool
    def guarded_write(value: str) -> str:
        """Append an approved marker to a disposable file."""
        with effects.open("a", encoding="utf-8") as fd:
            fd.write(value + "\n")
        return "written"

    conn = await aiosqlite.connect(str(db))
    saver = AsyncSqliteSaver(conn)
    try:
        for thread in ("approved", "rejected") if phase == "write" else ("rejected", "approved"):
            config = {"configurable": {"thread_id": thread}}
            if phase == "write":
                model = BindableFakeMessagesListChatModel(responses=[AIMessage(content="", tool_calls=[{
                    "name": "guarded_write", "args": {"value": thread}, "id": f"call-{thread}", "type": "tool_call"
                }])])
            else:
                model = BindableFakeMessagesListChatModel(responses=[AIMessage(content="finished")])
            agent = create_agent(
                model=model, tools=[guarded_write],
                middleware=[HumanInTheLoopMiddleware(interrupt_on={"guarded_write": True})],
                checkpointer=saver,
            )
            if phase == "write":
                stream = await agent.astream_events({"messages": [{"role": "user", "content": "perform an approved write"}]}, config=config, version="v3")
                await stream.output()
                assert len(await stream.interrupts()) == 1
                assert not effects.exists(), "Tool executed before approval"
            else:
                saved = await agent.aget_state(config)
                intr = [it for task in saved.tasks for it in task.interrupts]
                assert len(intr) == 1, (thread, saved.next)
                renderer = RunTurnRenderer(decisions=[{"type": "reject", "message": "no"}] if thread == "rejected" else [{"type": "approve"}])
                await run_turn(agent, "", renderer, thread, resume_interrupts=intr)
                assert len(renderer.approvals) == 1
                assert not (await agent.aget_state(config)).next
        if phase == "read":
            assert effects.read_text(encoding="utf-8").splitlines() == ["approved"]
        print(f" PASS HITL {phase}: checkpointed approval, rejection, one permitted side effect", flush=True)
    finally:
        await conn.close()


async def subagent_phase(workspace: Path, phase: str) -> None:
    """Run an actual DeepAgents 'task' subagent, checkpoint, reopen, continue."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda
    from deepagents import create_deep_agent
    from tests.test_checkpoint import BindableFakeMessagesListChatModel
    db = workspace / "subagent.db"
    config = {"configurable": {"thread_id": "subagent-parent"}}
    counter = workspace / "subagent_effects.txt"

    def worker_fn(state):
        with counter.open("a", encoding="utf-8") as f:
            f.write("child\n")
        return {"messages": [AIMessage(content="worker-result")]}

    worker = RunnableLambda(worker_fn)
    if phase == "write":
        model = BindableFakeMessagesListChatModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {"description": "Perform probe", "subagent_type": "probe-worker"},
                "id": "subagent-call", "type": "tool_call"
            }]),
            AIMessage(content="parent-finished"),
        ])
    else:
        model = BindableFakeMessagesListChatModel(responses=[AIMessage(content="parent-after-restart")])
    conn = await aiosqlite.connect(str(db))
    try:
        saver = AsyncSqliteSaver(conn)
        agent = create_deep_agent(
            model=model,
            subagents=[{"name": "probe-worker", "description": "Disposable fake worker", "runnable": worker}],
            checkpointer=saver,
        )
        if phase == "write":
            await agent.ainvoke({"messages": [HumanMessage(content="Delegate a task")]}, config=config)
            assert counter.read_text(encoding="utf-8").splitlines() == ["child"]
            state = await agent.aget_state(config)
            assert any(isinstance(m, ToolMessage) and "worker-result" in str(m.content) for m in state.values["messages"]), str(state.values["messages"])
            print(" PASS actual DeepAgents task tool and compiled worker wrote checkpoints", flush=True)
        else:
            state = await agent.aget_state(config)
            assert any(isinstance(m, ToolMessage) for m in state.values["messages"])
            assert len(counter.read_text(encoding="utf-8").splitlines()) == 1
            await agent.ainvoke({"messages": [HumanMessage(content="Continue without delegation")]}, config=config)
            assert len(counter.read_text(encoding="utf-8").splitlines()) == 1
            print(" PASS subagent cold restart recovered parent tool transcript without rerunning child", flush=True)
    finally:
        await conn.close()


async def compaction_phase(workspace: Path, phase: str) -> None:
    """Real MIRA compaction middleware, not a simplified mock; durable SQL."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langchain_core.messages import AIMessage, HumanMessage
    from tests.test_compaction_native import BindableModel
    from agent.factory import build_agent
    from core.execution.compaction import commit_forced_compaction

    db = workspace / "compaction.db"
    config = {"configurable": {"thread_id": "compaction-session"}}
    model = BindableModel(responses=[AIMessage(content="main reply")])
    model.profile = {"max_input_tokens": 5000}
    conn = await aiosqlite.connect(str(db))
    try:
        saver = AsyncSqliteSaver(conn)
        with patch("agent.factory.get_llm", return_value=model):
            agent = build_agent({}, workspace, saver)
        if phase == "write":
            for i in range(12):
                await agent.ainvoke({"messages": [HumanMessage(content=f"history {i}")]}, config)
            model.profile = {"max_input_tokens": 100}
            snap = await agent.aget_state(config)
            plan = await agent.mira_compaction.aplan_forced_compaction_update(snap.values)
            result = await commit_forced_compaction(agent, "compaction-session", snap, plan)
            assert result.compacted, result
            print(" PASS native forced /compact checkpoint and archive created", flush=True)
        else:
            snap = await agent.aget_state(config)
            state = snap.values
            assert state.get("_summarization_event"), "missing compact event"
            assert state.get("_summarization_session_id"), "missing compaction session ID"
            archive_path = state["_summarization_event"]["file_path"]
            files = await agent.mira_backend.adownload_files([archive_path])
            assert files and b"Summarized at" in files[0].content
            assert not snap.next
            await agent.ainvoke({"messages": [HumanMessage(content="followup after compact")]}, config)
            result = await agent.aget_state(config)
            assert any(isinstance(m, HumanMessage) and m.content == "followup after compact" for m in result.values["messages"])
            print(" PASS cold restart loaded native compaction state/archive and continued", flush=True)
    finally:
        await conn.close()


async def rubric_phase(workspace: Path, phase: str) -> None:
    """Real MiraRubricMiddleware performing verifier + grader, then restart."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langchain_core.messages import AIMessage, HumanMessage
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    from tests.test_rubric_middleware import FixedFakeChatModel, grader_message
    from agent.rubric.middleware import MiraRubricMiddleware

    db = workspace / "rubric.db"
    config = {"configurable": {"thread_id": "rubric-session"}}
    @tool
    def inspect_probe(value: str) -> str:
        """Return deterministic verification evidence."""
        return f"verified:{value}"

    def grader_model():
        return FixedFakeChatModel(messages=iter([
            AIMessage(content="", tool_calls=[{"name": "inspect_probe", "args": {"value": "gold"}, "id": "verify-call", "type": "tool_call"}]),
            AIMessage(content="VERIFICATION_COMPLETE"),
            grader_message(),
        ]))

    conn = await aiosqlite.connect(str(db))
    try:
        saver = AsyncSqliteSaver(conn)
        model = FixedFakeChatModel(messages=iter([AIMessage(content="main-answer")]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = create_deep_agent(
                model=model,
                middleware=[MiraRubricMiddleware(model=grader_model(), verifier_tools=[inspect_probe])],
                checkpointer=saver,
            )
        if phase == "write":
            await agent.ainvoke({"messages": [HumanMessage(content="verify output")], "rubric": "check evidence"}, config=config)
            state = (await agent.aget_state(config)).values
            assert state["_rubric_status"] == "satisfied", state.get("_rubric_status")
            assert state["_rubric_iterations"] >= 1
            print(" PASS real rubric verifier, grader, status and criteria checkpointed", flush=True)
        else:
            state = (await agent.aget_state(config)).values
            assert state["_rubric_status"] == "satisfied", state.get("_rubric_status")
            assert state["_rubric_iterations"] >= 1
            assert any(isinstance(m, HumanMessage) and m.content == "verify output" for m in state["messages"])
            print(" PASS cold restart loaded real rubric evaluated state", flush=True)
    finally:
        await conn.close()


def integration_suite(quick: bool) -> None:
    tests = INTEGRATION_TESTS[:10] if quick else INTEGRATION_TESTS
    print(f"\n[EXISTING MIRA INTEGRATION TESTS: {len(tests)} using stock serializer]", flush=True)
    with stock_persistence():
        suite = unittest.defaultTestLoader.loadTestsFromNames(tests)
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    if not result.wasSuccessful() or result.testsRun != len(tests):
        raise AssertionError(f"Regression suite: {result.testsRun} run, {len(result.failures)} failures, {len(result.errors)} errors, {len(result.skipped)} skipped")


async def phase_worker(workspace: Path, scenario: str, phase: str) -> None:
    mapping = {"core": core_phase, "hitl": hitl_phase, "subagent": subagent_phase,
               "compaction": compaction_phase, "rubric": rubric_phase}
    with stock_persistence():
        await mapping[scenario](workspace, phase)


def run_worker(repo: Path, workspace: Path, scenario: str, phase: str, timeout: int) -> bool:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--repo", str(repo),
           "--workspace", str(workspace), "--scenario", scenario, "--phase", phase]
    print(f"\n[SUBPROCESS {scenario.upper()} — {phase.upper()}]", flush=True)
    try:
        result = subprocess.run(cmd, cwd=repo, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f" FAIL {scenario}/{phase}: timeout ({timeout}s)", flush=True)
        return False
    if result.returncode != 0:
        print(f" FAIL {scenario}/{phase}: exit={result.returncode}", flush=True)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="MIRA repository checkout")
    parser.add_argument("--quick", action="store_true", help="Smaller existing-test selection, same process-restart scenarios")
    parser.add_argument("--keep", action="store_true", help="Keep disposable workspace in system temporary directory for inspection")
    parser.add_argument("--timeout", type=int, default=150, help="Timeout seconds for EACH subprocess")
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=["core", "hitl", "subagent", "compaction", "rubric", "tests"], help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=["write", "read", "run"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    setup_repo(repo)
    if args.scenario:
        if args.scenario == "tests":
            integration_suite(args.quick)
        else:
            if args.workspace is None or args.phase not in ("write", "read"):
                parser.error("internal --scenario needs --workspace and --phase write/read")
            asyncio.run(phase_worker(args.workspace.resolve(), args.scenario, args.phase))
        return 0

    print("MIRA COMPREHENSIVE STOCK SERIALIZER PROBE", flush=True)
    print(f"Python: {sys.executable}\nRepo: {repo}", flush=True)
    direct_checks()
    path = Path(tempfile.mkdtemp(prefix="mira_stock_serde_full_"))
    scenarios = ["core", "hitl", "subagent", "rubric", "compaction"]
    results = {}
    try:
        workspace = path / "disposable_workspace"
        workspace.mkdir(parents=True)
        for scenario in scenarios:
            results[scenario] = run_worker(repo, workspace, scenario, "write", args.timeout)
            if results[scenario]:
                results[scenario] = run_worker(repo, workspace, scenario, "read", args.timeout)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--repo", str(repo), "--scenario", "tests", "--phase", "run"]
        if args.quick:
            cmd.append("--quick")
        print("\n[SEPARATE PROCESS — EXISTING REGRESSION TESTS]", flush=True)
        try:
            result = subprocess.run(cmd, cwd=repo, check=False, timeout=args.timeout * (3 if args.quick else 8))
            results["existing MIRA integration tests"] = result.returncode == 0
        except subprocess.TimeoutExpired:
            results["existing MIRA integration tests"] = False
            print(" FAIL regression tests: timeout", flush=True)
        print("\n=== RESULTS ===", flush=True)
        for key, passed in results.items():
            print(f" {'PASS' if passed else 'FAIL'} {key}", flush=True)
        passed = sum(results.values())
        print(f"{passed}/{len(results)} categories passed", flush=True)
        print("Note: historical raw Python classes/Pydantic internals remain unsupported by stock serializer.", flush=True)
        print("Note: this does not cover arbitrary external model outputs or untested plugins.", flush=True)
        return 0 if passed == len(results) else 1
    finally:
        if args.keep:
            print(f"Disposable workspace kept at: {path}", flush=True)
        else:
            import shutil
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        print("FAILED. MIRA source files and existing sessions were NOT edited.", file=sys.stderr)
        raise SystemExit(1)
