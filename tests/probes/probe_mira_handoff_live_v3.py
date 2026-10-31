#!/usr/bin/env python
"""EXTENDED live-LLM diagnostic for MIRA Act, Plan, and Goal context handoff.

Usage from the MIRA repository, in its normal Python/Conda environment:
    python path/to/probe_mira_handoff_live.py

This is a candidate *probe*, not a modification to MIRA's production handoff.
It uses real MIRA Act and Plan compiled agents and the Main/Summarization models
selected by .mira/settings.yml, but creates all workspace files, archives,
and SQLite checkpoints inside a temporary directory. No MIRA session is opened.

It performs live model responses in Act and Plan, native forced summaries using
MIRA's compaction implementation, and real SQLite message-replacement checks.
Synthetic older messages are appended (to the scratch DB only). During the two
forced compactions, the native summarizer's keep policy is temporarily set to
8 messages; this ensures a short test conversation is eligible for compaction
without changing the real workspace's model profile or application settings.
The normal MIRA compaction planner, archive writer, and summary LLM are used.

IMPORTANT: This sends diagnostic text to the configured LLM provider and can
incur model usage/cost. It does NOT test the full Goal review UI and interruption
orchestration; Goal artifact lifecycle and native HITL are checked separately.
Real source workspace is READ ONLY. All checkpoint/archival changes are in tmp.
"""

from __future__ import annotations

import asyncio
import copy
import secrets
import sys
import tempfile
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Place in MIRA's tests/probes directory, or pass --repo /path/to/mira.
import argparse


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None, help="MIRA repository root (contains agent/ and config/)")
    parser.add_argument("--keep", action="store_true", help="Keep isolated temp workspace after probe for inspection")
    return parser.parse_args()


def locate_repo(root: Path | None) -> Path:
    if root is not None:
        resolved = root.expanduser().resolve()
    else:
        location = Path(__file__).resolve()
        candidates = [Path.cwd().resolve(), *location.parents]
        resolved = next((p for p in candidates if (p / "agent" / "factory.py").is_file()), Path.cwd().resolve())
    if not (resolved / "agent" / "factory.py").is_file():
        raise RuntimeError("MIRA repository not found; run from its root or supply --repo")
    return resolved


def conf(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def require(test: bool, description: str) -> None:
    if not test:
        raise AssertionError(description)
    print(f"  PASS {description}", flush=True)


def plain(value: Any) -> str:
    # LangChain's message.text is a property on current versions, callable on some older ones.
    # Prefer .content: newer LangChain versions deprecate calling .text().
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    field = getattr(value, "text", "")
    return field if isinstance(field, str) else str(content or "")


def signature(messages: list[Any]) -> list[tuple[str, str]]:
    return [(type(m).__name__, plain(m)) for m in messages]


def effective(state: dict[str, Any]) -> list[Any]:
    # Use MIRA's own middleware (including its provider-safety wrapper).
    from agent.middleware.compaction import normalize_summarization_event
    from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware

    return _DeepAgentsSummarizationMiddleware._apply_event_to_messages(
        list(state.get("messages") or []),
        normalize_summarization_event(state.get("_summarization_event")),
    )


async def snapshot(agent: Any, tid: str) -> Any:
    state = await agent.aget_state(conf(tid))
    if state.next or state.tasks:
        raise RuntimeError(
            f"Unsafe handoff: {tid} has pending work: next={state.next!r}, "
            f"tasks={len(state.tasks)}"
        )
    return state


async def replace_conversation(source: Any, source_id: str, target: Any, target_id: str) -> None:
    """EXPERIMENTAL candidate only; never call it on an existing MIRA session."""
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    src = await snapshot(source, source_id)
    dst = await snapshot(target, target_id)
    if dst.config is None:
        raise AssertionError(f"Target {target_id} was not initialized")
    inherited = effective(src.values)
    source_before = signature(list(src.values.get("messages") or []))
    source_event_before = repr(src.values.get("_summarization_event"))
    if isinstance(src.values.get("_summarization_event"), dict):
        history_path = src.values["_summarization_event"].get("file_path")
        if history_path and hasattr(target, "mira_backend"):
            archive = await target.mira_backend.adownload_files([history_path])
            require(bool(archive) and archive[0].error is None and bool(archive[0].content),
                    f"{target_id}: source compaction archive accessible across handoff")
    await target.aupdate_state(
        dst.config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *inherited],
         "_summarization_event": None},
    )
    now = await snapshot(target, target_id)
    require(
        signature(effective(now.values)) == signature(inherited),
        f"{source_id} -> {target_id}: effective conversation replaced exactly",
    )
    require(
        now.values.get("_summarization_event") is None,
        f"{target_id}: obsolete compaction event cleared",
    )
    if "todos" in dst.values:
        require(now.values.get("todos") == dst.values["todos"], f"{target_id}: original todos preserved")
    after_source = await snapshot(source, source_id)
    require(signature(list(after_source.values.get("messages") or [])) == source_before
            and repr(after_source.values.get("_summarization_event")) == source_event_before,
            f"{source_id}: source checkpoint is unmodified by handoff")


async def ask(agent: Any, thread_id: str, question: str, *, initial_history: list[Any] | None = None) -> str:
    from langchain_core.messages import HumanMessage

    payload = [*(initial_history or []), HumanMessage(content=question)]
    result = await agent.ainvoke({"messages": payload}, conf(thread_id))
    messages = result.get("messages") or []
    response = plain(messages[-1]) if messages else ""
    print(f"  LLM ({thread_id}): {response[:420]!r}", flush=True)
    require(bool(response.strip()), f"{thread_id}: received a real LLM answer")
    await snapshot(agent, thread_id)
    return response


async def append_old_history(agent: Any, thread_id: str, topic: str) -> None:
    """Make long *scratch* history so the real compaction engine can summarize."""
    from langchain_core.messages import AIMessage, HumanMessage

    previous = await snapshot(agent, thread_id)
    existing = len(previous.values.get("messages") or [])
    # Keep these short; the point is to make native message-count retention fire.
    # They are labelled as synthetic so a model should not treat them as user orders.
    additions = []
    for n in range(24):
        label = f"[Historical diagnostic note {topic} {n}]"
        if n % 2 == 0:
            additions.append(HumanMessage(content=f"{label} This is synthetic test history, not a new instruction."))
        else:
            additions.append(AIMessage(content=f"{label} Acknowledged for checkpoint testing."))
    await agent.aupdate_state(previous.config, {"messages": additions})
    now = await snapshot(agent, thread_id)
    require(len(now.values.get("messages") or []) >= existing + 24, f"{thread_id}: added scratch history for native compaction")


async def real_compact(agent: Any, thread_id: str) -> None:
    """Invoke MIRA's genuine summary LLM using a short probe-only keep window.

    Production DeepAgents selects a fractional retention window when the model
    advertises a context size (typically 10%). Our 24 brief diagnostic notes
    will fit inside that window, so forced compaction correctly returns None.

    ONLY for this temporary probe, set the *native* LangChain summarization
    helper's keep policy to 8 messages. No MIRA source or model config is
    changed. The real MIRA compaction planner, archiver, SQLite commit path,
    and configured summary LLM still run. Restore the policy in finally.
    """
    from core.execution.compaction import commit_forced_compaction

    state = await snapshot(agent, thread_id)
    summarization = agent.mira_compaction._summarization
    helper = summarization._lc_helper
    original_keep = helper.keep
    print(
        f"  Native retention: {original_keep!r}; "
        f"messages={len(state.values.get('messages') or [])}; "
        f"original cutoff={summarization._determine_cutoff_index(effective(state.values))}",
        flush=True,
    )
    try:
        helper.keep = ("messages", 8)  # TEST-ONLY; restore in finally.
        cutoff = summarization._determine_cutoff_index(effective(state.values))
        print(f"  Probe retention: {helper.keep!r}; cutoff={cutoff}", flush=True)
        require(cutoff > 0, f"{thread_id}: probe compaction has a valid cutoff")
        plan = await agent.mira_compaction.aplan_forced_compaction_update(state.values)
        require(plan is not None, f"{thread_id}: native forced-compaction plan produced")
        result = await commit_forced_compaction(agent, thread_id, state, plan)
    finally:
        helper.keep = original_keep

    require(result.compacted, f"{thread_id}: native compaction committed (with live summary LLM)")
    after = await snapshot(agent, thread_id)
    require(after.values.get("_summarization_event") is not None,
            f"{thread_id}: native summary event saved")
    require(len(effective(after.values)) < len(after.values.get("messages") or []),
            f"{thread_id}: effective history shorter than raw checkpoint")
    print(f"  Native summary excerpt: {result.summary[:260]!r}", flush=True)


async def check_archive(agent: Any, thread_id: str) -> None:
    state = (await snapshot(agent, thread_id)).values
    event = state.get("_summarization_event")
    require(isinstance(event, dict) and bool(event.get("file_path")),
            f"{thread_id}: persisted DeepAgents archive reference")
    response = await agent.mira_backend.adownload_files([event["file_path"]])
    require(bool(response) and response[0].error is None and bool(response[0].content),
            f"{thread_id}: native archive can be retrieved")


async def isolated_goal_probes(act, plan, session, marker: str) -> None:
    """Check the real mode/Goal semantic APIs and independent staging threads.

    The candidate message handoff is applied explicitly. Actual finalizer
    interrupt/front-end approval remains a separate untested integration path.
    """
    from session.goals import goal_artifact, replace_current_goal

    act_id = session.id
    plan_id = f"{session.id}:plan"

    print("\n[6] Goal cancellation: isolate context and preserve origin", flush=True)
    await session.set_mode("plan")
    plan_before = signature((await snapshot(plan, plan_id)).values["messages"])
    act_before = signature((await snapshot(act, act_id)).values["messages"])
    await session.begin_goal(f"Cancel test for {marker}")
    goal_cancel_id = session.mode["goal_staging"]["thread_id"]
    require(goal_cancel_id not in {act_id, plan_id}, "Goal cancellation uses unique temporary planning thread")
    answer = await ask(
        plan, goal_cancel_id,
        "Read our inherited conversation. What is its project codename? "
        "Answer only the codename. No tools or formal Plan.",
        initial_history=effective((await snapshot(plan, plan_id)).values),
    )
    require(marker.lower() in answer.lower(), "Goal staging LLM inherits Plan's context")
    require(signature((await snapshot(plan, plan_id)).values["messages"]) == plan_before,
            "Cancelled Goal did not mutate originating Plan conversation")
    require(signature((await snapshot(act, act_id)).values["messages"]) == act_before,
            "Cancelled Goal did not mutate parked Act conversation")
    session.mode["goal_staging"] = None  # Simulate decline, without production UI.
    await session.persist_graph_state()
    require(session.mode["goal_staging"] is None, "Declined Goal clears semantic staging pointer")

    print("\n[7] Goal approval: native Goal metadata + candidate handoff", flush=True)
    await session.set_mode("act")
    await session.begin_goal(f"Confirm project codename {marker}")
    approved_id = session.mode["goal_staging"]["thread_id"]
    require(approved_id not in {act_id, plan_id, goal_cancel_id},
            "New Goal staging has its own thread")
    act_before = signature((await snapshot(act, act_id)).values["messages"])
    answer = await ask(
        plan, approved_id,
        "From the inherited discussion, what is the project codename? "
        "Answer only the codename; do not prepare a Goal or use tools.",
        initial_history=effective((await snapshot(act, act_id)).values),
    )
    require(marker.lower() in answer.lower(), "Goal staging LLM inherits Act's context")
    require(signature((await snapshot(act, act_id)).values["messages"]) == act_before,
            "Goal staging did not alter parked Act")
    await append_old_history(plan, approved_id, "GOAL")
    await real_compact(plan, approved_id)
    await check_archive(plan, approved_id)

    # Construct an approved goal using MIRA's actual goal artifact/session APIs.
    # Deliberately DO NOT claim the real finalize_goal HITL UI was exercised.
    artifact = goal_artifact(
        goal_id="probe-goal-approved", title="Verify project memory",
        objective=f"Preserve the project codename {marker}",
        success_criteria=f"MIRA can recall {marker}",
        rubric_enabled=False, rubric_iterations=1,
    )
    replace_current_goal(session.record, artifact)
    session.mode["goal_staging"] = None
    await session.persist_graph_state()
    await replace_conversation(plan, approved_id, act, act_id)
    started = await session.start_artifact("goal")
    require(bool(started) and started.get("status") == "active",
            "MIRA's actual start_artifact('goal') activates approved Goal")
    require(session.mode["executing_goal"] and not session.mode["planning"],
            "Approved Goal execution switches to Act")
    answer = await ask(
        act, act_id,
        "What is the project codename discussed in Goal staging? "
        "Reply with the exact code only, no tools.",
    )
    require(marker.lower() in answer.lower(),
            "Live Act recalls context after compacted Goal handoff")
    return approved_id


def build_goal_finalizer_graph(saver):
    """Native checkpoint graph executing MIRA's actual finalize_goal tool function.

    This deliberately avoids dependence on a model deciding to call the tool;
    the full production frontend/formal-review orchestrator is not involved.
    """
    from langgraph.graph import START, StateGraph
    from langchain.tools import ToolRuntime
    from typing import TypedDict
    from agent.resources.defaults.tools.finalize_goal import finalize_goal

    class GoalReviewState(TypedDict, total=False):
        objective: str
        success_criteria: str
        decision: str

    def finalize(state: GoalReviewState) -> GoalReviewState:
        runtime = ToolRuntime(
            state={
                "planning_objective": state.get("objective", "Probe objective"),
                "planning_success_criteria": state.get("success_criteria", "Probe criteria"),
            },
            context=None, config={}, stream_writer=lambda _event: None,
            tool_call_id="probe-goal-finalizer", store=None,
        )
        decision = finalize_goal.func(title="Probe Goal", runtime=runtime)
        return {"decision": str(decision)}

    builder = StateGraph(GoalReviewState)
    builder.add_node("finalize_goal", finalize)
    builder.add_edge(START, "finalize_goal")
    return builder.compile(checkpointer=saver)


async def pending_interrupt_probe(saver, thread_id: str) -> None:
    """Verify actual MIRA finalize_goal interrupt and checkpoint guard."""
    graph = build_goal_finalizer_graph(saver)
    await graph.ainvoke(
        {"objective": "Test safe Goal review", "success_criteria": "Review survives restart"},
        conf(thread_id),
    )
    pending = await graph.aget_state(conf(thread_id))
    require(bool(pending.next and pending.tasks),
            "MIRA's actual finalize_goal tool left a native HITL checkpoint pending")
    interruptions = [item for task in pending.tasks for item in task.interrupts]
    require(any(isinstance(item.value, dict) and item.value.get("type") == "finalize_goal"
                for item in interruptions),
            "Pending interrupt carries the real finalize_goal review payload")
    try:
        await snapshot(graph, thread_id)
    except RuntimeError as exc:
        require("Unsafe handoff" in str(exc),
                "Handoff guard refuses pending finalize_goal checkpoint")
    else:
        raise AssertionError("Unsafe handoff was permitted on interrupted Goal graph")


def rebuild_agents(config, tmp, saver, metadata):
    from agent.factory import build_agent, build_plan_agent
    from agent.resources import build_resources
    settings = config["settings"]
    action_resources = build_resources(
        tmp, settings=settings, config=config, create_examples=False, enable_execute=False,
    )
    plan_resources = build_resources(
        tmp, settings=settings, config=config, create_examples=False, enable_execute=False,
    )
    return (
        build_agent(config, tmp, saver, metadata=metadata, mcp_manager=None, resources=action_resources),
        build_plan_agent(config, tmp, saver, metadata=metadata, mcp_manager=None, resources=plan_resources),
    )


def create_semantic_session(config, tmp, store, saver, act, plan, session_id: str):
    """Real MiraSession + session graph, without live frontend orchestration."""
    from types import SimpleNamespace
    from core.application.session import MiraSession
    from core.interface.protocol import NullFrontend
    from session.graph_state import make_session_graph
    app = SimpleNamespace(
        config=config, store=store, workspace=tmp, _sessions={},
        agent=act, plan_agent=plan, frontend=NullFrontend(),
        model_name="probe-model", context_limit_tokens=32768,
        context_limit_source="probe", mcp_manager=None,
    )
    record = store.load(session_id, resume=True, workspace=tmp)
    session = MiraSession(
        app, record, agent=act, plan_agent=plan,
        semantic_graph=make_session_graph(saver),
        # The connection owner is managed by this probe, not MiraSession.close().
        checkpointer=None,
    )
    return session


async def work(repo: Path, tmp: Path) -> None:
    import aiosqlite
    from config.loader import load_config
    from config.settings import INBUILT_DANGEROUS_TOOLS, context_limit_tokens
    from config.metadata import ModelMetadata
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command
    from session.store import SessionStore

    config = dict(load_config(repo))  # Read ONLY the user's model profile.
    settings = copy.deepcopy(config["settings"])
    settings["tracing"]["enabled"] = False
    settings["system"]["rubric"]["enabled"] = False
    settings["system"]["dynamic_subagents"]["enabled"] = False
    settings["system"]["planning_todos"]["enabled"] = True
    for name in INBUILT_DANGEROUS_TOOLS:
        settings["hitl"]["tools"].setdefault(name, {})["enabled"] = False
    config["settings"] = settings
    config["workspace"] = str(tmp)
    config["session_dir"] = str(tmp / ".mira" / "_sessions")
    (tmp / ".mira" / "skills").mkdir(parents=True, exist_ok=True)

    metadata = ModelMetadata(context_tokens=context_limit_tokens(config), context_source="probe")
    db = tmp / "probe-checkpoints.db"
    store = SessionStore(Path(config["session_dir"]))
    session_id = "handoff-live"
    act_id, plan_id = session_id, f"{session_id}:plan"
    marker = "MIRA-LIVE-" + secrets.token_hex(4).upper()
    print(f"  Test canary: {marker}", flush=True)
    connection = await aiosqlite.connect(str(db))
    try:
        saver = AsyncSqliteSaver(connection)
        act, plan = rebuild_agents(config, tmp, saver, metadata)
        session = create_semantic_session(config, tmp, store, saver, act, plan, session_id)

        print("\n[1] LIVE Act conversation", flush=True)
        await ask(act, act_id, f"This is a memory test. Our project codename is {marker}. "
                  "Remember that exact codename. Reply with 'Acknowledged'. Do not call tools.")
        state_a = await snapshot(act, act_id)
        todo = [{"content": "Protect original Act todo state", "status": "pending"}]
        await act.aupdate_state(state_a.config, {"todos": todo})
        after_todo = (await snapshot(act, act_id)).values
        require(after_todo.get("todos") == todo, "Act's non-message todo state initialized")
        frozen_act = signature(after_todo.get("messages") or [])

        print("\n[2] LIVE Plan: first handoff from Act", flush=True)
        answer = await ask(plan, plan_id,
                           "What is the exact project codename from the previous conversation? "
                           "Answer with only the codename. This is a discussion, not a formal Plan. Do not call tools.",
                           initial_history=effective(after_todo))
        require(marker.lower() in answer.lower(), "Live Plan recalls Act-only marker")
        require(signature((await snapshot(act, act_id)).values["messages"]) == frozen_act,
                "Original Act context unmodified")
        state_p = await snapshot(plan, plan_id)
        plan_todo = [{"content": "Preserve independent Plan todo", "status": "pending"}]
        await plan.aupdate_state(state_p.config, {"todos": plan_todo})
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan's own todo state initialized")

        print("\n[3] LIVE Plan compaction -> Act", flush=True)
        await append_old_history(plan, plan_id, "PLAN")
        await real_compact(plan, plan_id)
        await check_archive(plan, plan_id)
        await replace_conversation(plan, plan_id, act, act_id)
        answer = await ask(act, act_id, "What is the project codename from earlier? "
                           "Give only the exact code; don't call tools.")
        require(marker.lower() in answer.lower(), "Act recalls marker after Plan compaction")
        require((await snapshot(act, act_id)).values.get("todos") == todo,
                "Act todo preserved after Plan compaction")

        print("\n[4] LIVE Act compaction -> Plan", flush=True)
        await append_old_history(act, act_id, "ACT")
        await real_compact(act, act_id)
        await check_archive(act, act_id)
        await replace_conversation(act, act_id, plan, plan_id)
        answer = await ask(plan, plan_id, "What was our project codename? Return it "
                           "verbatim, with no tool calls or formal Plan.")
        require(marker.lower() in answer.lower(), "Plan recalls marker after Act compaction")
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan's original todo preserved")

        print("\n[5] REPEATED compaction + handoff + real LLM", flush=True)
        await append_old_history(plan, plan_id, "PLAN-SECOND")
        await real_compact(plan, plan_id)
        await check_archive(plan, plan_id)
        # Do a true second native compaction on this SAME thread while the
        # first compaction event is still present (rather than clearing it).
        first_plan_event = (await snapshot(plan, plan_id)).values["_summarization_event"]
        first_plan_session_id = (await snapshot(plan, plan_id)).values.get("_summarization_session_id")
        await append_old_history(plan, plan_id, "PLAN-SECOND-CHAIN")
        await real_compact(plan, plan_id)
        await check_archive(plan, plan_id)
        chained = (await snapshot(plan, plan_id)).values
        require(chained["_summarization_event"]["cutoff_index"] > first_plan_event["cutoff_index"],
                "Second compaction on same thread advances native cutoff index")
        require(chained.get("_summarization_session_id") == first_plan_session_id,
                "Repeated compaction retains native archive identity")
        archive = await plan.mira_backend.adownload_files(
            [chained["_summarization_event"]["file_path"]]
        )
        require(archive[0].content.count(b"## Summarized at") >= 2,
                "Repeated compaction appends to the native archive")
        await replace_conversation(plan, plan_id, act, act_id)
        answer = await ask(act, act_id,
                           "Recall the exact original project codename, reply only that code, no tools.")
        require(marker.lower() in answer.lower(),
                "Act still recalls marker after chained Plan compaction")
        require((await snapshot(act, act_id)).values.get("todos") == todo,
                "Act state survives multiple handoffs")
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan state survives multiple handoffs")

        await isolated_goal_probes(act, plan, session, marker)
        session.mode["goal_staging"] = None
        await session.persist_graph_state()
        store.save(session.record)
        before_restart = {
            "act": signature(effective((await snapshot(act, act_id)).values)),
            "plan": signature(effective((await snapshot(plan, plan_id)).values)),
        }
        print("\n[8] NATIVE HITL and pending MIRA Act checkpoint", flush=True)
        hitl_id = f"{session_id}:interrupt-probe"
        await pending_interrupt_probe(saver, hitl_id)

        from langchain_core.messages import HumanMessage
        pending_act_id = f"{session_id}:pending-act"
        await act.aupdate_state(
            conf(pending_act_id),
            {"messages": [HumanMessage(content="Do not execute: pending checkpoint probe")]},
            as_node="__start__",
        )
        pending_act = await act.aget_state(conf(pending_act_id))
        require(bool(pending_act.next), "Real MIRA Act checkpoint has pending work")
        try:
            await replace_conversation(plan, plan_id, act, pending_act_id)
        except RuntimeError as exc:
            require("Unsafe handoff" in str(exc),
                    "Candidate handoff refuses a target Act checkpoint with pending work")
        else:
            raise AssertionError("Candidate overwrote a pending MIRA Act checkpoint")
    finally:
        await connection.close()

    print("\n[9] COLD RESTART: reopen SQLite + rebuild MIRA's actual agents", flush=True)
    connection = await aiosqlite.connect(str(db))
    try:
        saver = AsyncSqliteSaver(connection)
        act, plan = rebuild_agents(config, tmp, saver, metadata)
        restored = create_semantic_session(config, tmp, store, saver, act, plan, session_id)
        await restored.restore_graph_state()
        require(restored.record.get("current_goal", {}).get("id") == "probe-goal-approved",
                "MIRA authoritative Goal recovered from native session checkpoint")
        require(restored.record["current_goal"]["status"] == "paused",
                "MIRA pauses active Goal safely after cold restart")
        require(restored.mode["goal_staging"] is None,
                "Goal temporary staging pointer remains cleared after restart")
        require(signature(effective((await snapshot(act, act_id)).values)) == before_restart["act"],
                "Act effective conversation survives cold restart")
        require(signature(effective((await snapshot(plan, plan_id)).values)) == before_restart["plan"],
                "Plan effective conversation survives cold restart")
        require((await snapshot(act, act_id)).values.get("todos") == todo,
                "Act independent todo survives cold restart")
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan independent todo survives cold restart")
        answer = await ask(act, act_id,
                           "After restart, recall the exact project codename. Reply only the code. No tools.")
        require(marker.lower() in answer.lower(), "Act live LLM recalls marker after restart")
        await replace_conversation(act, act_id, plan, plan_id)
        answer = await ask(plan, plan_id,
                           "After restart, what is our original project codename? Just the code; no tools.")
        require(marker.lower() in answer.lower(), "Plan live LLM recalls marker after restart + handoff")
        await restored.clear_artifact("goal")
        require(restored.record.get("current_goal") is None,
                "MIRA clear_artifact('goal') removes active Goal from semantic state")

        print("\n[10] MIRA FINALIZE_GOAL native HITL recovery after cold restart", flush=True)
        pending_act = await act.aget_state(conf(pending_act_id))
        require(bool(pending_act.next),
                "Pending MIRA Act checkpoint survives restart without accidental execution")
        native_goal_hitl = build_goal_finalizer_graph(saver)
        state = await native_goal_hitl.aget_state(conf(hitl_id))
        require(bool(state.next and state.tasks),
                "MIRA finalize_goal interrupt survived SQLite restart")
        await native_goal_hitl.ainvoke(Command(resume="implement"), conf(hitl_id))
        state = await native_goal_hitl.aget_state(conf(hitl_id))
        require(not state.next and state.values.get("decision") == "implement",
                "MIRA finalize_goal native interrupt resumes safely after restart")

        print("\nPASS: ALL EXECUTED EXTENDED PROBE CHECKS PASSED.", flush=True)
        print("SCOPE: Real Act/Plan LLM, MIRA compaction, checkpoint restore, ",
              "Goal semantic APIs, candidate handoff, actual finalize_goal interrupt.", flush=True)
        print("NOT YET COVERED: Actual MIRA frontend approve/revise/close ",
              "interaction; concurrent in-flight application orchestration.", flush=True)
    finally:
        await connection.close()


def main():
    options = args()
    repo = locate_repo(options.repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print("MIRA LIVE context-handoff probe — REAL provider calls", flush=True)
    print(f"Source repository: {repo}", flush=True)
    for name in ("deepagents", "langgraph", "langchain-core"):
        with suppress(PackageNotFoundError):
            print(f"  {name} {version(name)}", flush=True)
    if options.keep:
        temp = Path(tempfile.mkdtemp(prefix="mira-live-handoff-"))
        print(f"Scratch directory (retained): {temp}", flush=True)
        asyncio.run(work(repo, temp))
    else:
        with tempfile.TemporaryDirectory(prefix="mira-live-handoff-") as path:
            print(f"Scratch directory: {path} (auto-cleanup)", flush=True)
            asyncio.run(work(repo, Path(path)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
