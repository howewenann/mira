#!/usr/bin/env python
"""LIVE-LLM diagnostic for MIRA's proposed Act <-> Plan handoff.

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
incur model usage/cost. It does NOT test the full Goal approval lifecycle.
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

async def work(repo: Path, tmp: Path) -> None:
    import aiosqlite
    from config.loader import load_config
    from config.settings import INBUILT_DANGEROUS_TOOLS, context_limit_tokens
    from config.metadata import ModelMetadata
    from agent.resources import build_resources
    from agent.factory import build_agent, build_plan_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    config = load_config(repo)  # Read user's *model profile*; don't write to repo.
    config = dict(config)
    settings = copy.deepcopy(config["settings"])
    settings["tracing"]["enabled"] = False
    settings["system"]["rubric"]["enabled"] = False
    settings["system"]["dynamic_subagents"]["enabled"] = False
    settings["system"]["planning_todos"]["enabled"] = True
    for name in INBUILT_DANGEROUS_TOOLS:
        # Even Act is sandboxed to a temporary workspace, without mutation tools.
        settings["hitl"]["tools"].setdefault(name, {})["enabled"] = False
    config["settings"] = settings
    config["workspace"] = str(tmp)
    config["session_dir"] = str(tmp / ".mira" / "_sessions")

    metadata = ModelMetadata(context_tokens=context_limit_tokens(config), context_source="probe")
    db = tmp / "probe-checkpoints.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(str(db))
    saver = AsyncSqliteSaver(connection)
    try:
        # Mirror the skills directory layout without modifying the real repo.
        (tmp / ".mira" / "skills").mkdir(parents=True, exist_ok=True)
        action_resources = build_resources(tmp, settings=settings, config=config, create_examples=False, enable_execute=False)
        plan_resources = build_resources(tmp, settings=settings, config=config, create_examples=False, enable_execute=False)
        act = build_agent(config, tmp, saver, metadata=metadata, mcp_manager=None, resources=action_resources)
        plan = build_plan_agent(config, tmp, saver, metadata=metadata, mcp_manager=None, resources=plan_resources)
        act_id, plan_id = "handoff-live:act", "handoff-live:plan"
        marker = "MIRA-LIVE-" + secrets.token_hex(4).upper()
        print(f"  Test canary: {marker}", flush=True)

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
        plan_answer = await ask(plan, plan_id, "What is the exact project codename from the previous conversation? "
                                "Answer with only that codename. This is a discussion question, not a request to prepare a Plan. "
                                "Do not call tools.", initial_history=effective(after_todo))
        require(marker.lower() in plan_answer.lower(), "Live Plan recalls the Act-only marker")
        require(signature((await snapshot(act, act_id)).values["messages"]) == frozen_act,
                "Original Act conversation is unmodified by Plan")
        # Plan can own a different todo state without changing Act's.
        state_p = await snapshot(plan, plan_id)
        plan_todo = [{"content": "Preserve independent Plan todo", "status": "pending"}]
        await plan.aupdate_state(state_p.config, {"todos": plan_todo})
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan's own todo state initialized")

        print("\n[3] LIVE Plan compaction -> Act", flush=True)
        await append_old_history(plan, plan_id, "PLAN")
        await real_compact(plan, plan_id)
        await replace_conversation(plan, plan_id, act, act_id)
        response = await ask(act, act_id,
                             "What is our project codename from the earlier discussion? "
                             "Please give the exact code and no other text. Do not call tools.")
        require(marker.lower() in response.lower(),
                "Live Act recalls marker after native Plan compaction (otherwise inspect summary)")
        require((await snapshot(act, act_id)).values.get("todos") == todo,
                "Act original todo still present after Plan compaction")

        print("\n[4] LIVE Act compaction -> Plan", flush=True)
        await append_old_history(act, act_id, "ACT")
        await real_compact(act, act_id)
        await replace_conversation(act, act_id, plan, plan_id)
        response = await ask(plan, plan_id,
                             "What is the project codename from the conversation? "
                             "Return it verbatim, no tool calls and no formal Plan.")
        require(marker.lower() in response.lower(),
                "Live Plan recalls marker after native Act compaction (otherwise inspect summary)")
        require((await snapshot(plan, plan_id)).values.get("todos") == plan_todo,
                "Plan original todo still present after Act compaction")
        require((await snapshot(act, act_id)).values.get("todos") == todo,
                "Act original todo still present at end")
        print("\nPASS: LIVE candidate handoff checks passed in scratch environment.", flush=True)
        print("LIMIT: Does not prove full MIRA Goal/HITL workflow or cold-start orchestration.", flush=True)
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
