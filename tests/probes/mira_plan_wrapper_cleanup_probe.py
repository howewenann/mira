#!/usr/bin/env python
"""Reproduce a REAL MIRA Plan -> Act refusal and isolate old Plan user instructions.

Place under tests/probes/ and run from the MIRA repository:
    python tests/probes/mira_plan_wrapper_cleanup_probe.py

This reuses the tested real-plan probe and the exact Plan transcript for all
three Act experiments (baseline, cleaned, cleaned+named). The original two
Plan turns run with MIRA's CURRENT, UNMODIFIED Plan request templates.

Only the old transferred Plan HumanMessages are changed in the cleaned tests:
full PLAN_REQUEST_TEMPLATE wrappers become the exact original user text.
All assistant/tool messages are left unchanged, except that the third condition
also sets historical AIMessage.name='plan'. No inline XML, no production changes.

The cleanup is patched only into the source projection of MIRA's EXISTING
Plan -> Act handoff. No checkpoint injection, no fake synthetic Plan replies,
no production code or real-workspace writes. Real model calls use your local
model and may take time.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import core.execution.handoff as handoff
from config.loader import load_config
from core.application.app import MiraApplication
from core.interface.requests import (
    ApprovalRequest, ArtifactDisplayRequest, ArtifactReviewRequest,
    AskUserRequest, MCPApprovalRequest,
)

FILE_NAME = "palindrome_utility.py"
PLAN_REQUEST_1 = "write a palindrome function to the root directory"
PLAN_REQUEST_2 = "i dont want you to prepare a plan. just write the file"
ACT_REQUEST = "write the file now"
MODE_POLICY_MARKER = "Plan mode is one continuous read-only conversation"


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


class AutoFrontend:
    """Reproduce the manual name choice and Clear (never Implement) decisions."""

    def __init__(self) -> None:
        self.reviews: list[tuple[str, str]] = []
        self.questions: list[str] = []

    def emit(self, event: object) -> None:
        pass

    async def request(self, request: object) -> object:
        if isinstance(request, ArtifactReviewRequest):
            self.reviews.append((request.artifact_type, "clear"))
            print(f"  frontend: {request.artifact_type} review -> CLEAR (not Implement)", flush=True)
            return {"action": "clear"}
        if isinstance(request, AskUserRequest):
            self.questions.append(str(request.interrupt.value))
            if len(self.questions) > 4:
                raise RuntimeError("Plan asked more than four questions; aborting probe")
            print(f"  frontend: ask_user -> {FILE_NAME}", flush=True)
            return FILE_NAME
        if isinstance(request, ApprovalRequest):
            print("  frontend: approved a tool operation in isolated scratch workspace", flush=True)
            return [{"type": "allow_once"} for _ in request.interrupts]
        if isinstance(request, ArtifactDisplayRequest):
            return "Displayed retained artifact"
        if isinstance(request, MCPApprovalRequest):
            return "deny"
        raise RuntimeError(f"Unexpected frontend request: {type(request).__name__}")


def make_config(repo: Path, workspace: Path) -> dict[str, Any]:
    config = dict(load_config(repo))
    settings = copy.deepcopy(config["settings"])
    # Disable only tracing and unrelated dynamic subagents, as in the prior
    # working probes; leave Plan prompts, tool policies, and template wrappers intact.
    settings.setdefault("tracing", {})["enabled"] = False
    settings.setdefault("system", {}).setdefault("dynamic_subagents", {})["enabled"] = False
    config["settings"] = settings
    config["workspace"] = str(workspace)
    config["session_dir"] = str(workspace / ".mira" / "_sessions")
    return config


def seed_workspace(repo: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "pyproject.toml"):
        origin = repo / name
        if origin.is_file():
            shutil.copy2(origin, dest / name)
    (dest / "scripts").mkdir(exist_ok=True)
    (dest / "core").mkdir(exist_ok=True)
    # The test runs against the scratch workspace, NOT the actual MIRA repo.


async def async_assert_settled(agent: Any, thread_id: str) -> Any:
    state = await agent.aget_state(thread_config(thread_id))
    if state.next or state.tasks:
        raise RuntimeError(
            f"Checkpoint {thread_id} has pending graph work: "
            f"next={state.next!r}, tasks={len(state.tasks)}"
        )
    return state


def message_shape(messages: list[Any]) -> list[tuple[Any, ...]]:
    # Compare the actual historical content and message roles, ignoring only
    # speaker names and auto-generated message IDs.
    return [
        (type(m).__name__, m.model_dump(exclude={"id", "name"}))
        for m in messages
    ]


def analyze_file(workspace: Path, response: Any) -> dict[str, Any]:
    file = workspace / FILE_NAME
    content = file.read_text(encoding="utf-8") if file.exists() else None
    funcs: list[str] = []
    parse_error = None
    if content is not None:
        try:
            tree = ast.parse(content)
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        except SyntaxError as exc:
            parse_error = str(exc)
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    answer = str(getattr(response, "final_text", "") or "").strip()
    wrote = "write_file" in tool_calls
    valid = bool(file.exists() and wrote and any("palindrome" in name.lower() for name in funcs))
    refusal = bool(re.search(r"plan(?:ning)? mode|read.only|disabled|cannot write|can't write", answer, re.I))
    return {
        "valid": valid,
        "file_exists": file.exists(),
        "wrote": wrote,
        "functions": funcs,
        "parse_error": parse_error,
        "refusal": refusal and not valid,
        "tool_calls": tool_calls,
        "answer": answer,
    }


def display_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n[{label}] {'WROTE FILE' if result['valid'] else 'DID NOT WRITE VALID FILE'}", flush=True)
    print(f"  tool_calls={result['tool_calls']}", flush=True)
    print(f"  file_exists={result['file_exists']} "
          f"write_file_called={result['wrote']} "
          f"functions={result['functions']} "
          f"parse_error={result['parse_error']!r}", flush=True)
    print(f"  plan_mode_refusal={result['refusal']}", flush=True)
    print(f"  answer={result['answer'][:900]!r}", flush=True)


async def run_original_plan(repo: Path, workspace: Path, timeout: float) -> tuple[list[Any], dict[str, Any]]:
    frontend = AutoFrontend()
    app = await MiraApplication.start(
        workspace=workspace, config=make_config(repo, workspace), frontend=frontend
    )
    try:
        print(f"  Main model (baseline): {getattr(app, 'model_name', 'unknown')}", flush=True)
        session = await app.open_session("original-plan")
        await session.set_mode("plan")
        assert session.mode["planning"]
        print("\n=== Generate REAL Plan transcript (same manual user requests) ===", flush=True)
        one = await asyncio.wait_for(session.prompt(PLAN_REQUEST_1), timeout)
        print(f"  Plan response 1: {str(one.final_text)[:250]!r}", flush=True)
        two = await asyncio.wait_for(session.prompt(PLAN_REQUEST_2), timeout)
        print(f"  Plan response 2: {str(two.final_text)[:550]!r}", flush=True)
        if not session.mode["planning"]:
            raise AssertionError("Plan mode changed unexpectedly before the manual switch")
        source = await async_assert_settled(session.plan_agent, f"{session.id}:plan")
        history = list(handoff.effective_conversation(session.plan_agent, source.values))
        ai_count = sum(isinstance(m, AIMessage) for m in history)
        wrapped_count = sum(
            isinstance(m, HumanMessage) and MODE_POLICY_MARKER in str(m.content)
            for m in history
        )
        print(f"  Real checkpoint: {len(history)} messages, "
              f"{ai_count} AI messages, {wrapped_count} Plan policy wrappers", flush=True)
        if ai_count < 1 or wrapped_count < 2:
            raise AssertionError(
                "Real Plan transcript did not include the expected old Plan HumanMessage "
                "wrappers and model replies; cannot run the controlled comparison"
            )
        # Baseline: do not patch anything, use production synchronize_conversation.
        print("\n=== BASELINE: real Plan -> Act, unmodified handoff ===", flush=True)
        await session.set_mode("act")
        assert not session.mode["planning"]
        inherited = await async_assert_settled(session.agent, session.id)
        actual = list(handoff.effective_conversation(session.agent, inherited.values))
        assert message_shape(actual) == message_shape(history), (
            "Baseline handoff lost/changed Plan history"
        )
        assert "write_file" in {t["name"] for t in session.snapshot().tools}, (
            "Act write_file tool unavailable; this is a tool-policy error, not mode contamination"
        )
        answer = await asyncio.wait_for(session.prompt(ACT_REQUEST), timeout)
        result = analyze_file(workspace, answer)
        display_result("BASELINE", result)
        print("  source Plan review actions:", frontend.reviews, flush=True)
        return history, result
    finally:
        await app.shutdown()




def clean_plan_request_wrappers(history: list[Any]) -> tuple[list[Any], int]:
    """Strip the duplicated Plan policy wrapper, retaining exact user words.

    This strips only PLAN_REQUEST_TEMPLATE-shaped HumanMessages. It does not
    alter other HumanMessages (e.g. actual answers or control-tool replies),
    and intentionally DOES NOT modify any historical Plan AIMessage.
    """
    cleaned: list[Any] = []
    removed = 0
    prefix = "You are in planning mode (Plan mode)."
    delimiter = "\nUser request:\n"
    for message in history:
        if isinstance(message, HumanMessage) and isinstance(message.content, str) and message.content.startswith(prefix):
            if delimiter not in message.content:
                raise AssertionError("Found Plan wrapper without User request delimiter; refusing unsafe cleanup")
            # Only remove operational instructions. Preserve exact original
            # user request, including whitespace. The Plan system prompt is
            # untouched; only an inherited HumanMessage changes.
            user_request = message.content.rsplit(delimiter, 1)[1]
            if not user_request.strip():
                raise AssertionError("Plan wrapper contained an empty user request")
            cleaned.append(message.model_copy(deep=True, update={"content": user_request}))
            removed += 1
        else:
            cleaned.append(message.model_copy(deep=True))
    return cleaned, removed


def make_variant_history(history: list[Any], *, named: bool) -> tuple[list[Any], int]:
    prepared, stripped = clean_plan_request_wrappers(history)
    if stripped != 2:
        raise AssertionError(f"Expected exactly 2 actual Plan request wrappers, found {stripped}")
    if named:
        prepared = [
            m.model_copy(deep=True, update={"name": "plan"}) if isinstance(m, AIMessage) else m
            for m in prepared
        ]
    # Ensure only the intended fields changed: no dropped history, reordering,
    # role changes, tool-call changes, or failed tool linkage.
    if len(prepared) != len(history):
        raise AssertionError("Cleanup modified message count")
    for index, (old, new) in enumerate(zip(history, prepared)):
        if type(old) is not type(new):
            raise AssertionError(f"Cleanup modified role at index {index}")
        old_data = old.model_dump(exclude={"id", "content", "name"})
        new_data = new.model_dump(exclude={"id", "content", "name"})
        if old_data != new_data:
            raise AssertionError(f"Cleanup modified non-text metadata at index {index}")
        if not (isinstance(old, HumanMessage) and isinstance(old.content, str)
                and old.content.startswith("You are in planning mode (Plan mode).")):
            if old.content != new.content:
                raise AssertionError(f"Cleanup changed a non-wrapper message at index {index}")
        if not named and old.name != new.name:
            raise AssertionError(f"Cleanup unexpectedly renamed message at index {index}")
        if named and not isinstance(old, AIMessage) and old.name != new.name:
            raise AssertionError(f"Cleanup unexpectedly renamed non-AI message at index {index}")
    if any(isinstance(m, HumanMessage) and MODE_POLICY_MARKER in str(m.content) for m in prepared):
        raise AssertionError("Plan behavior policy still present in a transferred HumanMessage")
    return prepared, stripped


async def run_cleaned_replay(
    repo: Path,
    workspace: Path,
    history: list[Any],
    timeout: float,
    *,
    named: bool,
) -> dict[str, Any]:
    condition = "CLEAN+NAMED" if named else "CLEAN ONLY"
    frontend = AutoFrontend()
    app = await MiraApplication.start(
        workspace=workspace, config=make_config(repo, workspace), frontend=frontend
    )
    try:
        print(f"  Main model ({condition}): {getattr(app, 'model_name', 'unknown')}", flush=True)
        session = await app.open_session("clean-named-plan" if named else "clean-only-plan")
        # Initialize checkpoints using the EXACT approach that worked in v3.
        # No aupdate_state()/as_node manipulation.
        await asyncio.wait_for(session.prompt("Reply READY. Do not call tools."), timeout)
        await session.set_mode("plan")
        await async_assert_settled(session.plan_agent, f"{session.id}:plan")

        transferred_source, stripped = make_variant_history(history, named=named)
        ai_count = sum(isinstance(m, AIMessage) for m in transferred_source)
        assert ai_count >= 1
        if named:
            assert all(m.name == "plan" for m in transferred_source if isinstance(m, AIMessage))
        else:
            assert all(m.name == old.name for m, old in zip(transferred_source, history))

        print(f"\n=== {condition}: SAME REAL Plan history, {stripped} redundant user wrappers removed ===", flush=True)
        for old, new in zip(history, transferred_source):
            if isinstance(old, HumanMessage) and old.content != new.content:
                print(f"  Before Plan HumanMessage: {str(old.content)[:165]!r}...", flush=True)
                print(f"  After  Plan HumanMessage: {new.content!r}", flush=True)
        original_projection = handoff.effective_conversation
        injection_count = 0

        def project_cleaned_source(agent: Any, values: dict[str, Any]) -> list[Any]:
            nonlocal injection_count
            if agent is session.plan_agent:
                injection_count += 1
                return [m.model_copy(deep=True) for m in transferred_source]
            return original_projection(agent, values)

        # Keep MIRA's actual handoff & checkpoint synchronization; patch ONLY
        # its source history read, then restore before the Act LLM runs.
        with patch.object(handoff, "effective_conversation", side_effect=project_cleaned_source):
            await session.set_mode("act")
        assert injection_count == 1, f"Expected 1 source projection, got {injection_count}"
        assert not session.mode["planning"], "Act mode was not selected"

        inherited = await async_assert_settled(session.agent, session.id)
        actual = list(original_projection(session.agent, inherited.values))
        assert len(actual) == len(transferred_source), "Handoff dropped some messages"
        assert all(
            a.model_dump(exclude={"id"}) == b.model_dump(exclude={"id"})
            for a, b in zip(actual, transferred_source)
        ), "Handoff changed the prepared real Plan transcript"
        assert sum(isinstance(m, HumanMessage) and MODE_POLICY_MARKER in str(m.content)
                   for m in actual) == 0
        assert sum(isinstance(m, AIMessage) for m in actual) == ai_count
        assert "write_file" in {t["name"] for t in session.snapshot().tools}, (
            "Act cannot use write_file; this is not a prompt contamination test"
        )
        print(f"  Verified checkpoint: {len(actual)} messages; {stripped} policy wrappers removed; "
              f"{ai_count} AI messages {'named plan' if named else 'unchanged'}", flush=True)
        answer = await asyncio.wait_for(session.prompt(ACT_REQUEST), timeout)
        result = analyze_file(workspace, answer)
        display_result(condition, result)
        return result
    finally:
        await app.shutdown()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None, help="MIRA repository root")
    parser.add_argument("--keep", action="store_true", help="Keep all scratch workspaces for inspection")
    parser.add_argument("--timeout", type=float, default=360, help="Per-model-turn timeout in seconds")
    parser.add_argument("--no-named", action="store_true", help="Skip optional cleaned+named Act comparison")
    args = parser.parse_args()
    repo = (args.repo or Path(__file__).resolve().parents[2]).resolve()
    if not (repo / "agent" / "factory.py").is_file():
        parser.error("Put probe in tests/probes/ or pass --repo D:/Projects/mira")

    async def evaluate(root: Path) -> None:
        base_dir = root / "baseline"
        clean_dir = root / "clean-only"
        named_dir = root / "clean-named"
        for path in [base_dir, clean_dir, named_dir]:
            seed_workspace(repo, path)
        print(f"Source config: {repo}", flush=True)
        print(f"Isolated workspaces: {root}", flush=True)
        history, baseline = await run_original_plan(repo, base_dir, args.timeout)
        cleaned = await run_cleaned_replay(repo, clean_dir, history, args.timeout, named=False)
        cleaned_named = None
        if not args.no_named:
            cleaned_named = await run_cleaned_replay(repo, named_dir, history, args.timeout, named=True)

        def score(r: dict[str, Any]) -> str:
            return ("WRITE SUCCESS" if r["valid"] else
                    "PLAN-MODE REFUSAL" if r["refusal"] else "OTHER FAILURE / INCONCLUSIVE")

        print("\n================ SAME-TRANSCRIPT COMPARISON ================", flush=True)
        print(f"BASELINE:      {score(baseline)}", flush=True)
        print(f"CLEAN ONLY:    {score(cleaned)}", flush=True)
        if cleaned_named is not None:
            print(f"CLEAN + NAMED: {score(cleaned_named)}", flush=True)
        if baseline["refusal"] and cleaned["valid"]:
            print("FINDING: Removing redundant Plan HumanMessage instructions fixed "
                  "this reproducible case without agent naming.", flush=True)
        elif baseline["refusal"] and not cleaned["valid"] and cleaned_named is not None and cleaned_named["valid"]:
            print("FINDING: Cleanup alone did not fix this case, but cleanup + naming did.", flush=True)
        elif baseline["refusal"] and cleaned["refusal"]:
            print("FINDING: Cleanup alone did not prevent mode confusion in this run. "
                  "Historical Plan replies remain and may matter.", flush=True)
        elif baseline["valid"]:
            print("FINDING: Baseline unexpectedly passed, so this run cannot prove a cleanup benefit.", flush=True)
        else:
            print("FINDING: Inspect the actual replies. This run does not establish a causal fix.", flush=True)
        print("NOTE: Actual Plan generation still used original wrappers. Only the "
              "transferred historical HumanMessages were cleaned for the Act trials. "
              "No production edits; a single run does not prove universal reliability.", flush=True)

    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="mira-plan-wrapper-clean-"))
        try:
            await evaluate(root)
        finally:
            print(f"Scratch workspaces kept at: {root}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="mira-plan-wrapper-clean-") as tmp:
            await evaluate(Path(tmp))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
