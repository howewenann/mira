"""Deterministic proof-of-concept: Rubric verifier offload-read coverage nudges.

Copy to MIRA/tests/probes/probe_rubric_offload_read_coverage.py, then run:

    python tests/probes/probe_rubric_offload_read_coverage.py

No live model, workspace writes, or MIRA source changes. Uses a real LangGraph
agent with a deterministic fake verifier model and an instrumented read_file.
It tests that incomplete verifier reads trigger transient, bounded nudges.

PRODUCTION REQUIREMENT NOT IMPLEMENTED BY THIS PROBE:
MIRA's deterministic offload step must supply an AUTHORITATIVE per-verification
manifest (file path + exact total source-line count) to the nested verifier's
state. This must not be inferred from the model's prose, guessed from the
bounded Rubric transcript, or taken from untrusted tool-result instructions.
The guard belongs in the nested verifier, before _verification_evidence() is
assembled for the final grader. It must not change the Rubric grade itself.

A tool read proves evidence was retrieved, NOT that the LLM understood it.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, PrivateStateAttr, hook_config
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The manifest is supplied by the deterministic offloader, not the LLM.
MANIFEST = [
    {"path": "/rubric_offload/msg-001.txt", "total_lines": 3},
    {"path": "/rubric_offload/msg-002.txt", "total_lines": 2},
]
FILES = {
    MANIFEST[0]["path"]: ["alpha", "beta", "gamma"],
    MANIFEST[1]["path"]: ["one", "two"],
}
READ_HEADER = re.compile(r"^@@ lines (\d+)-(\d+) of (\d+)(?: \| .*)? @@$", re.MULTILINE)


@tool("read_file")
def probe_read_file(file_path: str, offset: int = 0, limit: int = 500) -> str:
    """Read numbered lines from one synthetic offload file (probe only)."""
    if file_path not in FILES:
        raise FileNotFoundError(file_path)
    rows = FILES[file_path]
    if offset >= len(rows) or offset < 0 or limit <= 0:
        raise ValueError(f"Invalid range {offset=} {limit=} for {file_path}")
    selected = rows[offset : offset + limit]
    last = offset + len(selected)
    content = "\n".join(f"{offset + n + 1}  {line}" for n, line in enumerate(selected))
    return f"@@ lines {offset + 1}-{last} of {len(rows)} @@\n{content}"


def coverage_gaps(messages: list[Any], manifest: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Which source lines remain unread, using linked successful tool results?

    Requires a matching AI tool-call ID, ToolMessage ID, successful result,
    exact path, parsable source-line header, and expected total-line count.
    Calls alone, unrelated files, errors, and truncated pages do not count.
    """
    expected: dict[str, int] = {str(item["path"]): int(item["total_lines"]) for item in manifest}
    covered: dict[str, set[int]] = {path: set() for path in expected}
    call_args: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls:
            if call.get("name") != "read_file" or not isinstance(call.get("args"), dict):
                continue
            call_id = str(call.get("id") or "")
            args = call["args"]
            path = str(args.get("file_path") or "")
            if call_id and path in expected:
                call_args[call_id] = (path, args)

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.status == "error":
            continue
        linked = call_args.get(str(msg.tool_call_id or ""))
        if linked is None:
            continue
        path, args = linked
        text = str(msg.content)
        match = READ_HEADER.search(text)
        if match is None:
            continue
        start, end, reported_total = map(int, match.groups())
        expected_total = expected[path]
        if not (1 <= start <= end <= expected_total == reported_total):
            continue
        try:
            requested_offset = int(args.get("offset") or 0)
            requested_limit = int(args.get("limit") or 500)
        except (TypeError, ValueError):
            continue
        if requested_offset != start - 1 or end - start + 1 > requested_limit:
            continue
        covered[path].update(range(start, end + 1))

    return {
        path: [line for line in range(1, total + 1) if line not in covered[path]]
        for path, total in expected.items()
        if len(covered[path]) < total
    }


class GuardState(AgentState):
    offloaded_message_manifest: NotRequired[list[dict[str, Any]]]
    _probe_guard_attempts: NotRequired[Annotated[int, PrivateStateAttr]]
    _probe_guard_pending: NotRequired[Annotated[str | None, PrivateStateAttr]]


class ProbeVerifierReadCoverageGuard(AgentMiddleware):
    """Small test-only verifier gate; does not judge rubric satisfaction."""

    state_schema = GuardState
    name = "ProbeVerifierReadCoverageGuard"

    def __init__(self, max_nudges: int = 2) -> None:
        self.max_nudges = max_nudges

    def before_agent(self, state: GuardState, runtime: Any) -> dict[str, Any]:  # noqa: ARG002
        return {"_probe_guard_attempts": 0, "_probe_guard_pending": None}

    async def abefore_agent(self, state: GuardState, runtime: Any) -> dict[str, Any]:
        return self.before_agent(state, runtime)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._with_guidance(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._with_guidance(request))

    @staticmethod
    def _with_guidance(request: Any) -> Any:
        reminder = str(request.state.get("_probe_guard_pending") or "")
        if not reminder:
            return request
        system = request.system_message
        if system is None:
            system = SystemMessage(content=reminder)
        else:
            system = system.model_copy(update={
                "content": [*system.content_blocks, {"type": "text", "text": "\n\n" + reminder}],
            })
        return request.override(system_message=system)

    def after_model(self, state: GuardState, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        latest = next((m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), None)
        if latest is not None and latest.tool_calls:
            return {"_probe_guard_pending": None}
        return None

    async def aafter_model(self, state: GuardState, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: GuardState, runtime: Any) -> dict[str, Any] | None:
        return self._after_agent_update(state)

    async def aafter_agent(self, state: GuardState, runtime: Any) -> dict[str, Any] | None:
        return self._after_agent_update(state)

    def _after_agent_update(self, state: GuardState) -> dict[str, Any] | None:
        manifest = list(state.get("offloaded_message_manifest") or [])
        if not manifest:
            return None
        missing = coverage_gaps(list(state.get("messages") or []), manifest)
        if not missing:
            return {"_probe_guard_pending": None}
        count = int(state.get("_probe_guard_attempts") or 0)
        if count >= self.max_nudges:
            # Fail closed: never silently proceed to the final grader when the
            # required evidence has not been read, even after nudging.
            raise RuntimeError(f"Rubric verifier offload-read coverage incomplete: {missing}")
        summary = "; ".join(f"{path} missing source lines {ranges}" for path, ranges in missing.items())
        return {
            "_probe_guard_attempts": count + 1,
            "_probe_guard_pending": (
                "VERIFIER READ-COVERAGE NUDGE: Before finishing verification, use "
                "read_file to retrieve ALL still-missing offloaded message lines. "
                f"Required evidence: {summary}. Do not claim those files were read "
                "until successful tool results are available."
            ),
            "jump_to": "model",
        }


class RecordingFakeModel(FakeMessagesListChatModel):
    requests: list[list[Any]] = Field(default_factory=list, exclude=True)

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(call_id: str, path: str, offset: int, limit: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "read_file", "id": call_id, "type": "tool_call",
        "args": {"file_path": path, "offset": offset, "limit": limit},
    }])


def request_has_nudge(messages: list[Any]) -> bool:
    return any(
        isinstance(m, SystemMessage) and "VERIFIER READ-COVERAGE NUDGE" in str(m.content)
        for m in messages
    )


async def successful_coverage_case() -> None:
    a, b = MANIFEST[0]["path"], MANIFEST[1]["path"]
    model = RecordingFakeModel(responses=[
        AIMessage(content="VERIFICATION_COMPLETE"),  # incorrect premature finish
        tool_call("a-page1", a, 0, 2),
        tool_call("b-page1", b, 0, 2),
        AIMessage(content="VERIFICATION_COMPLETE"),  # still missing a:3
        tool_call("a-page2", a, 2, 1),
        AIMessage(content="VERIFICATION_COMPLETE"),  # now fully covered
    ])
    graph = create_agent(
        model=model,
        tools=[probe_read_file],
        middleware=[ProbeVerifierReadCoverageGuard(max_nudges=2)],
    )
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Collect all offload evidence for grading")],
        "offloaded_message_manifest": MANIFEST,
    })
    assert not coverage_gaps(result["messages"], MANIFEST), "Not all messages read"
    assert len(model.requests) == 6, f"Expected 6 model calls, got {len(model.requests)}"
    assert request_has_nudge(model.requests[1]), "First nudge missing on retry"
    assert request_has_nudge(model.requests[4]), "Second nudge missing for remaining page"
    assert sum(isinstance(m, HumanMessage) for m in result["messages"]) == 1, (
        "Nudges should be transient, not checkpointed as human messages"
    )
    print("[VERIFIER SUCCESS] PASS: wrong early stops were nudged; both paths and every line read")
    print("  requested reads:", [
        c["args"] for m in result["messages"] if isinstance(m, AIMessage)
        for c in m.tool_calls if c.get("name") == "read_file"
    ])


async def exhausted_case() -> None:
    model = RecordingFakeModel(responses=[AIMessage(content="VERIFICATION_COMPLETE") for _ in range(3)])
    graph = create_agent(
        model=model,
        tools=[probe_read_file],
        middleware=[ProbeVerifierReadCoverageGuard(max_nudges=2)],
    )
    try:
        await graph.ainvoke({
            "messages": [HumanMessage(content="Verify offloaded content")],
            "offloaded_message_manifest": MANIFEST,
        })
    except RuntimeError as exc:
        assert "offload-read coverage incomplete" in str(exc)
        assert len(model.requests) == 3
        print("[VERIFIER EXHAUSTED] PASS: retry cap fails closed instead of grading unseen data")
        return
    raise AssertionError("Unverified offloads incorrectly reached the final grader")


def negative_evidence_cases() -> None:
    path = MANIFEST[0]["path"]
    call = tool_call("call-x", path, 0, 3)
    bad_error = ToolMessage(content="@@ lines 1-3 of 3 @@", tool_call_id="call-x", name="read_file", status="error")
    unmatched = ToolMessage(content="@@ lines 1-3 of 3 @@", tool_call_id="different", name="read_file")
    partial = ToolMessage(content="@@ lines 1-2 of 3 @@", tool_call_id="call-x", name="read_file")
    assert coverage_gaps([call, bad_error], [MANIFEST[0]])[path] == [1, 2, 3]
    assert coverage_gaps([call, unmatched], [MANIFEST[0]])[path] == [1, 2, 3]
    assert coverage_gaps([call, partial], [MANIFEST[0]])[path] == [3]
    print("[EVIDENCE MATCHING] PASS: errors, unmatched IDs, partial reads do not count as complete")


async def main() -> None:
    negative_evidence_cases()
    await successful_coverage_case()
    await exhausted_case()
    print("\nPASS: deterministic guard behavior; no real LLMs or external files were used.")
    print("INTEGRATION NOTE: pass a verified offload manifest into MIRA's nested verifier first.")


if __name__ == "__main__":
    asyncio.run(main())
