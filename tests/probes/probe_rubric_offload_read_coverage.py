"""Deterministic production Rubric offload/read-coverage probe.

    python tests/probes/probe_rubric_offload_read_coverage.py

Uses production offloading, native StateBackend file reads, and the production
guard with a fake model. No saved sessions, network calls, or project writes.
Retrieval coverage does not establish model comprehension or rubric satisfaction.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rubric.middleware import MiraRubricMiddleware  # noqa: E402
from agent.rubric.offload import (  # noqa: E402
    EVIDENCE_TOOL,
    VerifierReadCoverageGuard,
    coverage_gaps,
    evidence_filesystem,
)

# Production creates the files and exact manifest from oversized source text.
PARENT_MESSAGES = [
    HumanMessage(content="Review the source notes."),
    AIMessage(content="\n".join(["alpha " * 240, "beta " * 300, "gamma " * 240])),
    AIMessage(content="\n".join(["one " * 600, "two " * 600])),
]
PARENT_STATE = {"rubric": "Ground the summary in the source notes.", "messages": PARENT_MESSAGES}
PAYLOAD = MiraRubricMiddleware._verifier_input(PARENT_STATE, iteration=0)
MANIFEST = PAYLOAD["offloaded_message_manifest"]
FILES = PAYLOAD["files"]


class RecordingFakeModel(FakeMessagesListChatModel):
    requests: list[list[Any]] = Field(default_factory=list, exclude=True)

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(call_id: str, path: str, offset: int, limit: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": EVIDENCE_TOOL, "id": call_id, "type": "tool_call",
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
        middleware=[evidence_filesystem(), VerifierReadCoverageGuard(max_nudges=2)],
    )
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Collect all offload evidence for grading")],
        "offloaded_message_manifest": MANIFEST,
        "files": FILES,
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
        for c in m.tool_calls if c.get("name") == EVIDENCE_TOOL
    ])


async def exhausted_case() -> None:
    model = RecordingFakeModel(responses=[AIMessage(content="VERIFICATION_COMPLETE") for _ in range(3)])
    graph = create_agent(
        model=model,
        middleware=[evidence_filesystem(), VerifierReadCoverageGuard(max_nudges=2)],
    )
    try:
        await graph.ainvoke({
            "messages": [HumanMessage(content="Verify offloaded content")],
            "offloaded_message_manifest": MANIFEST,
            "files": FILES,
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
    partial = ToolMessage(content="@@ lines 1-2 of 3 @@\nalpha\nbeta", tool_call_id="call-x", name="read_file")
    assert coverage_gaps([call, bad_error], [MANIFEST[0]])[path] == [1, 2, 3]
    assert coverage_gaps([call, unmatched], [MANIFEST[0]])[path] == [1, 2, 3]
    assert coverage_gaps([call, partial], [MANIFEST[0]])[path] == [3]
    print("[EVIDENCE MATCHING] PASS: errors, unmatched IDs, partial reads do not count as complete")


async def main() -> None:
    negative_evidence_cases()
    await successful_coverage_case()
    await exhausted_case()
    print("\nPASS: deterministic guard behavior; no real LLMs or external files were used.")


if __name__ == "__main__":
    asyncio.run(main())
