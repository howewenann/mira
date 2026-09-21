"""Probe the native v3 subagent-handle message stream for Rubric grader.

Place in:
    tests/probes/probe_rubric_grader_subagent_messages.py

Run FROM the probe directory:
    python probe_rubric_grader_subagent_messages.py

This tests the seam MIRA already consumes in core/execution/streams/subagents.py:
    parent run.subagents
        -> internal rubric_grader handle
        -> child.messages

No MIRA production code is modified.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.stream.transformers import CustomTransformer

from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL
from core.execution.streams.messages import consume_messages


REQUEST = "Reply with exactly: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


class Capture:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.reasoning = ""
        self.assistant = ""

    def reasoning_delta(self, text: str, **_kwargs: Any) -> None:
        self.counts["reasoning_delta"] += 1
        self.reasoning += str(text)

    def text_delta(self, text: str, **_kwargs: Any) -> None:
        self.counts["assistant_delta"] += 1
        self.assistant += str(text)

    def model_stream_finished(self) -> None:
        self.counts["model_stream_finished"] += 1

    # Grader has tools=[], but consume_messages expects a renderer-shaped object.
    def tool_call(self, *_args: Any, **_kwargs: Any) -> None:
        self.counts["tool_call"] += 1

    def tool_call_delta(self, *_args: Any, **_kwargs: Any) -> None:
        self.counts["tool_call_delta"] += 1

    def delegation_started(self, *_args: Any, **_kwargs: Any) -> None:
        self.counts["delegation_started"] += 1


def short(value: str, limit: int = 700) -> str:
    value = value.replace("\r", "")
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Answer the user's request directly.",
        name="rubric_subagent_messages_probe",
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    found: dict[str, Capture] = {}
    child_status: dict[str, str] = {}
    child_outputs: dict[str, Any] = {}
    tasks: list[asyncio.Task[None]] = []

    async def consume_child(child: Any) -> None:
        name = str(
            getattr(child, "graph_name", None)
            or getattr(child, "name", None)
            or "<unnamed>"
        )
        if name not in {"rubric_verifier", "rubric_grader"}:
            try:
                await child.output()
            except Exception:
                pass
            return

        capture = Capture()
        found[name] = capture

        async def messages() -> None:
            stream = getattr(child, "messages", None)
            if stream is not None:
                await consume_messages(
                    stream,
                    capture,
                    render_normal_tools=False,
                )

        async def output() -> None:
            try:
                child_outputs[name] = await child.output()
            except Exception as exc:
                child_outputs[name] = f"<error: {exc}>"

        await asyncio.gather(messages(), output())
        child_status[name] = str(getattr(child, "status", "") or "")

    async def consume_subagents() -> None:
        async for child in run.subagents:
            tasks.append(asyncio.create_task(consume_child(child)))

    async def drain_custom() -> None:
        async for _ in run.custom:
            pass

    async def drive_output() -> None:
        await run.output()

    await asyncio.gather(
        consume_subagents(),
        drain_custom(),
        drive_output(),
    )

    if tasks:
        await asyncio.gather(*tasks)

    print("=" * 88)
    print("MIRA native Rubric subagent.messages probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 88)

    print("\nInternal Rubric handles found:", list(found))

    for name in ("rubric_verifier", "rubric_grader"):
        print("\n" + "-" * 88)
        print(name)
        capture = found.get(name)
        if capture is None:
            print("NOT FOUND on run.subagents")
            continue

        print("status:", child_status.get(name))
        print("message capture counts:", dict(capture.counts))
        print("reasoning chars:", len(capture.reasoning))
        print("assistant chars:", len(capture.assistant))
        print("reasoning:")
        print(short(capture.reasoning) if capture.reasoning else "<none>")
        print("\nassistant:")
        print(short(capture.assistant) if capture.assistant else "<none>")

        output = child_outputs.get(name)
        if isinstance(output, dict):
            print("\noutput keys:", list(output))
            print("structured_response present:", output.get("structured_response") is not None)
        else:
            print("\noutput:", repr(output))

    print("\n" + "=" * 88)
    grader = found.get("rubric_grader")
    if (
        grader is not None
        and grader.reasoning
        and grader.assistant
    ):
        print("PASS")
        print(
            "The grader's live reasoning + assistant stream exists on the native "
            "v3 run.subagents -> rubric_grader -> child.messages handle. "
            "MIRA's current internal-Rubric drain ignores child.messages, which "
            "explains why the Inspector stops at the grader input."
        )
    elif grader is not None and (grader.reasoning or grader.assistant):
        print("PARTIAL")
        print(
            "The grader child.messages stream exists, but only one visible content "
            "kind was captured. Inspect the output above before changing MIRA."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "The native rubric_grader child.messages handle did not expose the "
            "expected live content. Do not change production code."
        )
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
