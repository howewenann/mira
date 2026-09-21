"""HIGH-LEVEL REAL-LLM probe: lazy vs prebuilt Rubric grader.

Place in:
    tests/probes/probe_rubric_prebuilt_grader.py

Run FROM the probe directory:
    python probe_rubric_prebuilt_grader.py

This deliberately starts from the earlier fact we already proved:
MIRA's real `_astream_final_grader()` can stream real reasoning + assistant JSON
when the grader exists outside the parent run.

This probe changes ONE high-level thing only.

A) CURRENT MIRA
   The final grader is created lazily during the parent Rubric run.

B) CANDIDATE
   The exact same final grader is created once BEFORE the parent v3 run starts,
   then the full real parent -> verifier -> grader flow runs unchanged.

No fake model.
No mocked grader output.
No stream-version hacks.
No callback surgery.
No raw-event transformer.
No subagent changes.
No rewritten grader loop.

PASS means eager/prebuilt grader construction is enough to restore the real
grader reasoning + assistant stream while preserving the real structured result.
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

REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


def short(text: str, limit: int = 900) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


async def run_case(*, prebuild_grader: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    if prebuild_grader:
        # THE ONLY CANDIDATE CHANGE.
        # Build the same real ProviderStrategy grader before the outer v3 run.
        middleware._ensure_final_grader()

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name=(
            "rubric_prebuilt_grader_probe"
            if prebuild_grader
            else "rubric_lazy_grader_probe"
        ),
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    counts: Counter[str] = Counter()
    reasoning_parts: list[str] = []
    assistant_parts: list[str] = []
    evaluations: list[dict[str, Any]] = []

    async def consume_custom() -> None:
        async for event in run.custom:
            if not isinstance(event, dict):
                continue

            if event.get("type") != "rubric_inspection_delta":
                continue

            if str(event.get("phase") or "") != "grader":
                continue

            kind = str(event.get("kind") or "")
            text = str(event.get("text") or "")

            counts[kind] += 1

            if kind == "reasoning":
                reasoning_parts.append(text)
            elif kind == "assistant":
                assistant_parts.append(text)

    async def consume_output() -> None:
        output = await run.output()
        if isinstance(output, dict):
            evaluations.extend(output.get("_rubric_evaluations") or [])

    await asyncio.gather(
        consume_custom(),
        consume_output(),
    )

    return {
        "counts": counts,
        "reasoning": "".join(reasoning_parts),
        "assistant": "".join(assistant_parts),
        "evaluations": evaluations,
    }


def show(label: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)

    print("grader deltas on parent stream.custom:", dict(result["counts"]))
    print("real rubric evaluations:", len(result["evaluations"]))

    if result["evaluations"]:
        evaluation = result["evaluations"][-1]
        print("evaluation result:", evaluation.get("result"))
        print("criteria:", len(evaluation.get("criteria") or []))

    print("reasoning chars:", len(result["reasoning"]))
    print("assistant chars:", len(result["assistant"]))

    if result["reasoning"]:
        print("\nREAL GRADER REASONING:")
        print(short(result["reasoning"]))

    if result["assistant"]:
        print("\nREAL GRADER ASSISTANT:")
        print(short(result["assistant"]))


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    print("=" * 96)
    print("MIRA HIGH-LEVEL REAL-LLM PREBUILT-GRADER PROBE")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print()
    print("Both cases run real parent -> verifier -> ProviderStrategy grader calls.")
    print("Only difference: whether the grader is built before the parent run.")
    print("=" * 96)

    print("\nRunning A) CURRENT MIRA — lazy grader...")
    current = await run_case(prebuild_grader=False)

    print("\nRunning B) CANDIDATE — grader prebuilt before parent v3 run...")
    prebuilt = await run_case(prebuild_grader=True)

    show("A) CURRENT MIRA — LAZY GRADER", current)
    show("B) CANDIDATE — PREBUILT GRADER", prebuilt)

    current_bug = (
        not current["reasoning"]
        and not current["assistant"]
        and len(current["evaluations"]) > 0
    )

    prebuilt_ok = (
        bool(prebuilt["reasoning"])
        and bool(prebuilt["assistant"])
        and prebuilt["counts"]["reasoning"] > 0
        and prebuilt["counts"]["assistant"] > 0
        and len(prebuilt["evaluations"]) > 0
    )

    print("\n" + "=" * 96)

    if current_bug and prebuilt_ok:
        print("PASS — REAL LLM FIX PROVEN")
        print(
            "Creating the exact same final grader before the outer v3 run restores "
            "its real reasoning + assistant stream in the full parent flow while "
            "preserving the real structured Rubric evaluation."
        )
        print()
        print(
            "Production fix direction: construct/cache the final Rubric grader "
            "before entering the parent streaming execution instead of lazily "
            "creating it inside the Rubric after-agent run."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not implement eager grader construction. It did not restore the "
            "full real grader stream."
        )

    print("=" * 96)


if __name__ == "__main__":
    asyncio.run(main())
