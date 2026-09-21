"""High-level REAL-LLM Rubric streaming differential probe.

Place in:
    tests/probes/probe_rubric_high_level_context_isolation.py

Run FROM the probe directory:
    python probe_rubric_high_level_context_isolation.py

This deliberately starts from the earlier probe that already succeeded.

It runs THREE real cases:

1) DIRECT BASELINE
   Calls MIRA's real `_astream_final_grader()` directly, exactly like the
   previously successful probe. This must show real grader reasoning + assistant.

2) FULL PARENT — CURRENT MIRA
   Runs the real parent agent + real verifier + real grader. This should
   reproduce the bug: grader inspection deltas disappear.

3) FULL PARENT — CONTEXT-ISOLATED CANDIDATE
   Runs the SAME full parent flow, but executes the SAME production
   `_astream_final_grader()` in a fresh Python ContextVar context while carrying
   across ONLY MIRA's `_RUBRIC_RUN` event identity/writer.

There is no fake model and no mocked model output.

The candidate does NOT rewrite grader streaming, ProviderStrategy, prompts,
values handling, event normalization, or inspection rendering. It simply gives
the already-known-good production grader method the same clean execution
context it has in case (1).

PASS requires:
- case 1: real reasoning + assistant + structured_response
- case 2: no grader inspection deltas (bug reproduced)
- case 3: real reasoning + assistant + structured_response AND those deltas
          reach the parent custom stream
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.stream.transformers import CustomTransformer

import agent.rubric.middleware as rubric_module
from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL

REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)

DIRECT_INPUT = {
    "messages": [
        HumanMessage(
            content=(
                "Evaluate this diagnostic case.\n"
                "Criteria:\n"
                "1. Response contains STREAM_OK.\n"
                "2. Response says two plus two equals four.\n\n"
                "Response:\nSTREAM_OK. Two plus two equals four.\n\n"
                "Return the normal structured GraderResponse."
            )
        )
    ]
}


def short(text: str, limit: int = 900) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


async def direct_known_good() -> dict[str, Any]:
    """Re-run the previously successful direct production grader path."""
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(model=model, verifier_tools=[], max_iterations=1)
    grader = middleware._ensure_final_grader()

    counts: Counter[str] = Counter()
    reasoning_parts: list[str] = []
    assistant_parts: list[str] = []

    def capture_emit(event_type: str, **values: Any) -> None:
        if event_type != "rubric_inspection_delta":
            return
        if str(values.get("phase") or "") != "grader":
            return
        kind = str(values.get("kind") or "")
        text = str(values.get("text") or "")
        counts[kind] += 1
        if kind == "reasoning":
            reasoning_parts.append(text)
        elif kind == "assistant":
            assistant_parts.append(text)

    with patch.object(rubric_module, "_emit_rubric_event", new=capture_emit):
        result = await middleware._astream_final_grader(
            grader,
            DIRECT_INPUT,
            config={},
            context=None,
        )

    return {
        "counts": counts,
        "reasoning": "".join(reasoning_parts),
        "assistant": "".join(assistant_parts),
        "structured": (
            isinstance(result, dict)
            and result.get("structured_response") is not None
        ),
    }


async def full_parent(*, isolate_grader_context: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(model=model, verifier_tools=[], max_iterations=1)

    # Keep the exact production implementation.
    production_astream_final_grader = MiraRubricMiddleware._astream_final_grader

    if isolate_grader_context:
        async def context_isolated_grader(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            # Preserve ONLY the Rubric event identity/writer that MIRA needs for
            # inspection lifecycle emission.
            rubric_run = rubric_module._RUBRIC_RUN.get()

            clean_context = contextvars.Context()
            if rubric_run is not None:
                clean_context.run(rubric_module._RUBRIC_RUN.set, rubric_run)

            # Run the UNCHANGED production grader coroutine in that clean context.
            coro = production_astream_final_grader(
                self,
                grader,
                state,
                config=config,
                context=context,
            )
            task = asyncio.create_task(coro, context=clean_context)
            return await task

        middleware._astream_final_grader = MethodType(
            context_isolated_grader,
            middleware,
        )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name=(
            "rubric_high_level_isolated_probe"
            if isolate_grader_context
            else "rubric_high_level_current_probe"
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


def show_direct(result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("1) DIRECT BASELINE — PREVIOUSLY KNOWN-GOOD PRODUCTION GRADER METHOD")
    print("=" * 100)
    print("inspection delta counts:", dict(result["counts"]))
    print("structured_response:", result["structured"])
    print("reasoning chars:", len(result["reasoning"]))
    print("assistant chars:", len(result["assistant"]))
    print("\nREAL GRADER REASONING:")
    print(short(result["reasoning"]) if result["reasoning"] else "<none>")
    print("\nREAL GRADER ASSISTANT:")
    print(short(result["assistant"]) if result["assistant"] else "<none>")


def show_parent(label: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print("grader deltas on parent stream.custom:", dict(result["counts"]))
    print("real rubric evaluations:", len(result["evaluations"]))
    if result["evaluations"]:
        print("evaluation result:", result["evaluations"][-1].get("result"))
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

    print("=" * 100)
    print("MIRA HIGH-LEVEL REAL-LLM RUBRIC STREAMING PROBE")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print()
    print("Starting from the earlier direct grader probe that already worked.")
    print("No fake model. No fake grader output.")
    print("=" * 100)

    print("\nRunning 1) direct known-good grader...")
    direct = await direct_known_good()

    print("\nRunning 2) full parent with CURRENT MIRA...")
    current = await full_parent(isolate_grader_context=False)

    print("\nRunning 3) full parent with CLEAN grader ContextVar context...")
    isolated = await full_parent(isolate_grader_context=True)

    show_direct(direct)
    show_parent("2) FULL PARENT — CURRENT MIRA", current)
    show_parent("3) FULL PARENT — CONTEXT-ISOLATED CANDIDATE", isolated)

    direct_ok = (
        direct["structured"]
        and bool(direct["reasoning"])
        and bool(direct["assistant"])
    )
    current_bug = (
        not current["reasoning"]
        and not current["assistant"]
        and len(current["evaluations"]) > 0
    )
    isolated_ok = (
        bool(isolated["reasoning"])
        and bool(isolated["assistant"])
        and isolated["counts"]["reasoning"] > 0
        and isolated["counts"]["assistant"] > 0
        and len(isolated["evaluations"]) > 0
    )

    print("\n" + "=" * 100)
    if direct_ok and current_bug and isolated_ok:
        print("PASS — HIGH-LEVEL FIX PROVEN WITH REAL LLM CALLS")
        print(
            "The exact production grader streaming method works directly, fails "
            "only when it inherits the outer parent execution ContextVars, and "
            "works again inside the full parent run when executed in a clean "
            "ContextVar context while preserving MIRA's Rubric event writer."
        )
        print()
        print(
            "This isolates the bug to parent ContextVar inheritance around the "
            "nested grader call. The grader implementation, ProviderStrategy, "
            "normalizer, RubricEventRenderer, and Inspector pipeline do not need "
            "to be redesigned."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not implement the candidate. The three-way differential did not "
            "prove context isolation as the fix."
        )
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
