"""REAL-LLM proof for the concrete Rubric grader streaming fix.

Place in:
    tests/probes/probe_rubric_grader_clear_inherited_callbacks.py

Run FROM the probe directory:
    python probe_rubric_grader_clear_inherited_callbacks.py

This makes real calls to MIRA's configured Rubric model.

It runs TWO real parent-agent Rubric evaluations:

A) CURRENT MIRA
   Uses the production `_astream_final_grader` unchanged.

B) CANDIDATE FIX
   Uses the same real grader, same ProviderStrategy(GraderResponse), same input,
   same outer Rubric lifecycle, same `_emit_inspection_message`, same values
   handling — but passes `callbacks=[]` explicitly to the nested grader
   `astream()` call.

Why this exact change:
`langchain_core.runnables.config.ensure_config()` inherits the outer parent
RunnableConfig from a ContextVar when a nested config omits a key. MIRA's
`_grader_invocation_config()` returns metadata only, so the nested grader
implicitly inherits the parent v3 streaming callbacks. Explicit `callbacks=[]`
is the smallest way to test whether those inherited callbacks are what suppress
the grader's own `stream_mode="messages"` projection.

PASS requires the candidate run to have:
  - real grader messages > 0
  - real reasoning > 0
  - real assistant output > 0
  - structured_response preserved
  - grader rubric_inspection_delta events visible on the parent custom stream
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path
from types import MethodType
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
from core.execution.streams.messages import streamed_message_deltas

REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


async def run_case(clear_inherited_callbacks: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    nested = Counter()
    parent = Counter()
    structured_found = False
    samples: list[str] = []

    if clear_inherited_callbacks:
        async def grader_with_isolated_callbacks(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            nonlocal structured_found

            # THE ONLY CANDIDATE CHANGE:
            # explicitly override inherited parent callbacks.
            isolated_config = {
                **config,
                "callbacks": [],
            }

            result: dict[str, Any] = {}

            async for mode, value in grader.astream(
                state,
                config=isolated_config,
                context=context,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    nested["messages"] += 1

                    reasoning, text = streamed_message_deltas(value)
                    if reasoning:
                        nested["reasoning"] += 1
                        nested["reasoning_chars"] += len(reasoning)
                    if text:
                        nested["assistant"] += 1
                        nested["assistant_chars"] += len(text)

                    # EXACT current MIRA inspection emission.
                    self._emit_inspection_message("grader", value)

                elif mode == "values" and isinstance(value, dict):
                    nested["values"] += 1
                    result = value
                    if value.get("structured_response") is not None:
                        structured_found = True

            return result

        middleware._astream_final_grader = MethodType(
            grader_with_isolated_callbacks,
            middleware,
        )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name="rubric_callback_isolation_probe",
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    evaluations: list[dict[str, Any]] = []

    async def consume_custom() -> None:
        async for event in run.custom:
            if not isinstance(event, dict):
                continue

            if (
                event.get("type") == "rubric_inspection_delta"
                and str(event.get("phase") or "") == "grader"
            ):
                kind = str(event.get("kind") or "")
                parent[kind] += 1
                if len(samples) < 6:
                    samples.append(
                        f"{kind}: {str(event.get('text') or '')[:180]!r}"
                    )

    async def consume_output() -> None:
        output = await run.output()
        if isinstance(output, dict):
            evaluations.extend(output.get("_rubric_evaluations") or [])

    await asyncio.gather(
        consume_custom(),
        consume_output(),
    )

    return {
        "nested": nested,
        "parent": parent,
        "structured_found": structured_found,
        "evaluations": evaluations,
        "samples": samples,
    }


def show(label: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    print("nested grader stream counts:", dict(result["nested"]))
    print("structured_response seen:", result["structured_found"])
    print("real rubric evaluations:", len(result["evaluations"]))
    if result["evaluations"]:
        print("evaluation result:", result["evaluations"][-1].get("result"))
    print("grader deltas on parent stream.custom:", dict(result["parent"]))

    if result["samples"]:
        print("\nfirst real grader deltas reaching the parent:")
        for sample in result["samples"]:
            print("-", sample)


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    print("=" * 96)
    print("MIRA REAL-LLM nested grader callback-isolation proof")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print()
    print("Both cases make real parent/verifier/grader model calls.")
    print("=" * 96)

    print("\nRunning A) CURRENT MIRA...")
    current = await run_case(False)

    print("\nRunning B) CANDIDATE FIX: explicit callbacks=[] on nested grader...")
    fixed = await run_case(True)

    show("A) CURRENT MIRA", current)
    show("B) CANDIDATE FIX", fixed)

    current_parent = sum(current["parent"].values())
    fixed_parent = sum(fixed["parent"].values())

    print("\n" + "=" * 96)
    if (
        current_parent == 0
        and fixed["nested"]["messages"] > 0
        and fixed["nested"]["reasoning"] > 0
        and fixed["nested"]["assistant"] > 0
        and fixed["structured_found"]
        and len(fixed["evaluations"]) > 0
        and fixed_parent > 0
        and fixed["parent"]["reasoning"] > 0
        and fixed["parent"]["assistant"] > 0
    ):
        print("PASS — REAL LLM FIX PROVEN")
        print(
            "Explicitly clearing the inherited parent callbacks restores the "
            "nested grader's real message stream while preserving its real "
            "structured_response and Rubric evaluation. The existing "
            "_emit_inspection_message path then delivers real grader reasoning "
            "and assistant output to the parent custom stream."
        )
        print()
        print(
            "Root cause: nested RunnableConfig callback inheritance from the "
            "outer v3 run interferes with the grader's own messages-mode stream."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not implement this change. The candidate did not restore all "
            "required real grader outputs."
        )
    print("=" * 96)


if __name__ == "__main__":
    asyncio.run(main())
