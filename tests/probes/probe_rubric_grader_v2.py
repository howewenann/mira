"""Corrected proof probe: current nested grader stream vs LangGraph v2 stream.

Place in:
    tests/probes/probe_rubric_grader_v2.py

Run FROM the probe directory:
    python probe_rubric_grader_v2.py

This probe changes ONE thing in-memory for the second run:
    grader.astream(..., version="v2")

Important: LangGraph v2 `astream()` yields DICTS:
    {"type": "messages"|"values", "ns": ..., "data": ...}
not `(mode, value)` tuples. This probe handles that exact contract.
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


REQUEST = "Reply with exactly: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


async def run_case(use_v2: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(model=model, verifier_tools=[], max_iterations=1)

    nested = Counter()
    parent = Counter()
    structured_found = False
    samples: list[str] = []

    if use_v2:
        async def grader_v2(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            nonlocal structured_found

            result: dict[str, Any] = {}

            async for item in grader.astream(
                state,
                config=config,
                context=context,
                stream_mode=["messages", "values"],
                version="v2",
            ):
                if not isinstance(item, dict):
                    nested["unexpected_non_dict"] += 1
                    continue

                mode = str(item.get("type") or "")
                value = item.get("data")
                nested[mode or "<missing>"] += 1

                if mode == "messages":
                    self._emit_inspection_message("grader", value)

                elif mode == "values" and isinstance(value, dict):
                    result = value
                    if value.get("structured_response") is not None:
                        structured_found = True

            return result

        middleware._astream_final_grader = MethodType(grader_v2, middleware)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Answer the user's request directly.",
        name="rubric_grader_v2_probe",
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    async def consume_custom() -> None:
        async for event in run.custom:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "rubric_inspection_delta":
                continue
            if str(event.get("phase") or "") != "grader":
                continue

            kind = str(event.get("kind") or "")
            parent[kind] += 1
            if len(samples) < 6:
                samples.append(
                    f"{kind}: {str(event.get('text') or '')[:180]!r}"
                )

    async def consume_output() -> None:
        await run.output()

    await asyncio.gather(consume_custom(), consume_output())

    return {
        "nested": nested,
        "parent": parent,
        "structured_found": structured_found,
        "samples": samples,
    }


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    print("=" * 88)
    print("MIRA corrected nested grader v2 proof probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 88)

    print("\nRunning CURRENT MIRA...")
    current = await run_case(False)

    print("\nRunning V2 candidate...")
    v2 = await run_case(True)

    print("\n" + "=" * 88)
    print("CURRENT MIRA")
    print("=" * 88)
    print("grader deltas on parent stream.custom:", dict(current["parent"]))

    print("\n" + "=" * 88)
    print("V2 CANDIDATE")
    print("=" * 88)
    print("nested grader stream item counts:", dict(v2["nested"]))
    print("structured_response seen in values:", v2["structured_found"])
    print("grader deltas on parent stream.custom:", dict(v2["parent"]))

    if v2["samples"]:
        print("\nfirst parent grader deltas:")
        for sample in v2["samples"]:
            print("-", sample)

    print("\n" + "=" * 88)
    current_count = sum(current["parent"].values())
    v2_count = sum(v2["parent"].values())

    if (
        current_count == 0
        and v2["nested"].get("messages", 0) > 0
        and v2["structured_found"]
        and v2_count > 0
        and v2["parent"].get("assistant", 0) > 0
    ):
        print("PASS")
        print(
            "Confirmed: current nested v1 streaming loses the grader message stream "
            "inside the parent v3 run. LangGraph version='v2' exposes real grader "
            "messages, preserves structured_response, and those deltas reach the "
            "parent custom stream. Adding version='v2' is the proven fix."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not change production code. Expected v2 to show nested messages, "
            "structured_response=True, and grader reasoning/assistant deltas on "
            "parent stream.custom."
        )
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
