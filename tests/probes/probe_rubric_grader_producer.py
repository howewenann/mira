"""Pinpoint why live grader deltas disappear in a full parent-agent run.

Place in:
    tests/probes/probe_rubric_grader_producer.py

Run FROM the probe directory:
    python probe_rubric_grader_producer.py

The probe runs two real parent-agent Rubric evaluations:

A) BASELINE
   Uses MIRA's current _astream_final_grader unchanged.

B) +CUSTOM MODE
   Changes ONLY the nested grader stream modes from:
       ["messages", "values"]
   to:
       ["custom", "messages", "values"]

For each run it counts four boundaries:

1. grader messages-mode items seen by _emit_inspection_message()
2. rubric_inspection_delta calls made to _emit_rubric_event()
3. whether _RUBRIC_RUN still has a callable captured writer at those calls
4. grader rubric_inspection_delta events visible on the PARENT stream.custom

This tells us whether:
- the grader producer receives no message chunks in the real nested run,
- MIRA produces deltas but the captured writer scopes them away,
- or simply adding nested "custom" mode fixes the current behavior.
"""

from __future__ import annotations

import asyncio
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


REQUEST = "Reply with exactly: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


async def run_case(*, add_custom_mode: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    # Optional candidate: alter ONLY nested grader stream_mode.
    if add_custom_mode:
        async def grader_with_custom(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            if not callable(getattr(type(grader), "astream", None)):
                return await grader.ainvoke(state, config=config, context=context)

            result: dict[str, Any] = {}
            async for mode, value in grader.astream(
                state,
                config=config,
                context=context,
                stream_mode=["custom", "messages", "values"],
            ):
                if mode == "messages":
                    self._emit_inspection_message("grader", value)
                elif mode == "values" and isinstance(value, dict):
                    result = value
                # Grader has no tools/custom payload contract today, so custom
                # items are intentionally drained but not interpreted.
            return result

        middleware._astream_final_grader = MethodType(grader_with_custom, middleware)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Answer the user's request directly.",
        name="rubric_grader_producer_probe",
    )

    producer_calls: Counter[str] = Counter()
    emit_calls: Counter[str] = Counter()
    writer_state: Counter[str] = Counter()
    parent_custom: Counter[str] = Counter()
    samples: list[str] = []

    original_emit_message = MiraRubricMiddleware._emit_inspection_message
    original_emit_event = rubric_module._emit_rubric_event

    def wrapped_emit_message(phase: str, value: Any) -> None:
        producer_calls[phase] += 1
        original_emit_message(phase, value)

    def wrapped_emit_event(event_type: str, **values: Any) -> None:
        phase = str(values.get("phase") or "")
        if event_type == "rubric_inspection_delta":
            emit_calls[phase] += 1
            identity = rubric_module._RUBRIC_RUN.get()
            if identity is None:
                writer_state[f"{phase}:identity_none"] += 1
            elif callable(identity[2]):
                writer_state[f"{phase}:writer_callable"] += 1
            else:
                writer_state[f"{phase}:writer_not_callable"] += 1

            if phase == "grader" and len(samples) < 6:
                samples.append(
                    f"kind={values.get('kind')!r} text={str(values.get('text') or '')[:100]!r}"
                )

        original_emit_event(event_type, **values)

    run = None
    with (
        patch.object(
            MiraRubricMiddleware,
            "_emit_inspection_message",
            new=staticmethod(wrapped_emit_message),
        ),
        patch.object(
            rubric_module,
            "_emit_rubric_event",
            new=wrapped_emit_event,
        ),
    ):
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
                phase = str(event.get("phase") or "")
                kind = str(event.get("kind") or "")
                parent_custom[f"{phase}:{kind}"] += 1

        async def consume_output() -> None:
            await run.output()

        await asyncio.gather(consume_custom(), consume_output())

    return {
        "producer_calls": producer_calls,
        "emit_calls": emit_calls,
        "writer_state": writer_state,
        "parent_custom": parent_custom,
        "samples": samples,
    }


def show(name: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)

    print("\n_emit_inspection_message calls")
    print(dict(result["producer_calls"]))

    print("\nrubric_inspection_delta calls into _emit_rubric_event")
    print(dict(result["emit_calls"]))

    print("\n_RUBRIC_RUN writer state during inspection-delta emission")
    print(dict(result["writer_state"]))

    print("\ninspection deltas actually visible on PARENT stream.custom")
    print(dict(result["parent_custom"]))

    if result["samples"]:
        print("\nfirst grader delta samples produced inside middleware")
        for sample in result["samples"]:
            print("-", sample)


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    print("=" * 92)
    print("MIRA full-run grader producer/scoping probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 92)

    baseline = await run_case(add_custom_mode=False)
    custom_mode = await run_case(add_custom_mode=True)

    show("CASE A — CURRENT MIRA", baseline)
    show('CASE B — ONLY ADD "custom" TO NESTED GRADER stream_mode', custom_mode)

    b_prod = baseline["producer_calls"]["grader"]
    b_emit = baseline["emit_calls"]["grader"]
    b_parent = sum(
        count
        for key, count in baseline["parent_custom"].items()
        if key.startswith("grader:")
    )

    c_prod = custom_mode["producer_calls"]["grader"]
    c_emit = custom_mode["emit_calls"]["grader"]
    c_parent = sum(
        count
        for key, count in custom_mode["parent_custom"].items()
        if key.startswith("grader:")
    )

    print("\n" + "=" * 92)
    print("DIAGNOSIS")
    print("=" * 92)

    if b_prod == 0:
        print(
            "BASELINE: the real nested grader produces no messages-mode items "
            "to MIRA's _emit_inspection_message in this parent-run context."
        )
    elif b_emit == 0:
        print(
            "BASELINE: grader messages reach _emit_inspection_message, but "
            "normalization produces no rubric_inspection_delta events."
        )
    elif b_parent == 0:
        print(
            "BASELINE: MIRA DOES produce grader rubric_inspection_delta events, "
            "but they do not reach the parent stream.custom. This is a writer/"
            "scope-routing problem, not a message-normalization problem."
        )
    else:
        print(
            "BASELINE unexpectedly works here; compare counts with the real TUI."
        )

    if b_parent == 0 and c_parent > 0:
        print(
            '\nPASS CANDIDATE: adding "custom" to the nested grader stream_mode '
            "alone makes grader deltas reach the parent custom stream."
        )
    elif b_parent == 0 and c_prod > b_prod and c_parent == 0:
        print(
            '\nAdding "custom" changes grader message production but STILL does '
            "not route deltas to the parent. Do not use that as the fix."
        )
    elif b_parent == 0 and c_parent == 0:
        print(
            '\nAdding "custom" does NOT fix parent delivery. The next fix must '
            "target writer/scope routing rather than the nested stream modes."
        )
    else:
        print("\nCompare the two cases above before changing production code.")

    print("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
