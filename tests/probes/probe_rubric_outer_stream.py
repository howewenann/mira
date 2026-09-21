"""Probe MIRA Rubric events through the REAL parent-agent outer custom stream.

Place in:
    tests/probes/probe_rubric_outer_stream.py

Run FROM the probe directory:
    python probe_rubric_outer_stream.py

This specifically tests the layer NOT covered by probe_rubric_inspector_stream.py:

    MiraRubricMiddleware
        -> runtime.stream_writer
        -> parent agent astream_events(version="v3")
        -> CustomTransformer / stream.custom
        -> RubricEventRenderer
        -> LiveInspectionStore
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
from core.execution.inspection.live import LiveInspectionStore
from core.execution.streams.rubric import RubricEventRenderer


class ProbeRenderer:
    def __init__(self) -> None:
        self.live_inspections = LiveInspectionStore()

    def rubric_evaluation_started(self, *args: Any, **kwargs: Any) -> None:
        pass

    def rubric_lifecycle_event(self, event: dict[str, Any]) -> None:
        pass

    def rubric_evaluation_finished(self, *args: Any, **kwargs: Any) -> None:
        pass

    def rubric_evaluation_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def rubric_evaluations_cancelled(self) -> None:
        pass


def short(text: str, n: int = 320) -> str:
    value = repr(text)
    return value if len(value) <= n else value[:n] + "...<truncated>"


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    rubric = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    # Minimal parent agent: same LangChain agent mechanism MIRA ultimately uses,
    # but no unrelated MIRA tools/subagents/UI.
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[rubric],
        system_prompt="Answer the user's request directly.",
        name="rubric_outer_stream_probe",
    )

    request = "Reply with exactly: STREAM_OK. Two plus two equals four."
    criteria = (
        "1. The final response contains the exact token STREAM_OK.\n"
        "2. The final response states that two plus two equals four."
    )

    print("=" * 88)
    print("MIRA Rubric OUTER custom-stream probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 88)

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=request)],
            "rubric": criteria,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    renderer = ProbeRenderer()
    rubric_renderer = RubricEventRenderer(
        renderer,
        max_iterations=1,
        grader_model=get_rubric_model_name(config),
    )

    counts: Counter[str] = Counter()
    grading_deltas: Counter[str] = Counter()
    verifier_deltas: Counter[str] = Counter()
    samples: list[tuple[str, str, str]] = []

    async def consume_custom() -> None:
        async for event in run.custom:
            if not isinstance(event, dict):
                counts["<non-dict>"] += 1
                continue

            event_type = str(event.get("type") or "<missing>")
            counts[event_type] += 1

            if event_type == "rubric_inspection_delta":
                phase = str(event.get("phase") or "")
                kind = str(event.get("kind") or "")
                text = str(event.get("text") or "")
                if phase == "grader":
                    grading_deltas[kind] += 1
                elif phase == "verifier":
                    verifier_deltas[kind] += 1

                if len(samples) < 12:
                    samples.append((phase, kind, text))

            rubric_renderer.handle(event)

    output_holder: dict[str, Any] = {}

    async def consume_output() -> None:
        output_holder["value"] = await run.output()

    await asyncio.gather(consume_custom(), consume_output())

    print("\nOUTER stream.custom COUNTS")
    for key, value in counts.items():
        print(f"{key:36} {value}")

    print("\nINSPECTION DELTAS SEEN ON OUTER stream.custom")
    print(f"verifier: {dict(verifier_deltas)}")
    print(f"grader:   {dict(grading_deltas)}")

    if samples:
        print("\nFIRST INSPECTION-DELTA SAMPLES")
        for phase, kind, text in samples:
            print(f"- {phase:8} {kind:10} {short(text)}")

    # Discover the one run-id emitted by the real middleware.
    run_ids: set[str] = set()
    # RubricEventRenderer keeps completed evaluations when the full end event arrives.
    for evaluation in rubric_renderer.evaluations:
        run_id = str(evaluation.get("grading_run_id") or "")
        if run_id:
            run_ids.add(run_id)

    print("\nPROJECTED LiveInspectionStore")
    if not run_ids:
        print("No completed rubric run id found.")
    for run_id in sorted(run_ids):
        for phase in ("verifier", "grader"):
            inspection_id = f"rubric:{run_id}:0:{phase}"
            inspection = renderer.live_inspections.get(inspection_id)
            print(f"\n{inspection_id}")
            if inspection is None:
                print("  MISSING")
                continue
            kinds = Counter(event.kind for event in inspection.events)
            print(f"  status: {inspection.status}")
            print(f"  kinds:  {dict(kinds)}")
            for event in inspection.events[-5:]:
                print(f"  - {event.kind:10} {short(event.text)}")

    print("\n" + "=" * 88)
    if grading_deltas:
        print(
            "DIAGNOSIS: grader reasoning/assistant deltas DO reach the parent "
            "stream.custom. If the TUI still loses them, the bug is in MIRA's "
            "runner/UI integration after this stream boundary."
        )
    elif verifier_deltas and not grading_deltas:
        print(
            "DIAGNOSIS: reproduced the bug below the TUI: verifier deltas reach "
            "the parent custom stream but grader deltas do not. Focus on "
            "_emit_rubric_event / grader nested-stream scoping."
        )
    else:
        print(
            "DIAGNOSIS: neither phase produced outer inspection deltas. "
            "Send the full output; custom-stream scoping is the prime suspect."
        )
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
