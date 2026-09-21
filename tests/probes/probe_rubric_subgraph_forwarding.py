"""Proof probe for the proposed Rubric grader inspection fix.

Place this file in:
    tests/probes/probe_rubric_subgraph_forwarding.py

Run FROM the probe directory:
    python probe_rubric_subgraph_forwarding.py

This does NOT modify MIRA.

It reproduces the real parent-agent streaming topology, then simulates exactly
the proposed fix in memory:

    current internal Rubric subgraph drain:
        forward rubric_tool_start / rubric_tool_end

    candidate addition:
        ALSO forward rubric_inspection_delta for phase == "grader"

If the hypothesis is correct, the final Grader LiveInspection will contain:
    user -> reasoning -> assistant(raw structured JSON)

while the root custom stream by itself still contains zero grader deltas.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter, defaultdict
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
from core.execution.streams.rubric import (
    RUBRIC_INSPECTION_DELTA,
    RUBRIC_TOOL_END,
    RUBRIC_TOOL_START,
    RubricEventRenderer,
)

INTERNAL_RUBRIC_GRAPHS = {"rubric_verifier", "rubric_grader"}


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


def short(value: Any, limit: int = 360) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def is_internal_rubric_subgraph(handle: Any) -> bool:
    graph_name = str(getattr(handle, "graph_name", "") or "")
    if graph_name in INTERNAL_RUBRIC_GRAPHS:
        return True

    path = getattr(handle, "path", ())
    return isinstance(path, (list, tuple)) and any(
        str(part).split(":", 1)[0] in INTERNAL_RUBRIC_GRAPHS
        for part in path
    )


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    rubric_middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    # Minimal parent agent. This gives us the real nested rubric_verifier /
    # rubric_grader subgraphs and the same v3 custom/subgraph scoping behavior.
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[rubric_middleware],
        system_prompt="Answer the user's request directly.",
        name="rubric_subgraph_forwarding_probe",
    )

    request = "Reply with exactly: STREAM_OK. Two plus two equals four."
    rubric = (
        "1. The final response contains the exact token STREAM_OK.\n"
        "2. The final response states that two plus two equals four."
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=request)],
            "rubric": rubric,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    renderer = ProbeRenderer()
    projected = RubricEventRenderer(
        renderer,
        max_iterations=1,
        grader_model=get_rubric_model_name(config),
    )

    root_counts: Counter[str] = Counter()
    root_phase_deltas: dict[str, Counter[str]] = defaultdict(Counter)

    child_counts: dict[str, Counter[str]] = defaultdict(Counter)
    child_phase_deltas: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    forwarded_candidate = 0
    child_tasks: list[asyncio.Task[None]] = []

    async def consume_root_custom() -> None:
        async for event in run.custom:
            if not isinstance(event, dict):
                root_counts["<non-dict>"] += 1
                continue

            event_type = str(event.get("type") or "<missing>")
            root_counts[event_type] += 1

            if event_type == RUBRIC_INSPECTION_DELTA:
                phase = str(event.get("phase") or "")
                kind = str(event.get("kind") or "")
                root_phase_deltas[phase][kind] += 1

            # This is exactly what MIRA's root consume_custom_events() does.
            projected.handle(event)

    async def drain_one_child(child: Any) -> None:
        nonlocal forwarded_candidate

        graph_name = str(getattr(child, "graph_name", "") or "<unnamed>")
        label = graph_name

        if not is_internal_rubric_subgraph(child):
            # Still drain output so the handle completes cleanly.
            try:
                await child.output()
            except Exception:
                pass
            return

        async def consume_child_custom() -> None:
            nonlocal forwarded_candidate

            custom = getattr(child, "custom", None)
            if custom is None:
                return

            async for event in custom:
                if not isinstance(event, dict):
                    child_counts[label]["<non-dict>"] += 1
                    continue

                event_type = str(event.get("type") or "<missing>")
                child_counts[label][event_type] += 1

                if event_type == RUBRIC_INSPECTION_DELTA:
                    phase = str(event.get("phase") or "")
                    kind = str(event.get("kind") or "")
                    child_phase_deltas[label][phase][kind] += 1

                # Existing MIRA behavior:
                if event_type in {RUBRIC_TOOL_START, RUBRIC_TOOL_END}:
                    projected.handle(event)
                    continue

                # Candidate fix under test:
                # forward grader inspection deltas from the internal grader
                # subgraph into the existing RubricEventRenderer.
                if (
                    event_type == RUBRIC_INSPECTION_DELTA
                    and str(event.get("phase") or "") == "grader"
                ):
                    projected.handle(event)
                    forwarded_candidate += 1

        async def consume_child_output() -> None:
            try:
                await child.output()
            except Exception:
                pass

        await asyncio.gather(
            consume_child_custom(),
            consume_child_output(),
        )

    async def consume_subgraphs() -> None:
        async for child in run.subgraphs:
            task = asyncio.create_task(drain_one_child(child))
            child_tasks.append(task)

    async def consume_output() -> None:
        await run.output()

    await asyncio.gather(
        consume_root_custom(),
        consume_subgraphs(),
        consume_output(),
    )

    if child_tasks:
        await asyncio.gather(*child_tasks)

    print("=" * 92)
    print("MIRA Rubric subgraph-forwarding proof probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 92)

    print("\nROOT stream.custom")
    for key, value in root_counts.items():
        print(f"{key:38} {value}")

    print("\nROOT inspection deltas by phase")
    for phase, counts in root_phase_deltas.items():
        print(f"{phase:10} {dict(counts)}")

    print("\nINTERNAL RUBRIC SUBGRAPH custom streams")
    if not child_counts:
        print("<no internal Rubric child custom events observed>")
    for graph_name, counts in child_counts.items():
        print(f"\n{graph_name}")
        for key, value in counts.items():
            print(f"  {key:36} {value}")
        phase_counts = child_phase_deltas.get(graph_name, {})
        for phase, counts2 in phase_counts.items():
            print(f"  phase={phase:10} {dict(counts2)}")

    print(f"\nCandidate grader deltas forwarded: {forwarded_candidate}")

    # Find completed run ids from RubricEventRenderer.
    run_ids = {
        str(evaluation.get("grading_run_id") or "")
        for evaluation in projected.evaluations
        if evaluation.get("grading_run_id")
    }

    grader_ok = False

    print("\nPROJECTED INSPECTIONS AFTER IN-MEMORY CANDIDATE FIX")
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

            if phase == "grader":
                grader_ok = (
                    inspection.status == "DONE"
                    and any(e.kind == "reasoning" and e.text for e in inspection.events)
                    and any(e.kind == "assistant" and e.text for e in inspection.events)
                )

            print("  tail:")
            for event in inspection.events[-6:]:
                print(f"    - {event.kind:10} {short(event.text)}")

    print("\n" + "=" * 92)

    root_grader = root_phase_deltas.get("grader", Counter())
    child_grader = sum(
        (
            counts.get("grader", Counter())
            for counts in child_phase_deltas.values()
        ),
        Counter(),
    )

    if (
        not root_grader
        and child_grader
        and forwarded_candidate > 0
        and grader_ok
    ):
        print("PASS")
        print(
            "The hypothesis is confirmed: grader inspection deltas are absent "
            "from the parent custom stream, present on an internal Rubric "
            "subgraph custom stream, and forwarding those deltas into the "
            "existing RubricEventRenderer produces the expected Grader "
            "Inspector reasoning + raw assistant JSON."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Expected: no root grader deltas, grader deltas on an internal "
            "Rubric child, forwarded_candidate > 0, and final grader inspection "
            "containing reasoning + assistant."
        )

    verifier_child = sum(
        (
            counts.get("verifier", Counter())
            for counts in child_phase_deltas.values()
        ),
        Counter(),
    )
    if verifier_child:
        print(
            "\nNOTE: verifier inspection deltas also appeared on an internal "
            "child stream. The proposed production patch must avoid duplicating "
            "events already visible on the root stream; this probe forwards only "
            "phase='grader' deltas for that reason."
        )

    print("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
