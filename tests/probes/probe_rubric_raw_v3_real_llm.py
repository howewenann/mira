"""REAL-LLM proof probe for the Rubric grader Inspector fix.

Place in:
    tests/probes/probe_rubric_raw_v3_real_llm.py

Run FROM the probe directory:
    python probe_rubric_raw_v3_real_llm.py

THIS MAKES REAL MODEL CALLS using MIRA's configured RUBRIC model.
There are no fake models and no mocked grader output.

The real execution is:

    real parent agent LLM call
        -> real verifier LLM call
        -> real ProviderStrategy GraderResponse grader LLM call

The only probe-only code is an observation-only LangGraph v3 transformer that
implements the proposed production fix at the parent stream boundary:

    raw v3 nested rubric_grader messages
        -> existing rubric_inspection_delta
        -> existing RubricEventRenderer
        -> existing RubricInspectionProjector
        -> existing LiveInspectionStore

PASS requires BOTH:
  1. the real grader still returns a real structured Rubric result; and
  2. the resulting Grader inspection contains real reasoning + real assistant
     output captured from that same grader call.
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
from langgraph.stream import ProtocolEvent, StreamTransformer

from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL
from core.execution.inspection.live import LiveInspectionStore
from core.execution.streams.messages import streamed_message_deltas
from core.execution.streams.provider import event_delta
from core.execution.streams.rubric import (
    RUBRIC_GRADING_END,
    RUBRIC_GRADING_START,
    RUBRIC_INSPECTION_DELTA,
    RubricEventRenderer,
)

REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


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


class RealRubricGraderRawMessageProjector(StreamTransformer):
    """Candidate production fix, exercised against the real live model run."""

    required_stream_modes = ("custom", "messages")

    def __init__(
        self,
        scope: tuple[str, ...],
        rubric_renderer: RubricEventRenderer,
        stats: Counter[str],
    ) -> None:
        super().__init__(scope)
        self.rubric_renderer = rubric_renderer
        self.stats = stats
        self.grading_run_id = ""
        self.iteration = 0

    def init(self) -> dict[str, Any]:
        # Observation only. Do not add/replace any graph state or projection.
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        # Factories are cloned into child muxes. Only the root observer acts.
        # Root raw v3 sees nested namespaces as protocol events.
        if self.scope:
            return True

        method = str(event.get("method") or "")
        params = event.get("params") or {}

        # Feed the ACTUAL Rubric lifecycle events from the live run through
        # MIRA's existing renderer/projector.
        if method == "custom":
            value = params.get("data")
            if not isinstance(value, dict):
                return True

            event_type = str(value.get("type") or "")
            if not event_type.startswith("rubric_"):
                return True

            self.stats[f"custom:{event_type}"] += 1

            if event_type == RUBRIC_GRADING_START:
                self.grading_run_id = str(value.get("grading_run_id") or "")
                self.iteration = int(value.get("iteration") or 0)

            self.rubric_renderer.handle(value)

            if event_type == RUBRIC_GRADING_END:
                self.grading_run_id = ""

            return True

        if method != "messages":
            return True

        data = params.get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            return True

        payload, metadata = data
        if not isinstance(metadata, dict):
            return True

        # This is the REAL nested grader created by MiraRubricMiddleware.
        if str(metadata.get("lc_agent_name") or "") != "rubric_grader":
            return True

        self.stats["raw_grader_message_events"] += 1
        self.stats[f"grader_namespace_depth:{len(params.get('namespace') or [])}"] += 1

        if not self.grading_run_id:
            self.stats["grader_event_before_grading_start"] += 1
            return True

        reasoning = ""
        text = ""

        # v3 protocol event generated by the REAL provider/model stream.
        if isinstance(payload, dict) and "event" in payload:
            lifecycle = str(payload.get("event") or "")
            self.stats[f"protocol:{lifecycle}"] += 1

            delta = event_delta(payload)
            delta_type = str(delta.get("type") or "").lower()

            if delta_type in {
                "reasoning-delta",
                "reasoning_delta",
                "thinking-delta",
                "thinking_delta",
            }:
                reasoning = str(
                    delta.get("reasoning") or delta.get("text") or ""
                )
            elif delta_type in {"text-delta", "text_delta"}:
                text = str(delta.get("text") or "")

        # Finalized/legacy fallback from the same REAL run.
        else:
            reasoning, text = streamed_message_deltas((payload, metadata))

        if reasoning:
            self.stats["grader_reasoning_deltas"] += 1
            self.stats["grader_reasoning_chars"] += len(reasoning)
            self.rubric_renderer.handle(
                {
                    "type": RUBRIC_INSPECTION_DELTA,
                    "grading_run_id": self.grading_run_id,
                    "iteration": self.iteration,
                    "phase": "grader",
                    "kind": "reasoning",
                    "text": reasoning,
                }
            )

        if text:
            self.stats["grader_assistant_deltas"] += 1
            self.stats["grader_assistant_chars"] += len(text)
            self.rubric_renderer.handle(
                {
                    "type": RUBRIC_INSPECTION_DELTA,
                    "grading_run_id": self.grading_run_id,
                    "iteration": self.iteration,
                    "phase": "grader",
                    "kind": "assistant",
                    "text": text,
                }
            )

        # Never alter/suppress the real protocol event.
        return True


def short(value: str, limit: int = 1000) -> str:
    value = value.replace("\r", "")
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    # REAL configured MIRA Rubric model.
    model = get_llm(config, role=RUBRIC_MODEL)
    model_name = get_rubric_model_name(config)

    # REAL MIRA Rubric middleware: real verifier + real structured grader.
    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    # Minimal real parent agent so the Rubric middleware runs in its actual
    # parent-agent LangGraph context.
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name="rubric_raw_v3_real_llm_probe",
    )

    renderer = ProbeRenderer()
    rubric_renderer = RubricEventRenderer(
        renderer,
        max_iterations=1,
        grader_model=model_name,
    )
    stats: Counter[str] = Counter()

    def transformer_factory(
        scope: tuple[str, ...],
    ) -> RealRubricGraderRawMessageProjector:
        return RealRubricGraderRawMessageProjector(
            scope,
            rubric_renderer,
            stats,
        )

    print("=" * 96)
    print("MIRA REAL-LLM Rubric grader Inspector fix proof")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {model_name}")
    print()
    print("REAL CALLS: parent agent -> verifier -> structured grader")
    print("No fake model. No mocked grader response.")
    print("=" * 96)

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[transformer_factory],
    )

    # This drives the ACTUAL graph and ACTUAL model calls to completion.
    final_state = await run.output()

    print("\nREAL GRAPH FINAL STATE")
    if isinstance(final_state, dict):
        print("keys:", sorted(final_state.keys()))
        evaluations = list(final_state.get("_rubric_evaluations") or [])
    else:
        print("unexpected final state:", type(final_state).__name__)
        evaluations = []

    print(f"real rubric evaluations: {len(evaluations)}")
    for evaluation in evaluations:
        print(
            "  result=",
            evaluation.get("result"),
            " criteria=",
            len(evaluation.get("criteria") or []),
        )

    print("\nRAW V3 REAL-GRADER COUNTS")
    for key in sorted(stats):
        print(f"{key:50} {stats[key]}")

    run_ids = {
        str(evaluation.get("grading_run_id") or "")
        for evaluation in evaluations
        if evaluation.get("grading_run_id")
    }
    if not run_ids:
        run_ids = {
            str(evaluation.get("grading_run_id") or "")
            for evaluation in rubric_renderer.evaluations
            if evaluation.get("grading_run_id")
        }

    print("\nREAL GRADER INSPECTION PRODUCED BY CANDIDATE FIX")

    grader_ok = False
    for run_id in sorted(run_ids):
        inspection_id = f"rubric:{run_id}:0:grader"
        inspection = renderer.live_inspections.get(inspection_id)

        print(f"\n{inspection_id}")
        if inspection is None:
            print("  MISSING")
            continue

        kinds = Counter(event.kind for event in inspection.events)
        reasoning = "".join(
            event.text for event in inspection.events
            if event.kind == "reasoning"
        )
        assistant = "".join(
            event.text for event in inspection.events
            if event.kind == "assistant"
        )

        print(f"  status:          {inspection.status}")
        print(f"  kinds:           {dict(kinds)}")
        print(f"  reasoning chars: {len(reasoning)}")
        print(f"  assistant chars: {len(assistant)}")

        print("\n  REAL GRADER REASONING:")
        print(
            "  " + short(reasoning).replace("\n", "\n  ")
            if reasoning else "  <none>"
        )

        print("\n  REAL GRADER ASSISTANT OUTPUT:")
        print(
            "  " + short(assistant).replace("\n", "\n  ")
            if assistant else "  <none>"
        )

        grader_ok = (
            inspection.status == "DONE"
            and bool(reasoning)
            and bool(assistant)
        )

    real_eval_ok = bool(evaluations)
    raw_ok = stats["raw_grader_message_events"] > 0
    reasoning_ok = stats["grader_reasoning_deltas"] > 0
    assistant_ok = stats["grader_assistant_deltas"] > 0
    ordering_ok = stats["grader_event_before_grading_start"] == 0

    print("\n" + "=" * 96)
    if (
        real_eval_ok
        and raw_ok
        and reasoning_ok
        and assistant_ok
        and ordering_ok
        and grader_ok
    ):
        print("PASS — REAL LLM FIX PROVEN")
        print(
            "The real configured Rubric grader completed normally, produced a "
            "real structured evaluation, and its real reasoning + assistant "
            "stream was captured from the parent raw v3 protocol and rendered "
            "through MIRA's existing Rubric Inspector pipeline."
        )
        print()
        print(
            "Production fix direction is proven: observe nested "
            "lc_agent_name='rubric_grader' raw v3 message events at the parent "
            "stream boundary and project them into the existing "
            "rubric_inspection_delta path."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do NOT implement this approach. A real model call ran, but the "
            "candidate fix did not satisfy all proof conditions."
        )
    print("=" * 96)


if __name__ == "__main__":
    asyncio.run(main())
