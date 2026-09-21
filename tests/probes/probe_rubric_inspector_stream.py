"""Probe the real MIRA grader -> Rubric Inspector capture path.

Place in tests/probes and run FROM that directory:

    python probe_rubric_inspector_stream.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from langchain_core.messages import HumanMessage

import agent.rubric.middleware as rubric_middleware_module
from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL
from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.rubric import (
    inspection_event_values,
    rubric_inspection_id,
    rubric_inspection_title,
)
from core.execution.streams.messages import streamed_message_deltas as real_streamed_message_deltas
from core.execution.streams.output import message_text
from core.execution.streams.rubric import RubricEventRenderer


RUN_ID = "probe-grader-inspector"
ITERATION = 0


class ProbeRenderer:
    def __init__(self) -> None:
        self.live_inspections = LiveInspectionStore()

    def rubric_lifecycle_event(self, event: dict[str, Any]) -> None:
        pass


def short(value: Any, n: int = 220) -> str:
    text = repr(value)
    return text if len(text) <= n else text[:n] + "...<truncated>"


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(model=model, verifier_tools=[], max_iterations=1)
    grader = middleware._ensure_final_grader()

    grader_input = {
        "messages": [
            HumanMessage(
                content=(
                    "Evaluate this diagnostic case.\n"
                    "Criteria:\n"
                    "1. Response contains STREAM_OK.\n"
                    "2. Response says two plus two equals four.\n\n"
                    "Response:\nSTREAM_OK. Two plus two equals four.\n\n"
                    "Return the normal structured GraderResponse and give a "
                    "reasonably detailed explanation."
                )
            )
        ]
    }

    renderer = ProbeRenderer()
    rubric = RubricEventRenderer(
        renderer,
        max_iterations=1,
        grader_model=get_rubric_model_name(config),
    )
    inspection_id = rubric_inspection_id(RUN_ID, ITERATION, "grader")
    rubric.handle(
        {
            "type": "rubric_grading_start",
            "grading_run_id": RUN_ID,
            "iteration": ITERATION,
            "inspection_id": inspection_id,
            "inspection_title": rubric_inspection_title(ITERATION, "grader"),
            "inspection_events": inspection_event_values(grader_input["messages"]),
        }
    )

    stats: Counter[str] = Counter()
    dropped: list[dict[str, str]] = []

    def diagnostic_deltas(value: Any) -> tuple[str, str]:
        message = value[0] if isinstance(value, tuple) and value else value
        metadata = value[1] if isinstance(value, tuple) and len(value) > 1 else {}

        content = getattr(message, "content", None)
        blocks = getattr(message, "content_blocks", None)
        kwargs = getattr(message, "additional_kwargs", None)

        stats["messages"] += 1
        if content not in (None, "", [], {}):
            stats["raw_nonempty_content"] += 1
        if message_text(message):
            stats["message_text_nonempty"] += 1

        raw_reasoning = False
        if isinstance(blocks, (list, tuple)):
            raw_reasoning = any(
                isinstance(block, dict)
                and str(block.get("type") or "").lower() in {"reasoning", "thinking"}
                and bool(block.get("reasoning") or block.get("text"))
                for block in blocks
            )
        if isinstance(kwargs, dict):
            raw_reasoning = raw_reasoning or any(
                kwargs.get(k) for k in ("reasoning", "reasoning_content", "thinking")
            )
        if raw_reasoning:
            stats["raw_reasoning"] += 1

        reasoning, text = real_streamed_message_deltas(value)
        if reasoning:
            stats["normalized_reasoning"] += 1
            stats["normalized_reasoning_chars"] += len(reasoning)
        if text:
            stats["normalized_assistant"] += 1
            stats["normalized_assistant_chars"] += len(text)

        if (
            (content not in (None, "", [], {}) or raw_reasoning)
            and not reasoning
            and not text
        ):
            stats["raw_but_normalized_empty"] += 1
            if len(dropped) < 6:
                dropped.append(
                    {
                        "message_type": type(message).__name__,
                        "content": short(content),
                        "content_blocks": short(blocks),
                        "additional_kwargs": short(kwargs),
                        "metadata": short(metadata),
                    }
                )
        return reasoning, text

    def emit(event_type: str, **values: Any) -> None:
        stats[f"emitted:{event_type}"] += 1
        rubric.handle(
            {
                "type": event_type,
                "grading_run_id": RUN_ID,
                "iteration": ITERATION,
                **values,
            }
        )

    print("=" * 84)
    print("MIRA Rubric grader -> Inspector stream probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 84)

    with (
        patch.object(rubric_middleware_module, "streamed_message_deltas", new=diagnostic_deltas),
        patch.object(rubric_middleware_module, "_emit_rubric_event", new=emit),
    ):
        result = await middleware._astream_final_grader(
            grader,
            grader_input,
            config={},
            context=None,
        )

    rubric.handle(
        {
            "type": "rubric_grading_end",
            "grading_run_id": RUN_ID,
            "iteration": ITERATION,
            "inspection_id": inspection_id,
            "succeeded": True,
        }
    )

    inspection = renderer.live_inspections.get(inspection_id)
    structured = result.get("structured_response") if isinstance(result, dict) else None

    print("\nCOUNTS")
    for key in (
        "messages",
        "raw_nonempty_content",
        "raw_reasoning",
        "message_text_nonempty",
        "normalized_reasoning",
        "normalized_reasoning_chars",
        "normalized_assistant",
        "normalized_assistant_chars",
        "raw_but_normalized_empty",
        "emitted:rubric_inspection_delta",
    ):
        print(f"{key:34} {stats[key]}")

    print("\nFINAL")
    print(f"structured_response found: {structured is not None}")
    print(f"inspection found:          {inspection is not None}")

    if inspection is not None:
        kinds = Counter(event.kind for event in inspection.events)
        print(f"inspection status:         {inspection.status}")
        print(f"inspection kinds:          {dict(kinds)}")
        print("\nINSPECTION TAIL")
        for event in inspection.events[-8:]:
            print(f"- {event.kind:10} {short(event.text, 300)}")

    if dropped:
        print("\nRAW EVENTS DROPPED BY CURRENT NORMALIZER")
        for i, sample in enumerate(dropped, 1):
            print(f"\n--- sample {i} ---")
            for key, value in sample.items():
                print(f"{key}: {value}")

    print("\n" + "=" * 84)
    if stats["raw_nonempty_content"] and not stats["normalized_assistant"]:
        print("DIAGNOSIS: raw assistant content exists but streamed_message_deltas() drops it.")
    elif stats["raw_reasoning"] and not stats["normalized_reasoning"]:
        print("DIAGNOSIS: raw reasoning exists but streamed_message_deltas() drops it.")
    elif inspection is not None and any(
        event.kind in {"assistant", "reasoning"} for event in inspection.events
    ):
        print(
            "DIAGNOSIS: the real middleware -> RubricEventRenderer -> "
            "LiveInspectionStore path works in isolation. Look above this layer "
            "at outer stream delivery / TUI timing."
        )
    else:
        print("DIAGNOSIS: inconclusive; send the full output.")
    print("=" * 84)


if __name__ == "__main__":
    asyncio.run(main())
