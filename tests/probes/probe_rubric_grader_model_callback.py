"""HIGH-LEVEL REAL-LLM proof: capture grader tokens at the model callback.

Place in:
    tests/probes/probe_rubric_grader_model_callback.py

Run:
    python probe_rubric_grader_model_callback.py

This starts from the earlier successful fact: the real grader model DOES stream
reasoning + assistant chunks.

Two real full-parent runs are compared:

A) CURRENT MIRA
   Unchanged production path.

B) CANDIDATE
   Same production `_astream_final_grader()` unchanged, but one additional
   callback is attached ONLY to the grader invocation. The callback observes
   the real ChatGenerationChunk objects from the real model call and projects
   their reasoning/text into MIRA's existing `rubric_inspection_delta` events.

No fake model.
No mocked grader output.
No alternate stream versions.
No raw LangGraph event parsing.
No subagent machinery.
No ProviderStrategy changes.
No prompt/input changes.

PASS requires:
- current run still reproduces zero grader inspection deltas;
- candidate run has real reasoning + real assistant deltas;
- the real structured Rubric evaluation still completes.
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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import ensure_config
from langgraph.stream.transformers import CustomTransformer

import agent.rubric.middleware as rubric_module
from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL
from core.execution.inspection.rubric import rubric_inspection_id
from core.execution.streams.messages import streamed_message_deltas

REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


def short(text: str, limit: int = 900) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class GraderInspectionCallback(BaseCallbackHandler):
    """Observe the real grader model chunks and emit existing MIRA events."""

    run_inline = True

    def __init__(self, identity: tuple[str, int, Any], stats: Counter[str]) -> None:
        self.run_id = identity[0]
        self.iteration = identity[1]
        self.writer = identity[2]
        self.stats = stats
        self._seen_chunks: set[int] = set()

    def _emit(self, kind: str, text: str) -> None:
        if not text or not callable(self.writer):
            return

        self.stats[f"callback_{kind}"] += 1
        self.stats[f"callback_{kind}_chars"] += len(text)

        self.writer(
            {
                "type": "rubric_inspection_delta",
                "grading_run_id": self.run_id,
                "iteration": self.iteration,
                "phase": "grader",
                "kind": kind,
                "text": text,
                "inspection_id": rubric_inspection_id(
                    self.run_id,
                    self.iteration,
                    "grader",
                ),
            }
        )

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any = None,
        **kwargs: Any,
    ) -> None:
        self.stats["callback_on_llm_new_token"] += 1

        if chunk is None:
            self.stats["callback_no_chunk"] += 1
            return

        # Some model adapters can invoke the callback both inside their own
        # _astream() and again in BaseChatModel. The same ChatGenerationChunk
        # object is reused, so object identity safely prevents probe duplication.
        chunk_identity = id(chunk)
        if chunk_identity in self._seen_chunks:
            self.stats["callback_duplicate_chunk"] += 1
            return
        self._seen_chunks.add(chunk_identity)

        message = getattr(chunk, "message", None)
        if message is None:
            self.stats["callback_no_message"] += 1
            return

        reasoning, text = streamed_message_deltas(message)

        if reasoning:
            self._emit("reasoning", reasoning)
        if text:
            self._emit("assistant", text)


async def run_case(*, add_grader_callback: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    callback_stats: Counter[str] = Counter()
    original = MiraRubricMiddleware._astream_final_grader

    if add_grader_callback:
        async def instrumented_grader(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            identity = rubric_module._RUBRIC_RUN.get()
            if identity is None:
                callback_stats["missing_rubric_identity"] += 1
                return await original(
                    self,
                    grader,
                    state,
                    config=config,
                    context=context,
                )

            observer = GraderInspectionCallback(identity, callback_stats)

            # Preserve the grader's full effective config, including existing
            # tracing/parent callbacks, and ADD one observer.
            instrumented_config = ensure_config(config)
            callbacks = instrumented_config.get("callbacks")

            if callbacks is None:
                instrumented_config["callbacks"] = [observer]
            elif isinstance(callbacks, list):
                instrumented_config["callbacks"] = [*callbacks, observer]
            else:
                manager = callbacks.copy()
                manager.add_handler(observer, inherit=True)
                instrumented_config["callbacks"] = manager

            # Crucially: call MIRA's unchanged production grader streamer.
            return await original(
                self,
                grader,
                state,
                config=instrumented_config,
                context=context,
            )

        middleware._astream_final_grader = MethodType(
            instrumented_grader,
            middleware,
        )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name=(
            "rubric_grader_callback_candidate"
            if add_grader_callback
            else "rubric_grader_callback_current"
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

    parent_counts: Counter[str] = Counter()
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

            parent_counts[kind] += 1
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
        "callback_stats": callback_stats,
        "parent_counts": parent_counts,
        "reasoning": "".join(reasoning_parts),
        "assistant": "".join(assistant_parts),
        "evaluations": evaluations,
    }


def show(label: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    print("callback stats:", dict(result["callback_stats"]))
    print("grader deltas on parent stream.custom:", dict(result["parent_counts"]))
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

    print("=" * 100)
    print("MIRA HIGH-LEVEL REAL-LLM GRADER CALLBACK PROBE")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print()
    print("Both cases make real parent -> verifier -> structured grader calls.")
    print("Candidate only adds one observer to the REAL grader model callback.")
    print("=" * 100)

    print("\nRunning A) CURRENT MIRA...")
    current = await run_case(add_grader_callback=False)

    print("\nRunning B) CANDIDATE — grader model callback...")
    candidate = await run_case(add_grader_callback=True)

    show("A) CURRENT MIRA", current)
    show("B) CANDIDATE — REAL GRADER MODEL CALLBACK", candidate)

    current_bug = (
        not current["reasoning"]
        and not current["assistant"]
        and len(current["evaluations"]) > 0
    )

    candidate_ok = (
        candidate["callback_stats"]["callback_on_llm_new_token"] > 0
        and candidate["callback_stats"]["callback_reasoning"] > 0
        and candidate["callback_stats"]["callback_assistant"] > 0
        and candidate["parent_counts"]["reasoning"] > 0
        and candidate["parent_counts"]["assistant"] > 0
        and bool(candidate["reasoning"])
        and bool(candidate["assistant"])
        and len(candidate["evaluations"]) > 0
    )

    print("\n" + "=" * 100)
    if current_bug and candidate_ok:
        print("PASS — REAL LLM FIX PROVEN")
        print(
            "The current parent run reproduces the missing grader Inspector, "
            "while observing the real grader model callback restores its actual "
            "reasoning + assistant chunks and sends them through MIRA's existing "
            "rubric_inspection_delta path. The real structured Rubric evaluation "
            "still completes normally."
        )
        print()
        print(
            "Production fix direction: make the grader's model-call callback the "
            "authoritative source for live Grader inspection deltas. Keep "
            "ProviderStrategy and the existing grader graph/result handling."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not implement the callback approach. The real model callback did "
            "not satisfy all proof conditions."
        )
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
