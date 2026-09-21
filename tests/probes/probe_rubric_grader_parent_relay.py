"""REAL-LLM proof: callback -> queue -> parent-context relay.

Place in:
    tests/probes/probe_rubric_grader_parent_relay.py

Run:
    python probe_rubric_grader_parent_relay.py

This tests the concrete fix suggested by the last successful probe.

Known fact from the previous probe:
- the real grader model callback receives real reasoning + assistant chunks
  inside the full parent MIRA run;
- calling the Rubric stream writer directly from that nested callback context
  does NOT surface them on parent `stream.custom`.

This probe keeps the callback only as a token observer. It pushes normalized
grader deltas into an asyncio.Queue. A relay task is created BEFORE entering
the nested grader stream, so it retains the parent Rubric execution context.
That relay calls MIRA's existing `_emit_rubric_event()`.

Real flow:

    REAL grader model
        -> on_llm_new_token()
        -> asyncio.Queue
        -> parent-context relay task
        -> existing _emit_rubric_event()
        -> parent stream.custom
        -> existing Rubric inspection event contract

No fake model.
No mocked grader output.
No ProviderStrategy changes.
No prompt/input changes.
No stream-version changes.
No raw LangGraph event parsing.
No subagent changes.

PASS requires:
- current MIRA reproduces zero grader inspection deltas;
- candidate callback receives real reasoning + assistant;
- relay forwards real reasoning + assistant to parent stream.custom;
- real structured Rubric evaluation still completes normally.
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
from core.execution.streams.messages import streamed_message_deltas


REQUEST = "Reply exactly with: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)

_DONE = object()


def short(text: str, limit: int = 900) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class QueueingGraderCallback(BaseCallbackHandler):
    """Observe REAL grader chunks and enqueue normalized inspection deltas."""

    run_inline = True

    def __init__(
        self,
        queue: asyncio.Queue[Any],
        stats: Counter[str],
    ) -> None:
        self.queue = queue
        self.stats = stats
        self._seen_chunks: set[int] = set()

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

        # ChatAnyLLM / BaseChatModel can cause the same generation chunk object
        # to be observed twice by callbacks. Avoid duplicate probe output.
        chunk_id = id(chunk)
        if chunk_id in self._seen_chunks:
            self.stats["callback_duplicate_chunk"] += 1
            return
        self._seen_chunks.add(chunk_id)

        message = getattr(chunk, "message", None)
        if message is None:
            self.stats["callback_no_message"] += 1
            return

        reasoning, text = streamed_message_deltas(message)

        if reasoning:
            self.stats["callback_reasoning"] += 1
            self.stats["callback_reasoning_chars"] += len(reasoning)
            self.queue.put_nowait(("reasoning", reasoning))

        if text:
            self.stats["callback_assistant"] += 1
            self.stats["callback_assistant_chars"] += len(text)
            self.queue.put_nowait(("assistant", text))


async def run_case(*, use_parent_relay: bool) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)

    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        max_iterations=1,
    )

    callback_stats: Counter[str] = Counter()
    relay_stats: Counter[str] = Counter()
    original = MiraRubricMiddleware._astream_final_grader

    if use_parent_relay:

        async def grader_with_parent_relay(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            identity = rubric_module._RUBRIC_RUN.get()
            if identity is None:
                relay_stats["missing_rubric_identity"] += 1
                return await original(
                    self,
                    grader,
                    state,
                    config=config,
                    context=context,
                )

            queue: asyncio.Queue[Any] = asyncio.Queue()
            observer = QueueingGraderCallback(queue, callback_stats)

            # IMPORTANT:
            # Create this task NOW, before the nested grader call starts.
            # asyncio.Task captures the current ContextVars at creation time.
            # Therefore all `_emit_rubric_event()` calls below execute with the
            # parent Rubric context rather than the grader model callback context.
            async def relay() -> None:
                while True:
                    item = await queue.get()
                    if item is _DONE:
                        relay_stats["done"] += 1
                        return

                    kind, text = item
                    relay_stats[kind] += 1
                    relay_stats[f"{kind}_chars"] += len(text)

                    rubric_module._emit_rubric_event(
                        "rubric_inspection_delta",
                        phase="grader",
                        kind=kind,
                        text=text,
                        inspection_id=self._inspection_id("grader"),
                    )

            relay_task = asyncio.create_task(relay())

            # Preserve all existing config/tracing callbacks and ADD only our
            # real-model observer.
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

            try:
                # Use MIRA's UNCHANGED production grader streaming method.
                return await original(
                    self,
                    grader,
                    state,
                    config=instrumented_config,
                    context=context,
                )
            finally:
                # Drain every real model delta before the grader phase ends.
                queue.put_nowait(_DONE)
                await relay_task

        middleware._astream_final_grader = MethodType(
            grader_with_parent_relay,
            middleware,
        )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Follow the user's request exactly.",
        name=(
            "rubric_parent_relay_candidate"
            if use_parent_relay
            else "rubric_parent_relay_current"
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
        "relay_stats": relay_stats,
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
    print("relay stats:", dict(result["relay_stats"]))
    print("grader deltas on PARENT stream.custom:", dict(result["parent_counts"]))
    print("real rubric evaluations:", len(result["evaluations"]))

    if result["evaluations"]:
        evaluation = result["evaluations"][-1]
        print("evaluation result:", evaluation.get("result"))
        print("criteria:", len(evaluation.get("criteria") or []))

    print("parent reasoning chars:", len(result["reasoning"]))
    print("parent assistant chars:", len(result["assistant"]))

    if result["reasoning"]:
        print("\nREAL GRADER REASONING RECEIVED BY PARENT:")
        print(short(result["reasoning"]))

    if result["assistant"]:
        print("\nREAL GRADER ASSISTANT RECEIVED BY PARENT:")
        print(short(result["assistant"]))


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)

    print("=" * 100)
    print("MIRA REAL-LLM GRADER PARENT-RELAY PROBE")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print()
    print("A) current MIRA")
    print("B) real grader callback -> queue -> parent-context relay")
    print("=" * 100)

    print("\nRunning A) CURRENT MIRA...")
    current = await run_case(use_parent_relay=False)

    print("\nRunning B) CANDIDATE PARENT RELAY...")
    candidate = await run_case(use_parent_relay=True)

    show("A) CURRENT MIRA", current)
    show("B) CANDIDATE — CALLBACK -> QUEUE -> PARENT RELAY", candidate)

    current_bug = (
        not current["reasoning"]
        and not current["assistant"]
        and len(current["evaluations"]) > 0
    )

    candidate_ok = (
        candidate["callback_stats"]["callback_reasoning"] > 0
        and candidate["callback_stats"]["callback_assistant"] > 0
        and candidate["relay_stats"]["reasoning"] > 0
        and candidate["relay_stats"]["assistant"] > 0
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
            "The real grader callback captured its real reasoning + assistant "
            "chunks; the queue moved them out of the nested model callback "
            "context; the parent-context relay emitted them through MIRA's "
            "existing rubric_inspection_delta channel; and the real structured "
            "Rubric evaluation still completed normally."
        )
        print()
        print(
            "This proves the missing boundary is specifically nested-callback "
            "context -> parent custom stream publication."
        )
    else:
        print("FAIL / INCONCLUSIVE")
        print(
            "Do not implement the relay approach. Expected real callback "
            "reasoning/assistant, relay reasoning/assistant, and matching "
            "reasoning/assistant events on parent stream.custom."
        )

    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
