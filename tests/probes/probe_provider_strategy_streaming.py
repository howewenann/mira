"""Manual probe: does MIRA's ProviderStrategy grader expose a useful live stream?

Run from the MIRA repo root inside the MIRA environment:

    python tests/probes/probe_provider_strategy_streaming.py

This does NOT modify MIRA. It uses the same Rubric model selection and the same
MiraRubricMiddleware grader construction as the current repository.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from langchain_core.messages import HumanMessage

from agent.llm import get_llm, get_rubric_model_name
from agent.rubric.middleware import MiraRubricMiddleware
from config.loader import load_config
from config.settings import RUBRIC_MODEL


def message_from_stream_value(value: Any) -> Any:
    """LangChain messages mode commonly yields (message_chunk, metadata)."""
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


async def main() -> None:
    workspace = Path.cwd()
    config = load_config(workspace)

    # role="rubric" already follows MIRA's Main-inheritance rules when no
    # dedicated Rubric profile is selected.
    rubric_model = get_llm(config, role=RUBRIC_MODEL)

    # This is the exact MIRA class that constructs the final grader with:
    # ProviderStrategy(deepagents_rubric.GraderResponse)
    middleware = MiraRubricMiddleware(
        model=rubric_model,
        verifier_tools=[],
        max_iterations=1,
    )
    grader = middleware._ensure_final_grader()  # intentional probe of MIRA internals

    print("=" * 80)
    print("MIRA ProviderStrategy streaming probe")
    print(f"workspace: {workspace}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 80)

    # Long-ish request on purpose: if the provider really streams structured
    # output, this should give us enough material to observe multiple chunks.
    probe_prompt = """
You are being used for a streaming diagnostic.

Evaluate this simple case:

Rubric:
1. The assistant response must contain the exact token STREAM_OK.
2. The assistant response must state that two plus two equals four.

Assistant response:
STREAM_OK. Two plus two equals four.

This should be graded as satisfied.

For the explanation field, provide a detailed explanation of roughly eight
sentences so there is enough generated content to make streaming behavior easy
to observe. Return the normal structured grader response required by your
response schema.
""".strip()

    started = time.perf_counter()
    message_events = 0
    nonempty_content_events = 0
    reasoning_events = 0
    values_events = 0
    first_message_at: float | None = None
    last_message_at: float | None = None
    structured_response = None

    print("\nStarting grader.astream(..., stream_mode=['messages', 'values'])\n")

    async for mode, value in grader.astream(
        {"messages": [HumanMessage(content=probe_prompt)]},
        stream_mode=["messages", "values"],
    ):
        elapsed = time.perf_counter() - started

        if mode == "messages":
            message_events += 1
            first_message_at = elapsed if first_message_at is None else first_message_at
            last_message_at = elapsed

            message = message_from_stream_value(value)
            content = getattr(message, "content", None)
            content_blocks = getattr(message, "content_blocks", None)
            additional_kwargs = getattr(message, "additional_kwargs", None)

            if nonempty(content):
                nonempty_content_events += 1

            has_reasoning = False
            if isinstance(content_blocks, list):
                has_reasoning = any(
                    isinstance(block, dict)
                    and str(block.get("type") or "").lower() in {"reasoning", "thinking"}
                    for block in content_blocks
                )
            if isinstance(additional_kwargs, dict):
                has_reasoning = has_reasoning or any(
                    nonempty(additional_kwargs.get(key))
                    for key in ("reasoning", "reasoning_content", "thinking")
                )
            if has_reasoning:
                reasoning_events += 1

            print(f"[+{elapsed:7.3f}s] MESSAGES #{message_events}")
            print(f"  type: {type(message).__name__}")
            print(f"  content: {content!r}")
            if content_blocks:
                print(f"  content_blocks: {content_blocks!r}")
            if additional_kwargs:
                print(f"  additional_kwargs: {additional_kwargs!r}")
            print()

        elif mode == "values":
            values_events += 1
            if isinstance(value, dict) and value.get("structured_response") is not None:
                structured_response = value["structured_response"]

            keys = list(value) if isinstance(value, dict) else []
            print(f"[+{elapsed:7.3f}s] VALUES #{values_events} keys={keys}")
            if isinstance(value, dict) and value.get("structured_response") is not None:
                print(f"  structured_response: {value['structured_response']!r}")
            print()

    total = time.perf_counter() - started

    print("=" * 80)
    print("SUMMARY")
    print(f"total time:                 {total:.3f}s")
    print(f"messages events:            {message_events}")
    print(f"non-empty content events:   {nonempty_content_events}")
    print(f"reasoning-bearing events:   {reasoning_events}")
    print(f"values events:              {values_events}")
    print(f"first messages event:       {first_message_at}")
    print(f"last messages event:        {last_message_at}")
    print(f"structured_response found:  {structured_response is not None}")
    print()

    if message_events > 1:
        print("RESULT: useful incremental message streaming is available.")
    elif message_events == 1:
        print(
            "RESULT: messages mode emits data, but only one message event was seen.\n"
            "        That may be final-only rather than useful incremental streaming."
        )
    else:
        print(
            "RESULT: no messages-mode events were exposed. The grader appears final-only\n"
            "        for this model/provider under ProviderStrategy."
        )

    if structured_response is None:
        print(
            "\nWARNING: no final structured_response was observed. That is more important\n"
            "than the streaming result and should be investigated before changing MIRA."
        )

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
