"""Compare nested grader streaming APIs inside a real parent MIRA Rubric run.

Place in:
    tests/probes/probe_rubric_grader_stream_versions.py

Run FROM the probe directory:
    python probe_rubric_grader_stream_versions.py

Cases:
  A) current MIRA: grader.astream(..., version="v1" default)
  B) grader.astream(..., version="v2")
  C) grader.astream_events(..., version="v3") using its scoped messages projection

Goal: find the smallest API choice that actually exposes grader reasoning/text
inside the real parent-agent execution context while preserving the final
structured_response.
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
from core.execution.streams.messages import consume_messages

REQUEST = "Reply with exactly: STREAM_OK. Two plus two equals four."
RUBRIC = (
    "1. The final response contains the exact token STREAM_OK.\n"
    "2. The final response states that two plus two equals four."
)


class GraderCapture:
    """Renderer-shaped capture for v3 nested messages."""

    def __init__(self, middleware: MiraRubricMiddleware) -> None:
        self.middleware = middleware
        self.reasoning = 0
        self.assistant = 0

    def reasoning_delta(self, text: str, **_kwargs: Any) -> None:
        self.reasoning += 1
        # Use the exact production Rubric event route.
        import agent.rubric.middleware as mod
        mod._emit_rubric_event(
            "rubric_inspection_delta",
            phase="grader",
            kind="reasoning",
            text=str(text),
            inspection_id=self.middleware._inspection_id("grader"),
        )

    def text_delta(self, text: str, **_kwargs: Any) -> None:
        self.assistant += 1
        import agent.rubric.middleware as mod
        mod._emit_rubric_event(
            "rubric_inspection_delta",
            phase="grader",
            kind="assistant",
            text=str(text),
            inspection_id=self.middleware._inspection_id("grader"),
        )

    def model_stream_finished(self) -> None:
        pass

    # Grader has tools=[]; these keep consume_messages renderer-compatible.
    def tool_call(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def tool_call_delta(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def delegation_started(self, *_args: Any, **_kwargs: Any) -> None:
        pass


async def run_case(mode: str) -> dict[str, Any]:
    config = load_config(REPOSITORY_ROOT)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(model=model, verifier_tools=[], max_iterations=1)

    diagnostics: Counter[str] = Counter()
    message_samples: list[str] = []

    if mode == "v2":
        async def grader_v2(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}
            async for stream_mode, value in grader.astream(
                state,
                config=config,
                context=context,
                stream_mode=["messages", "values"],
                version="v2",
            ):
                diagnostics[f"nested_mode:{stream_mode}"] += 1
                if stream_mode == "messages":
                    diagnostics["nested_messages"] += 1
                    if len(message_samples) < 4:
                        message_samples.append(
                            f"{type(value).__name__}: {repr(value)[:280]}"
                        )
                    self._emit_inspection_message("grader", value)
                elif stream_mode == "values" and isinstance(value, dict):
                    result = value
            diagnostics["structured_response"] = int(
                isinstance(result, dict) and result.get("structured_response") is not None
            )
            return result

        middleware._astream_final_grader = MethodType(grader_v2, middleware)

    elif mode == "v3":
        async def grader_v3(
            self: MiraRubricMiddleware,
            grader: Any,
            state: dict[str, Any],
            *,
            config: dict[str, Any],
            context: object | None,
        ) -> dict[str, Any]:
            run = await grader.astream_events(
                state,
                config=config,
                context=context,
                version="v3",
            )
            capture = GraderCapture(self)

            async def capture_messages() -> None:
                await consume_messages(
                    run.messages,
                    capture,
                    render_normal_tools=False,
                )

            async def capture_output() -> dict[str, Any]:
                value = await run.output()
                return value if isinstance(value, dict) else {}

            result, _ = await asyncio.gather(
                capture_output(),
                capture_messages(),
            )
            diagnostics["nested_reasoning"] = capture.reasoning
            diagnostics["nested_assistant"] = capture.assistant
            diagnostics["structured_response"] = int(
                result.get("structured_response") is not None
            )
            return result

        middleware._astream_final_grader = MethodType(grader_v3, middleware)

    # mode == "v1" leaves current production code untouched.

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        system_prompt="Answer the user's request directly.",
        name=f"grader_stream_version_probe_{mode}",
    )

    run = await agent.astream_events(
        {
            "messages": [HumanMessage(content=REQUEST)],
            "rubric": RUBRIC,
        },
        version="v3",
        transformers=[CustomTransformer],
    )

    parent: Counter[str] = Counter()
    samples: list[str] = []

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
            if len(samples) < 4:
                samples.append(f"{kind}: {str(event.get('text') or '')[:180]!r}")

    async def consume_output() -> None:
        await run.output()

    await asyncio.gather(consume_custom(), consume_output())

    return {
        "diagnostics": diagnostics,
        "parent": parent,
        "samples": samples,
        "nested_samples": message_samples,
    }


async def main() -> None:
    config = load_config(REPOSITORY_ROOT)
    print("=" * 92)
    print("MIRA nested grader streaming-version probe")
    print(f"repo root:    {REPOSITORY_ROOT}")
    print(f"rubric model: {get_rubric_model_name(config)}")
    print("=" * 92)

    results = {}
    for mode in ("v1", "v2", "v3"):
        print(f"\nRunning {mode}...")
        results[mode] = await run_case(mode)

    for mode in ("v1", "v2", "v3"):
        result = results[mode]
        print("\n" + "=" * 92)
        print(f"CASE {mode.upper()}")
        print("=" * 92)
        print("nested diagnostics:", dict(result["diagnostics"]))
        print("grader deltas on parent stream.custom:", dict(result["parent"]))

        if result["nested_samples"]:
            print("nested messages samples:")
            for sample in result["nested_samples"]:
                print("-", sample)

        if result["samples"]:
            print("parent grader delta samples:")
            for sample in result["samples"]:
                print("-", sample)

    print("\n" + "=" * 92)
    print("DIAGNOSIS")
    print("=" * 92)

    v1_parent = sum(results["v1"]["parent"].values())
    v2_parent = sum(results["v2"]["parent"].values())
    v3_parent = sum(results["v3"]["parent"].values())

    if v1_parent == 0 and v2_parent > 0:
        print(
            "PASS: version='v2' is the smallest proven fix. The current nested "
            "v1 message stream is incompatible with this parent v3 execution "
            "context, while v2 restores grader reasoning/text to parent custom."
        )
    elif v1_parent == 0 and v2_parent == 0 and v3_parent > 0:
        print(
            "PASS: nested astream_events(version='v3') is the proven fix. "
            "Both direct astream v1 and v2 fail in this parent context, but the "
            "scoped v3 messages projection exposes the grader stream correctly."
        )
    elif v1_parent == 0 and v2_parent > 0 and v3_parent > 0:
        print(
            "Both v2 and v3 work. Prefer v2 if its payloads are correctly "
            "normalized and structured_response remains present; it is the "
            "smaller production change."
        )
    else:
        print(
            "No decisive candidate from these cases. Do not modify production "
            "streaming based on guesswork; inspect the counts above."
        )

    print("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
