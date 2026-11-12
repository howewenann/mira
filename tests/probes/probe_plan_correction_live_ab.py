"""LIVE A/B probe for MIRA Plan response-status corrections.

Run from the repository root with the MIRA environment active:

    python tests/probes/probe_plan_correction_live_ab.py
    python tests/probes/probe_plan_correction_live_ab.py --runs 3
    python tests/probes/probe_plan_correction_live_ab.py --profile lmstudio-gemma-4-12b

This runs the configured REAL MIRA model in both arms. The model's *first*
response is deliberately overridden with the same invalid Plan status in both
arms, after the live model has been called. Everything thereafter uses the real
model and MIRA's actual response-status rule and retry middleware.

Compares a probe-only legacy HumanMessage delivery adapter with production
transient system guidance. Both use the production validation, counters and
events. The deterministic companion probe tests production behavior directly.

No MIRA production modules or project files are modified. `read_file` is a
read-only, in-memory probe tool; the file content and marker are not provided
to the model before it calls the tool. Requires internet/local model access
according to your normal MIRA configuration. This is not a proof of improved
model reliability; run several rounds and compare tool use and final accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import get_llm, get_model_name, get_profile_model  # noqa: E402
from agent.middleware.correction import CORRECTION_SOURCE, CorrectionMiddleware  # noqa: E402
from agent.middleware.file_references import FileReferenceMiddleware  # noqa: E402
from agent.middleware.planning_stage_enforcement import PlanningStageEnforcementMiddleware  # noqa: E402
from agent.planning.policy import plan_system_prompt  # noqa: E402
from agent.planning.response_status import PlanningResponseStatusRule  # noqa: E402
from config.loader import load_config  # noqa: E402

PATH = "/probe_note.txt"
MARKER = "MIRA_LIVE_CORRECTION_7B91"
USER_REQUEST = (
    "In Plan mode, please read @probe_note.txt and report the exact TOKEN value "
    "contained in the file. This is a read-only inspection, NOT a request to "
    "create or prepare a Plan. Do not guess the token."
)
FORCED_INVALID_RESPONSE = (
    "I still need to inspect the referenced file before answering.\n"
    "RESPONSE_STATUS: NEEDS_RESEARCH"
)


class LegacyMessageCorrection(CorrectionMiddleware):
    """Reproduce the old delivery only, sharing production correction decisions.

    This baseline exists solely for the live A/B comparison. No independent
    validation or retry loop is maintained here.
    """

    def _after_agent_update(self, state: Any, runtime: Any) -> Any:
        update = super()._after_agent_update(state, runtime)
        if update and update.get("jump_to") == "model":
            prompt = next(iter(update["_correction_pending"].values()))
            update["_correction_pending"] = {}
            update["messages"] = [HumanMessage(
                content=prompt, name=CORRECTION_SOURCE,
                additional_kwargs={"lc_source": CORRECTION_SOURCE},
            )]
        elif update and update.get("messages"):
            # Preserve the old exhaustion diagnostic as well as its final text.
            rule = self._active_rules(state)[0]
            decision = rule.inspect(state["messages"][-1], state)
            update["messages"].insert(0, HumanMessage(
                content=(f"Correction check failed: {decision.failed_check}\n"
                         "No further retry will be attempted because the limit of "
                         f"{self.max_retries} was reached."),
                name=CORRECTION_SOURCE, additional_kwargs={"lc_source": CORRECTION_SOURCE},
            ))
        return update


@tool("read_file")
def read_probe_file(file_path: str) -> str:
    """Read the exact virtual file path specified by the user; do not guess its contents."""
    if "/" + file_path.replace("\\", "/").lstrip("/") != PATH:
        return f"ERROR: file not found: {file_path!r}. Available file is {PATH}."
    return f"# Probe-only virtual file\nTOKEN={MARKER}\nEnd of file."


def message_text(message: Any) -> str:
    try:
        return str(message.text)
    except Exception:
        return str(getattr(message, "content", ""))


def system_text(request: Any) -> str:
    # `system_message` includes guidance injected by upstream wrap_model_call.
    system = getattr(request, "system_message", None)
    return message_text(system) if system is not None else ""


class LiveRequestRecorderAndFault(AgentMiddleware):
    """Observe real prompts and replace ONLY the first provider result.

    This is a deliberately controlled failure, not an LLM-generated mistake.
    The next and all subsequent model results are returned unmodified.
    """

    def __init__(self) -> None:
        self.model_calls = 0
        self.actual_model_responses: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.wall_seconds = 0.0

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.model_calls += 1
        index = self.model_calls
        visible_system = system_text(request)
        history = list(request.messages)
        self.requests.append({
            "call": index,
            "file_guidance": PATH in visible_system,
            "status_contract": "RESPONSE_STATUS:" in visible_system,
            "correction_in_system": "Your previous response" in visible_system,
            "correction_in_history": any(
                isinstance(m, HumanMessage) and getattr(m, "name", "") == CORRECTION_SOURCE
                for m in history
            ),
            "history_messages": len(history),
            "system_chars": len(visible_system),
        })
        started = time.monotonic()
        response = await handler(request)  # REAL MODEL, including call 1.
        self.wall_seconds += time.monotonic() - started
        messages = list(getattr(response, "result", []) or [])
        last = messages[-1] if messages else None
        self.actual_model_responses.append(message_text(last)[:240] if last else "<no model response>")
        if index == 1:
            # Same guaranteed invalid initial response in BOTH A/B arms.
            # Returning AIMessage directly is supported by LangChain middleware.
            return AIMessage(content=FORCED_INVALID_RESPONSE)
        return response


async def run_one(
    *,
    config: dict[str, Any],
    profile: str | None,
    variant: str,
    round_num: int,
    max_retries: int,
) -> dict[str, Any]:
    real_model = get_profile_model(config, profile) if profile else get_llm(config)
    recorder = LiveRequestRecorderAndFault()
    rule = PlanningResponseStatusRule(workflow="plan")
    if variant == "transient":
        correction = CorrectionMiddleware(rules=(rule,), max_retries=max_retries)
    else:
        correction = LegacyMessageCorrection(rules=(rule,), max_retries=max_retries)

    graph = create_agent(
        model=real_model,
        tools=[read_probe_file],
        system_prompt=plan_system_prompt(),
        middleware=[
            FileReferenceMiddleware(),
            PlanningStageEnforcementMiddleware(),
            correction,
            recorder,
        ],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": f"live-correction-{variant}-{round_num}"}}
    started = time.monotonic()
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=USER_REQUEST)],
        "planning_stage": "plan_research",
    }, config=cfg)
    elapsed = time.monotonic() - started
    assert recorder.model_calls >= 2, (
        "Correction did not trigger a second real model invocation"
    )
    checkpoint = (await graph.aget_state(cfg)).values
    msgs = list(checkpoint.get("messages") or [])
    correction_messages = [
        m for m in msgs
        if isinstance(m, HumanMessage) and getattr(m, "name", "") == CORRECTION_SOURCE
    ]
    all_tool_calls = [
        c for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])
    ]
    read_calls = [c for c in all_tool_calls if c.get("name") == "read_file"]
    completed_ids = {
        str(m.tool_call_id) for m in msgs
        if isinstance(m, ToolMessage)
        and m.status != "error"
        and MARKER in message_text(m)
    }
    successful_read = any(
        str(c.get("id")) in completed_ids
        and "/" + str((c.get("args") or {}).get("file_path", "")).replace("\\", "/").lstrip("/") == PATH
        for c in read_calls
    )
    final = message_text(msgs[-1]) if msgs else ""
    last_two = recorder.requests[:2]
    # Exact first-message behavior must match known deterministic baseline.
    assert last_two and last_two[0]["file_guidance"], (
        "The first request lost the @file guidance; check middleware ordering"
    )
    if len(last_two) > 1:
        expected_guidance = variant == "transient"
        assert last_two[1]["file_guidance"] == expected_guidance, (
            f"Unexpected file-reference guidance on retry ({variant}): {last_two[1]}"
        )
    if variant == "transient":
        assert last_two[1]["correction_in_system"], "Retry missed transient correction"
        assert not correction_messages, "Correction leaked into checkpoint history"
        assert not checkpoint.get("_correction_pending"), "Completed correction was not cleared"
    return {
        "variant": variant,
        "round": round_num,
        "model_calls": recorder.model_calls,
        "elapsed_seconds": round(elapsed, 2),
        "provider_seconds": round(recorder.wall_seconds, 2),
        "read_file_calls": len(read_calls),
        "successful_read": successful_read,
        "reported_token": MARKER in final,
        "valid_complete_marker": final.rstrip().endswith("RESPONSE_STATUS: COMPLETE"),
        "checkpoint_correction_messages": len(correction_messages),
        "checkpoint_messages": len(msgs),
        "file_guidance_by_call": [r["file_guidance"] for r in recorder.requests],
        "correction_in_history_by_call": [r["correction_in_history"] for r in recorder.requests],
        "correction_in_system_by_call": [r["correction_in_system"] for r in recorder.requests],
        "raw_first_response_overridden": recorder.actual_model_responses[0] if recorder.actual_model_responses else "",
        "final": final,
        "requests": recorder.requests,
    }


def print_case(x: dict[str, Any]) -> None:
    ok = x["successful_read"] and x["reported_token"] and x["valid_complete_marker"]
    print(f"\n[{x['variant'].upper()} / ROUND {x['round']}] {'PASS' if ok else 'INCOMPLETE'}")
    print(f"  real model calls       : {x['model_calls']}")
    print(f"  elapsed / provider sec : {x['elapsed_seconds']} / {x['provider_seconds']}")
    print(f"  @file system guidance  : {x['file_guidance_by_call']}")
    print(f"  correction HumanMsgs   : {x['checkpoint_correction_messages']}")
    print(f"  checkpoint messages    : {x['checkpoint_messages']}")
    print(f"  correction in system   : {x['correction_in_system_by_call']}")
    print(f"  read_file calls        : {x['read_file_calls']}")
    print(f"  successful read        : {x['successful_read']}")
    print(f"  token in final         : {x['reported_token']}")
    print(f"  valid COMPLETE status  : {x['valid_complete_marker']}")
    print(f"  final                  : {x['final'][:500]!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Number of paired rounds (default: 3)")
    parser.add_argument("--profile", type=str, default=None, help="Optional MIRA model profile (default: configured Main)")
    parser.add_argument("--max-retries", type=int, default=2, help="Correction limit per run (default: 2)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if args.runs < 1 or args.max_retries < 1:
        parser.error("--runs and --max-retries must be positive")

    config = load_config(ROOT)
    label = args.profile or get_model_name(config)
    print("MIRA LIVE PLAN CORRECTION A/B (real model; forced first response)")
    print(f"Model: {label}")
    print("IMPORTANT: The first model's output is replaced with an intentionally invalid status")
    print("in BOTH variants. The recovery is generated by the real, configured LLM.")
    print("No repository files are changed; read_file is a virtual read-only probe tool.")
    results: list[dict[str, Any]] = []
    for round_num in range(1, args.runs + 1):
        # Alternate which variant goes first to reduce warm-cache/order bias.
        variants = ("legacy", "transient") if round_num % 2 else ("transient", "legacy")
        for variant in variants:
            print(f"\nRunning {variant} / round {round_num} ...", flush=True)
            try:
                record = await run_one(
                    config=config,
                    profile=args.profile,
                    variant=variant,
                    round_num=round_num,
                    max_retries=args.max_retries,
                )
            except Exception as exc:
                print(f"ERROR: {variant} round {round_num}: {type(exc).__name__}: {exc}", flush=True)
                raise
            results.append(record)
            print_case(record)

    print("\n" + "=" * 74)
    print("A/B SUMMARY (success = successful read AND correct token AND COMPLETE)")
    for variant in ("legacy", "transient"):
        subset = [r for r in results if r["variant"] == variant]
        wins = sum(r["successful_read"] and r["reported_token"] and r["valid_complete_marker"] for r in subset)
        print(f"  {variant:<10}: success {wins}/{len(subset)}; "
              f"avg model calls {sum(r['model_calls'] for r in subset)/len(subset):.2f}; "
              f"avg wall sec {sum(r['elapsed_seconds'] for r in subset)/len(subset):.2f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Detailed JSON report: {args.output}")
    print("\nNOTE: The deterministic failure guarantees a correction; live model behavior")
    print("after that failure is real. A small number of rounds is not statistically")
    print("conclusive. No changes to MIRA production middleware were made.")


if __name__ == "__main__":
    asyncio.run(main())
