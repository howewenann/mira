"""LIVE Rubric verifier offloaded-message read-coverage A/B probe for MIRA.

Put this in tests/probes beside probe_rubric_offload_read_coverage.py:

    python tests/probes/probe_rubric_offload_read_coverage_live.py
    python tests/probes/probe_rubric_offload_read_coverage_live.py --runs 3
    python tests/probes/probe_rubric_offload_read_coverage_live.py --natural
    python tests/probes/probe_rubric_offload_read_coverage_live.py --grade
    python tests/probes/probe_rubric_offload_read_coverage_live.py --profile lmstudio-gemma-4-12b

Default: two arms (unguarded and guarded), with the first REAL LLM response
replaced by an early VERIFICATION_COMPLETE in both arms. Every recovery model
call is real. This guarantees the guard is exercised without waiting for an
unpredictable natural verifier failure. --natural disables this intervention.

Uses MIRA's actual MiraRubricMiddleware._ensure_verifier(),
MiraRubricMiddleware._verifier_input(), _astream_verifier(), and
_verification_evidence(). Does NOT run the main agent or alter its source.
--grade optionally invokes MIRA's actual structured final grader as well.

Uses production offloading and its authoritative evaluation-local manifest.
The guarded arm runs the production nested verifier and read-coverage guard.
The unguarded baseline omits only the guard. Native StateBackend reads keep
synthetic source evidence out of project files and saved sessions.

PASS means the verifier successfully fetched every required line. It does NOT
prove that the LLM understood the content or that the final rubric passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import (  # noqa: E402
    assigned_profile_name,
    get_llm,
    get_profile_model,
)
from agent.rubric.middleware import MiraRubricMiddleware  # noqa: E402
from agent.rubric.offload import EVIDENCE_TOOL, VerifierReadCoverageGuard  # noqa: E402
from config.loader import load_config  # noqa: E402
from probe_rubric_offload_read_coverage import (  # noqa: E402
    FILES,
    MANIFEST,
    coverage_gaps,
    PARENT_STATE,
)

FORCED_STOP = "VERIFICATION_COMPLETE"


class UnguardedReadCoverageGuard(VerifierReadCoverageGuard):
    """A/B baseline: disable only the production completion check."""

    def after_agent(self, state: Any, runtime: Any) -> None:
        return None


def text_of(message: Any) -> str:
    if message is None:
        return ""
    try:
        return str(message.text)
    except Exception:
        return str(getattr(message, "content", ""))


def metadata_summary(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


class RealVerifierRecorder(AgentMiddleware):
    """Records exact model requests; optionally forces only the first stop.

    The FIRST call still reaches the real provider. Its response is recorded,
    then replaced with a terminal AIMessage to simulate the failure. All later
    provider responses are passed through, unmodified.
    """

    def __init__(self, *, force_first_stop: bool, max_model_calls: int) -> None:
        self.force_first_stop = force_first_stop
        self.max_model_calls = max_model_calls
        self.calls: list[dict[str, Any]] = []
        self.provider_seconds = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        idx = len(self.calls) + 1
        if idx > self.max_model_calls:
            raise RuntimeError(f"Verifier exceeded {self.max_model_calls} real model calls")
        sysmsg = getattr(request, "system_message", None)
        current_sys = text_of(sysmsg)
        history = list(getattr(request, "messages", []) or [])
        record = {
            "index": idx,
            "nudge": "VERIFIER READ-COVERAGE NUDGE" in current_sys,
            "history_messages": len(history),
            "request_is_system_scoped": bool(sysmsg),
            "offload_paths_in_user_prompt": [
                p for p in FILES if any(p in text_of(m) for m in history if isinstance(m, HumanMessage))
            ],
            "raw_response": "",
            "raw_tool_calls": [],
            "forced": False,
        }
        self.calls.append(record)
        started = time.perf_counter()
        response = await handler(request)  # REAL configured model call.
        self.provider_seconds += time.perf_counter() - started
        generated = list(getattr(response, "result", []) or [])
        last = next((m for m in reversed(generated) if isinstance(m, AIMessage)), None)
        record["raw_response"] = text_of(last)[:450]
        record["raw_tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args", {})}
            for c in (last.tool_calls or [])
        ] if last else []
        in_tokens, out_tokens = metadata_summary(last)
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        if idx == 1 and self.force_first_stop:
            record["forced"] = True
            return AIMessage(content=FORCED_STOP)
        return response


def verifier_input() -> dict[str, Any]:
    """Exercise the real deterministic production offloader and manifest."""
    return MiraRubricMiddleware._verifier_input(PARENT_STATE, iteration=0)


def observed_reads(messages: list[Any]) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls or []:
            if call.get("name") != EVIDENCE_TOOL:
                continue
            id_ = str(call.get("id") or "")
            calls[id_] = {
                "id": id_,
                "path": (call.get("args") or {}).get("file_path"),
                "offset": (call.get("args") or {}).get("offset", 0),
                "limit": (call.get("args") or {}).get("limit", 500),
                "result": "missing",
            }
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call = calls.get(str(msg.tool_call_id or ""))
        if call is None:
            continue
        call["result"] = "error" if msg.status == "error" else "returned"
        call["output_excerpt"] = text_of(msg)[:180]
    return list(calls.values())


async def run_case(*, cfg: dict[str, Any], profile: str | None,
                   variant: str, round_num: int, forced: bool,
                   max_model_calls: int, grade: bool,
                   timeout: float, shared_input: dict[str, Any]) -> dict[str, Any]:
    model = get_profile_model(cfg, profile) if profile else get_llm(cfg, role="rubric")
    recorder = RealVerifierRecorder(force_first_stop=forced, max_model_calls=max_model_calls)
    middleware = MiraRubricMiddleware(
        model=model,
        verifier_middleware=[recorder],
        max_iterations=1,
    )
    if variant == "unguarded":
        with patch("agent.rubric.middleware.VerifierReadCoverageGuard", UnguardedReadCoverageGuard):
            nested = middleware._ensure_verifier(offloaded=True)
    else:
        nested = middleware._ensure_verifier(offloaded=True)
    # Keep the nonce and entire initial prompt identical in both A/B arms.
    payload = {
        "messages": [msg.model_copy(deep=True) for msg in shared_input["messages"]],
        "offloaded_message_manifest": [dict(i) for i in MANIFEST],
        "files": shared_input["files"],
    }
    started = time.perf_counter()
    result: dict[str, Any] = {}
    error = ""
    grader_report: dict[str, Any] | None = None
    try:
        result = await asyncio.wait_for(
            middleware._astream_verifier(nested, payload, config={}, context=None),
            timeout=timeout,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    messages = list(result.get("messages") or [])
    reads = observed_reads(messages)
    gaps = coverage_gaps(messages, MANIFEST, FILES)
    complete = not gaps and not error
    evidence = middleware._verification_evidence(result) if result else []

    if grade and not error:
        # The default is verifier-only: the production guard is NOT a grader.
        # Optionally call the actual MIRA structured grader and show that it
        # receives paired verifier tool evidence, as production would.
        grader_payload = middleware._nested_grader_input(
            PARENT_STATE, iteration=0, correction=None, evidence=evidence,
        )
        try:
            gstarted = time.perf_counter()
            grader_result = await asyncio.wait_for(
                middleware._astream_final_grader(
                    middleware._ensure_final_grader(),
                    grader_payload, config={}, context=None,
                ), timeout=timeout,
            )
            parsed = middleware._extract_graded(grader_result)
            grader_report = {
                "result": str(getattr(parsed, "result", "")),
                "explanation": str(getattr(parsed, "explanation", ""))[:450],
                "seconds": round(time.perf_counter() - gstarted, 2),
                "evidence_messages": len(evidence),
            }
        except Exception as exc:
            grader_report = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "variant": variant,
        "round": round_num,
        "forced_first_stop": forced,
        "covered": complete,
        "failed_closed": bool(error and "offload-read coverage incomplete" in error),
        "error": error,
        "missing_lines": gaps,
        "read_file_calls": len(reads),
        "reads": reads,
        "paired_verification_messages": len(evidence),
        "model_calls": len(recorder.calls),
        "nudge_visible_by_call": [m["nudge"] for m in recorder.calls],
        "forced_calls": sum(bool(m["forced"]) for m in recorder.calls),
        "original_first_response": recorder.calls[0]["raw_response"] if recorder.calls else "",
        "original_first_calls": recorder.calls[0]["raw_tool_calls"] if recorder.calls else [],
        "final_verifier_response": text_of(messages[-1])[:350] if messages else "",
        "provider_seconds": round(recorder.provider_seconds, 2),
        "elapsed_seconds": round(elapsed, 2),
        "provider_input_tokens": recorder.input_tokens,
        "provider_output_tokens": recorder.output_tokens,
        "grader": grader_report,
        "request_log": recorder.calls,
    }


def show(record: dict[str, Any]) -> None:
    flag = "COVERED" if record["covered"] else ("FAIL-CLOSED" if record["failed_closed"] else "INCOMPLETE")
    print(f"\n[{record['variant'].upper()} / ROUND {record['round']}] {flag}")
    print(f"  model calls           : {record['model_calls']}")
    print(f"  elapsed/provider sec  : {record['elapsed_seconds']} / {record['provider_seconds']}")
    print(f"  nudge in model calls  : {record['nudge_visible_by_call']}")
    print(f"  attempted reads       : {record['read_file_calls']}")
    print(f"  paired evidence msgs  : {record['paired_verification_messages']}")
    print(f"  missing source lines  : {record['missing_lines']}")
    print(f"  natural first response: {record['original_first_response'][:220]!r}")
    if record["original_first_calls"]:
        print(f"  natural first tools   : {record['original_first_calls']}")
    for read in record["reads"]:
        print(f"  read_file             : {read['path']} "
              f"offset={read['offset']} limit={read['limit']} => {read['result']}")
    if record["error"]:
        print(f"  error                 : {record['error'][:500]}")
    if record["grader"] is not None:
        print(f"  actual grader         : {record['grader']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="A/B rounds; default 1")
    parser.add_argument("--profile", default=None, help="Optional named MIRA model profile; default Rubric role (inherits Main)")
    parser.add_argument("--max-nudges", type=int, choices=[2], default=2, help="Production correction cap (2)")
    parser.add_argument("--timeout", type=float, default=180, help="Per verifier/grader timeout in seconds")
    parser.add_argument("--max-model-calls", type=int, default=18, help="Prevent runaway verifier loops; default 18")
    parser.add_argument("--natural", action="store_true", help="Do not force the first verifier stop; measure organic model behavior")
    parser.add_argument("--grade", action="store_true", help="Also run real MIRA final grader after verifier returns")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path (writes only on request)")
    args = parser.parse_args()
    if args.runs < 1 or args.max_nudges < 1 or args.timeout <= 0 or args.max_model_calls < 1:
        parser.error("--runs, --max-nudges, --timeout and --max-model-calls must be positive")

    cfg = load_config(ROOT)
    label = args.profile or assigned_profile_name(cfg, "rubric") or "<unconfigured>"
    print("MIRA LIVE RUBRIC OFFLOAD READ-COVERAGE A/B")
    print(f"Rubric verifier model : {label}")
    print(f"Mode                  : {'NATURAL' if args.natural else 'FORCED FIRST STOP'}")
    print("Verifier              : MIRA's real nested Rubric verifier + real model")
    print("Manifest              : deterministic, synthetic; 2 virtual offloaded messages")
    print("No real project files are read/written. Guarded arm uses the production guard.")
    records: list[dict[str, Any]] = []
    for round_num in range(1, args.runs + 1):
        shared_input = verifier_input()  # Same exact prompt for this pair.
        order = ("unguarded", "guarded") if round_num % 2 else ("guarded", "unguarded")
        for variant in order:
            print(f"\nRunning {variant} / round {round_num}...", flush=True)
            record = await run_case(
                cfg=cfg, profile=args.profile, variant=variant,
                round_num=round_num, forced=not args.natural,
                max_model_calls=args.max_model_calls,
                grade=args.grade, timeout=args.timeout, shared_input=shared_input,
            )
            records.append(record)
            show(record)

    print("\n" + "=" * 74)
    print("A/B COVERAGE SUMMARY (successful, linked read_file results for all source lines)")
    for variant in ("unguarded", "guarded"):
        group = [r for r in records if r["variant"] == variant]
        covered = sum(bool(r["covered"]) for r in group)
        avg_calls = sum(r["model_calls"] for r in group) / len(group)
        avg_time = sum(r["elapsed_seconds"] for r in group) / len(group)
        print(f"  {variant:<10}: covered {covered}/{len(group)}; "
              f"avg calls {avg_calls:.2f}; avg seconds {avg_time:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report: {args.output}")
    if any((not r["covered"] or r["error"] or (r["grader"] and "error" in r["grader"])) for r in records if r["variant"] == "guarded"):
        raise SystemExit("FAIL: production verifier did not retrieve all required evidence or grading failed")
    print("\nImportant: A forced first stop is synthetic; subsequent recovery actions "
          "are performed by the real LLM. --natural tests organic behavior. "
          "Even full read coverage does not prove model comprehension or rubric satisfaction.")


if __name__ == "__main__":
    asyncio.run(main())
