"""Opt-in live streaming smoke test for DeepAgents event timing.

This script intentionally avoids SessionStore/SessionRecorder so it does not
write normal .mira/_sessions files. It needs the configured model endpoint
running, for example LM Studio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.factory import build_agent  # noqa: E402
from config.loader import load_config  # noqa: E402
from config.metadata import infer_model_metadata  # noqa: E402
from runtime.runner import run_turn  # noqa: E402
from session.checkpoint import make_checkpointer  # noqa: E402
from session.dashboard import token_counter_for_model  # noqa: E402

DEFAULT_PROMPT = "use 2 subagents to tell me 2 100 word different stories, one scary, one funny"


class SmokeRenderer:
    """Minimal renderer that records timestamped streaming callbacks."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self._subagent_labels: dict[int, str] = {}
        self._subagent_requests: dict[str, str] = {}
        self._blank_subagent_starts: list[str] = []
        self._seen_text = False
        self._seen_reasoning = False

    def startup_progress(self, state: str) -> None:
        self._record("startup", state=state)

    def waiting_started(self) -> None:
        self._record("waiting_started")

    def waiting_finished(self) -> None:
        self._record("waiting_finished")

    def reasoning_delta(self, delta: str) -> None:
        if delta and not self._seen_reasoning:
            self._seen_reasoning = True
            self._record("reasoning", chars=len(delta), sample=sample(delta))

    def text_delta(self, delta: str) -> None:
        if delta and not self._seen_text:
            self._seen_text = True
            self._record("text", chars=len(delta), sample=sample(delta))

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self._record("tool_call", name=name, call_id=call_id, args=args)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        self._record("tool_result", name=name, call_id=call_id, chars=len(result or ""))

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        self._record("delegation_delta", calls=simplify_calls(calls))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self._record("delegation_started", calls=simplify_calls(calls))

    def start_subagent_live(self) -> None:
        self._record("subagents_live_started")

    def stop_subagent_live(self) -> None:
        self._record("subagents_live_stopped")

    def subagent_label(self, subagent: Any) -> str:
        key = id(subagent)
        if key not in self._subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self._subagent_labels[key] = f"{name} [{len(self._subagent_labels) + 1}]"
        return self._subagent_labels[key]

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        if not task_input:
            self._blank_subagent_starts.append(subagent)
        else:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_started", name=subagent, request=task_input)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        if task_input:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_request_updated", name=subagent, request=task_input)

    def subagent_finished(self, subagent: str, result: str = "", task_input: str = "") -> None:
        if task_input:
            self._subagent_requests[subagent] = task_input
        self._record("subagent_finished", name=subagent, chars=len(result or ""), request=task_input)

    def subagent_cancelled(self, subagent: str, result: str = "", task_input: str = "") -> None:
        self._record("subagent_cancelled", name=subagent, chars=len(result or ""), request=task_input)

    def subagents_cancelled(self) -> None:
        self._record("subagents_cancelled")

    def tick_subagents(self) -> None:
        return None

    def finish_main(self) -> None:
        return None

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        raise RuntimeError(f"Smoke prompt unexpectedly requested approvals: {interrupts!r}")

    async def ask_user(self, interrupt: Any) -> str:
        raise RuntimeError(f"Smoke prompt unexpectedly asked the user: {interrupt!r}")

    def _record(self, kind: str, **payload: Any) -> None:
        event = {"t": round(time.monotonic() - self.started_at, 3), "kind": kind, **payload}
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=True, default=str), flush=True)


def simplify_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified = []
    for call in calls:
        args = call.get("args", {}) if isinstance(call, dict) else {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        args = args if isinstance(args, dict) else {"raw": str(args)}
        simplified.append(
            {
                "id": str(call.get("id") or call.get("call_id") or ""),
                "name": str(call.get("name") or call.get("tool_name") or ""),
                "subagent_type": str(args.get("subagent_type") or ""),
                "description": str(args.get("description") or ""),
            }
        )
    return simplified


def sample(text: str, limit: int = 80) -> str:
    normalized = " ".join(str(text).split())
    return normalized[:limit]


def assert_streaming_shape(renderer: SmokeRenderer, strict_start_requests: bool) -> None:
    kinds = [event["kind"] for event in renderer.events]
    activity_indices = [
        index
        for index, kind in enumerate(kinds)
        if kind in {"delegation_started", "subagent_started", "subagent_request_updated", "subagent_finished"}
    ]
    if not activity_indices:
        raise AssertionError("No task/subagent activity was observed.")

    text_indices = [index for index, kind in enumerate(kinds) if kind == "text"]
    if text_indices and min(activity_indices) > min(text_indices):
        raise AssertionError("Task/subagent activity started only after assistant text.")

    started = [event for event in renderer.events if event["kind"] == "subagent_started"]
    if not started:
        raise AssertionError("No subagent_started events were observed.")
    missing_final_requests = [event["name"] for event in started if not renderer._subagent_requests.get(event["name"])]
    if missing_final_requests:
        raise AssertionError(f"Subagents still have blank requests: {missing_final_requests}")
    if strict_start_requests:
        blank_starts = [event["name"] for event in started if not event.get("request")]
        if blank_starts:
            raise AssertionError(f"subagent_started events had blank initial requests: {blank_starts}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live DeepAgents streaming smoke test.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--direct", action="store_true", help="Use direct local HTTP client settings from MIRA.")
    parser.add_argument(
        "--strict-start-requests",
        action="store_true",
        help="Fail if a subagent start is initially blank, even if it is patched later.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    print(f"[smoke] workspace={workspace}", flush=True)
    config = load_config(workspace)
    config["llm_direct"] = bool(args.direct)
    print(
        f"[smoke] provider={config.get('llm_provider')} model={config.get('llm_model')} direct={config.get('llm_direct')}",
        flush=True,
    )
    print("[smoke] inferring model metadata...", flush=True)
    metadata = await infer_model_metadata(config)
    config["llm_inferred_context_tokens"] = metadata.context_tokens
    config["llm_context_source"] = metadata.context_source
    print(
        f"[smoke] metadata context={metadata.context_tokens} source={metadata.context_source}",
        flush=True,
    )
    print("[smoke] building agent...", flush=True)
    agent = build_agent(config=config, workspace=workspace, checkpointer=make_checkpointer(), metadata=metadata)
    renderer = SmokeRenderer()
    print(f"[smoke] running prompt={args.prompt!r}", flush=True)
    result = await asyncio.wait_for(
        run_turn(
            agent=agent,
            text=args.prompt,
            renderer=renderer,
            thread_id=f"stream-smoke-{uuid.uuid4()}",
            token_counter=token_counter_for_model(),
        ),
        timeout=args.timeout,
    )
    renderer._record("done", final_chars=len(result.final_text or ""))
    assert_streaming_shape(renderer, strict_start_requests=args.strict_start_requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
