"""Deterministic delivery checks and reasoning-aware local Gemma A/B probe.

    python tests/probes/probe_mode_transition.py
    python tests/probes/probe_mode_transition.py --live-only
    python tests/probes/probe_mode_transition.py --live-only --scenario regression
    python tests/probes/probe_mode_transition.py --live --reference-session <session.json>

Both live arms use ordinary MIRA streaming, SessionRecorder, approvals and mode
handoff in isolated equivalent workspaces. Only the baseline bypasses notice
injection, using a probe-scoped patch. Full recorded events and invocation links
are retained under --output-dir. No production or source session is modified.
Reasoning candidate matches are triage aids, not automated compliance verdicts.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.middleware.mode_transition import MODE_TRANSITION_SOURCE, ModeTransitionMiddleware  # noqa: E402
from config.loader import load_config  # noqa: E402
from config.settings import normalize_settings, save_settings  # noqa: E402
from core.application.app import MiraApplication  # noqa: E402
from core.interface.events import MessageEvent  # noqa: E402
from core.interface.requests import ArtifactDisplayRequest, ArtifactReviewRequest, AskUserRequest  # noqa: E402
from core.execution.runner import run_turn  # noqa: E402
from langgraph.types import Command  # noqa: E402
from tests.test_mode_transition import ApprovalFrontend, ModeTransitionTests  # noqa: E402

REFERENCE = ROOT / ".mira/_sessions/20261113-085837+0800-5bdca89f/session.json"
OUTPUT = ROOT / "tests/probes/mode_transition_evidence"
# The first three prompts are verbatim from the original regression reference.
SCENARIO = (
    ("plan", "dont make a plan, just write a palindrome function to the main directory as a python file"),
    ("act", "write the file now"),
    ("plan", "pls edit the file to include more tests inside. dont make a plan"),
    ("act", "edit the file now to include more tests inside. dont make a plan"),
    ("plan", "pls edit the file to include more tests inside. dont make a plan"),
)
INSPECTION_SCENARIO = tuple(
    (mode, "Read /palindrome.py and briefly explain it. This is read-only inspection; "
     "do not create a formal Plan." if mode == "plan" else
     f"Append the comment '# Act transition {index}' to /palindrome.py now.")
    for index, mode in enumerate(("plan", "act", "plan", "act", "plan"))
)
PALINDROME = "def is_palindrome(text):\n    return text == text[::-1]\n"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inspect_events(events):
    """Flag exact repetition and mode mentions for subsequent human inspection.

    Repeated calls are counted within a turn, not across intentionally repeated
    user requests. All arguments participate in equality. This does not assert
    that a repeat is unjustified or that a mentioned mode is the claimed mode.
    """
    calls = defaultdict(list)
    paragraphs = defaultdict(list)
    mentions = []
    call_ids = defaultdict(list)
    for event in events:
        if event["type"] == "tool_call":
            call_id = event.get("call_id")
            if call_id:
                call_ids[call_id].append(event["id"])
                if len(call_ids[call_id]) > 1:
                    continue  # A resumed stream can re-render the same call.
            key = (event["name"], json.dumps(event.get("args", {}), sort_keys=True))
            calls[key].append(event["id"])
        if event["type"] != "reasoning":
            continue
        for paragraph in re.split(r"\n\s*\n", event["text"]):
            normalized = " ".join(paragraph.casefold().split())
            if len(normalized) >= 80:
                paragraphs[normalized].append(event["id"])
            if re.search(r"\b(?:plan|act|action)[ -]mode\b|\bmode\b.{0,30}\b(?:plan|act)\b", paragraph, re.I):
                mentions.append({"event_id": event["id"], "mode": event.get("mode"), "text": paragraph})
    return {
        "reasoning_event_ids": [e["id"] for e in events if e["type"] == "reasoning"],
        "assistant_event_ids": [e["id"] for e in events if e["type"] == "assistant"],
        "tool_call_event_ids": [e["id"] for e in events if e["type"] == "tool_call"],
        "tool_result_event_ids": [e["id"] for e in events if e["type"] == "tool_result"],
        "mode_mention_candidates": mentions,
        "identical_call_groups": [{"name": key[0], "args": json.loads(key[1]), "event_ids": ids}
                                  for key, ids in calls.items() if len(ids) > 1],
        "replayed_call_groups": [{"call_id": call_id, "event_ids": ids}
                                 for call_id, ids in call_ids.items() if len(ids) > 1],
        "repeated_reasoning_paragraphs": [{"normalized_text": text, "event_ids": ids}
                                          for text, ids in paragraphs.items() if len(ids) > 1],
    }


class ModelInputs(BaseCallbackHandler):
    """Observe effective inputs and response IDs; never manufacture reasoning."""

    def __init__(self):
        self.requests = []
        self.by_run = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        for batch in messages:
            request = {
                "invocation": len(self.requests) + 1, "callback_run_id": str(run_id),
                "response_message_ids": [],
                "message_ids": [m.id for m in batch],
                "notice_count": sum(m.name == MODE_TRANSITION_SOURCE for m in batch),
                "human_messages": [{"id": m.id, "name": m.name, "text": m.text}
                                   for m in batch if isinstance(m, HumanMessage)],
                "returned_tools": [{"call_id": m.tool_call_id, "name": m.name, "text": m.text}
                                   for m in batch if isinstance(m, ToolMessage)],
            }
            self.requests.append(request)
            self.by_run[str(run_id)] = request

    def on_llm_end(self, response, *, run_id, **kwargs):
        request = self.by_run.get(str(run_id))
        if request is not None:
            request["response_message_ids"] = [str(g.message.id) for batch in response.generations
                                               for g in batch if getattr(g, "message", None) and g.message.id]


class RecordedFrontend(ApprovalFrontend):
    """Link recorder IDs to native stream IDs after MIRA records each delta.

    consume_messages -> SessionEventEmitter.reasoning_delta -> SessionRecorder
    -> FrontendEmitter -> this observer. Text comes exclusively from the saved
    events; the probe does not parse or generate a second reasoning stream.
    """

    def __init__(self):
        super().__init__()
        self.session = None
        self.links = {}
        self.display_answers = []
        self.decisions = []

    async def request(self, request):
        if isinstance(request, ArtifactDisplayRequest):
            artifact = self.session.record.get(f"current_{request.artifact_type}")
            answer = (f"Current {request.artifact_type.title()} rendered." if artifact else
                      f"No current {request.artifact_type.title()}.")
            self.display_answers.append(answer)
            self.decisions.append({"type": "display", "artifact_type": request.artifact_type, "answer": answer})
            return answer
        if isinstance(request, ArtifactReviewRequest):
            # No scenario requests a formal Plan. Reject
            # an unsolicited proposal without switching mode or retaining it.
            self.decisions.append({"type": "review", "answer": {"action": "clear"}})
            return {"action": "clear"}
        if isinstance(request, AskUserRequest):
            answer = ("Use the current implementation and add empty-string, single-character, "
                      "mixed-case, and punctuation examples. Do not create or retain a formal Plan.")
            self.decisions.append({"type": "ask_user", "answer": answer})
            return answer
        return await super().request(request)

    def emit(self, event):
        if self.session is None or not isinstance(event, MessageEvent):
            return
        event_type = {"reasoning": "reasoning", "content": "assistant"}.get(event.phase)
        if event_type is None:
            return
        recorded = next((e for e in reversed(self.session.record["events"]) if e["type"] == event_type), None)
        if recorded is not None:
            self.links[recorded["id"]] = {
                "message_id": event.message_id, "turn_id": event.turn_id,
                "mode": recorded["mode"], "type": event_type,
                "metadata": dict(event.metadata),
            }


def link_invocations(requests, links):
    """Join by native response ID, keeping missing identity distinct from no reasoning."""
    for request in requests:
        ids = set(request["response_message_ids"])
        matched = {key: value for key, value in links.items() if value["message_id"] in ids or (
            request.get("callback_run_id")
            and value.get("metadata", {}).get("run_id") == request["callback_run_id"]
        )}
        request["reasoning_event_ids"] = [int(key) for key, value in matched.items() if value["type"] == "reasoning"]
        request["assistant_event_ids"] = [int(key) for key, value in matched.items() if value["type"] == "assistant"]
        request["reasoning_availability"] = (
            "available" if request["reasoning_event_ids"] else "unavailable" if ids else "unmapped: no response ID"
        )
        request["response_finished"] = bool(ids)


def capture_reference(path, output):
    raw = path.read_bytes()
    record = json.loads(raw)
    events = record["events"]
    users = [event for event in events if event["type"] == "user"]
    assert [event["text"] for event in users[:3]] == [text for _, text in SCENARIO[:3]]
    # Preserve the original reasoning and first examples of the repeated reads,
    # not the user's entire local session or its unrelated metadata.
    retained = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 23, 24, 101}
    reference = {
        "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
        "sha256": hashlib.sha256(raw).hexdigest(), "session_id": record["id"],
        "event_count": len(events), "events": [e for e in events if e["id"] in retained],
        "turns": [],
    }
    for index, user in enumerate(users):
        end = events.index(users[index + 1]) if index + 1 < len(users) else len(events)
        turn_events = events[events.index(user):end]
        reference["turns"].append({"user_event_id": user["id"], "mode": user["mode"],
                                   "inspection": inspect_events(turn_events)})
    save_json(output / "reference.json", reference)
    return reference


async def resume_display(session, graph, answer):
    """Resume an answered native display interrupt through MIRA's normal runner.

    run_turn returns after displaying a Plan; the graph may still be paused in
    that tool. Supply its actual frontend answer as Command(resume=...), then
    let MIRA stream/record the continuation. Never overwrite checkpoint state or
    manufacture a model response. This probe adapter applies to either arm.
    """
    original_stream = graph.astream_events

    async def runner(**kwargs):
        first = True

        async def stream(payload, *args, **options):
            nonlocal first
            if first:
                first = False
                payload = Command(resume=answer)
            return await original_stream(payload, *args, **options)

        with patch.object(graph, "astream_events", stream):
            return await run_turn(**kwargs)

    return await session.prompt("", record_user=False, turn_runner=runner)


async def assert_no_synthetic_messages(session):
    # Reasoning/assistant text may legitimately quote the notice. Check message
    # identity and role rather than forbidding those words in model output.
    for graph, thread in ((session.agent, session.id), (session.plan_agent, f"{session.id}:plan")):
        async for snapshot in graph.aget_state_history({"configurable": {"thread_id": thread}}):
            assert not any(m.name == MODE_TRANSITION_SOURCE for m in snapshot.values.get("messages", []))
    assert not any(e.get("name") == MODE_TRANSITION_SOURCE for e in session.record["events"])


async def run_arm(source, profile, output, arm, timeout, scenario_name):
    from agent.llm import get_profile_model

    observer = ModelInputs()
    frontend = RecordedFrontend()
    scenario = INSPECTION_SCENARIO if scenario_name == "inspection" else SCENARIO
    evidence = {"arm": arm, "profile": profile.name, "model": profile.model, "scenario_name": scenario_name,
                "temperature": 0.0, "scenario": scenario, "turns": [], "event_links": {},
                "initial_project_files": [], "events": []}
    # Fresh projects, identical settings, same profile and sampling. Only
    # workspace paths differ. No preexisting transcript/checkpoint is imported.
    with tempfile.TemporaryDirectory(prefix=f"mira-mode-{arm}-") as directory:
        workspace = Path(directory)
        if scenario_name == "inspection":
            (workspace / "palindrome.py").write_text(PALINDROME, encoding="utf-8")
            evidence["initial_project_files"] = [{"path": "palindrome.py", "text": PALINDROME}]
        settings = normalize_settings({
            "models": {"main": profile.name},
            # The reference contains guard recovery errors. Use today's existing
            # guard equally in both arms so a repeated-call turn can terminate.
            "system": {"repeated_tool_call_guard": {"enabled": True}},
        })
        save_settings(workspace, settings)
        config = {**source, "settings": settings, "workspace": str(workspace),
                  "session_dir": str(workspace / ".mira" / "_sessions")}
        evidence["settings"] = settings
        model = get_profile_model(config, profile.name).model_copy(
            update={"callbacks": [observer], "temperature": 0.0},
        )
        bypass = patch.object(ModeTransitionMiddleware, "_with_notice", lambda self, request: request)
        with bypass if arm == "baseline" else nullcontext():
            with patch("agent.factory.get_llm", return_value=model):
                app = await MiraApplication.start(workspace=workspace, config=config, frontend=frontend)
                try:
                    session = await app.open_session("reasoning-ab")
                    frontend.session = session
                    for index, (mode, prompt) in enumerate(scenario, 1):
                        await session.set_mode(mode)
                        start_event = len(session.record["events"])
                        start_request = len(observer.requests)
                        turn = {"transition": index, "mode": mode, "prompt": prompt}
                        try:
                            result = await asyncio.wait_for(session.prompt(prompt), timeout=timeout)
                            graph = session.plan_agent if mode == "plan" else session.agent
                            thread = f"{session.id}:plan" if mode == "plan" else session.id
                            turn["display_resumes"] = 0
                            for _ in range(16):
                                snapshot = await graph.aget_state({"configurable": {"thread_id": thread}})
                                if not snapshot.next:
                                    break
                                interrupts = [i for task in snapshot.tasks for i in task.interrupts]
                                assert interrupts and all(i.value.get("type") in {"show_plan", "show_goal"}
                                                          for i in interrupts), "Unexpected pending graph work"
                                assert frontend.display_answers, "Display has no frontend answer"
                                result = await asyncio.wait_for(
                                    resume_display(session, graph, frontend.display_answers[-1]), timeout=timeout,
                                )
                                turn["display_resumes"] += 1
                            else:
                                raise RuntimeError("Repeated artifact displays exceeded the probe limit")
                            turn["final_reply"] = result.final_text
                        except Exception as error:
                            turn["error"] = f"{type(error).__name__}: {error}"
                        events = deepcopy(session.record["events"][start_event:])
                        requests = deepcopy(observer.requests[start_request:])
                        link_invocations(requests, frontend.links)
                        turn.update({"inspection": inspect_events(events), "invocations": requests,
                                     "event_ids": [e["id"] for e in events]})
                        file = workspace / "palindrome.py"
                        turn["palindrome_py"] = file.read_text(encoding="utf-8") if file.exists() else None
                        evidence["turns"].append(turn)
                        evidence["events"] = deepcopy(session.record["events"])
                        evidence["event_links"] = deepcopy(frontend.links)
                        evidence["frontend_decisions"] = deepcopy(frontend.decisions)
                        save_json(output / f"{scenario_name}-{arm}.json", evidence)
                        print(json.dumps({"arm": arm, "transition": index, "mode": mode,
                                          "calls": len(requests), "reasoning_ids": turn["inspection"]["reasoning_event_ids"],
                                          "availability": [r["reasoning_availability"] for r in requests],
                                          "repeat_groups": turn["inspection"]["identical_call_groups"],
                                          "error": turn.get("error")}), flush=True)
                        if turn.get("error"):
                            break  # Preserve incomplete work; never silently reset its graph.
                    # Delivery assertions are separate from judgments about reasoning.
                    for turn in evidence["turns"]:
                        assert all(r["notice_count"] == int(arm == "experiment") for r in turn["invocations"])
                    await assert_no_synthetic_messages(session)
                    saved = json.loads(app.store.path(session.id).read_text(encoding="utf-8"))
                    assert saved["events"] == session.record["events"], "Evidence differs from persisted recorder events"
                finally:
                    await app.shutdown()
    return evidence


async def live(args):
    source = load_config(ROOT)
    profile = source["model_registry"].profile(args.profile)
    if profile is None or "gemma-4-12b" not in profile.model.lower():
        raise RuntimeError("Select a configured local Gemma 4 12B profile with reasoning enabled")
    base = str(profile.values.get("api_base", "")).rstrip("/")
    if not base.startswith(("http://localhost:", "http://127.0.0.1:")):
        raise RuntimeError("The A/B probe requires a local endpoint")
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        response = await client.get(base + "/models")
        response.raise_for_status()
        assert profile.model in {item["id"] for item in response.json().get("data", [])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_reference(args.reference_session.resolve(), args.output_dir)
    arms = [await run_arm(source, profile, args.output_dir, arm, args.timeout, args.scenario)
            for arm in ("baseline", "experiment")]
    assert arms[0]["settings"] == arms[1]["settings"], "Arm settings differ"
    assert arms[0]["initial_project_files"] == arms[1]["initial_project_files"], "Arm seed files differ"
    assert arms[0]["scenario"] == arms[1]["scenario"], "Arm prompts differ"
    assert all(len(arm["turns"]) == 5 and not any(t.get("error") for t in arm["turns"])
               for arm in arms), "Incomplete paired run; inspect retained evidence"
    print("Paired run complete. Inspect recorded reasoning in", args.output_dir, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--profile", default="lmstudio-gemma-4-12b")
    parser.add_argument("--scenario", choices=("inspection", "regression"), default="inspection")
    parser.add_argument("--reference-session", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    if not args.live_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModeTransitionTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    if args.live or args.live_only:
        asyncio.run(live(args))


if __name__ == "__main__":
    main()
