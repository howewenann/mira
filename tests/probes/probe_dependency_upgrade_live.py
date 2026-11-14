"""Bounded Gemma smoke for tools, compaction, handoff and cold session recovery.

    python tests/probes/probe_dependency_upgrade_live.py --output-dir build/dependency-upgrade/gemma-runtime

Uses MIRA's existing SessionRecorder and the mode-transition probe's frontend.
All project writes and checkpoints live in a disposable workspace. The output
directory retains the ordinary session.json and a short assertion report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.llm import get_profile_model
from config.loader import load_config
from config.settings import normalize_settings, save_settings
from core.application.app import MiraApplication
from core.execution.compaction import commit_forced_compaction
from tests.probes.probe_mode_transition import RecordedFrontend, assert_no_synthetic_messages


async def live(args):
    source = load_config(ROOT)
    profile = source["model_registry"].profile(args.profile)
    if profile is None or "gemma-4-12b" not in profile.model.lower():
        raise RuntimeError("Select the configured Gemma 4 12B profile")
    model = get_profile_model(source, profile.name).model_copy(update={"temperature": 0.0})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="mira-upgrade-live-") as directory:
        workspace = Path(directory)
        settings = normalize_settings({"models": {"main": profile.name}})
        save_settings(workspace, settings)
        config = {**source, "settings": settings, "workspace": str(workspace),
                  "session_dir": str(workspace / ".mira/_sessions")}
        frontend = RecordedFrontend()
        with patch("agent.factory.get_llm", return_value=model):
            app = await MiraApplication.start(workspace=workspace, config=config, frontend=frontend)
            session = await app.open_session("dependency-upgrade-live")
            frontend.session = session

            async def prompt(label, text):
                reply = await asyncio.wait_for(session.prompt(text), timeout=args.timeout)
                results.append({"check": label, "reply": reply.final_text})
                print(label, reply.final_text[:160], flush=True)
                return reply

            def retain():
                shutil.copyfile(app.store.path(session.id), args.output_dir / "session.json")
                (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

            try:
                await session.set_mode("plan")
                await prompt("Plan write refusal", "Do not create a formal Plan. Write /forbidden.txt containing forbidden now.")
                assert not (workspace / "forbidden.txt").exists()
                await session.set_mode("act")
                await prompt("Act multi-tool write and read", "Write /upgrade-a.txt containing exactly MIRA_UPGRADE_42, then read that file to verify its contents. Do not create a Goal or delegate.")
                assert (workspace / "upgrade-a.txt").read_text(encoding="utf-8").strip() == "MIRA_UPGRADE_42"
                calls = [e["name"] for e in session.record["events"] if e["type"] == "tool_call"]
                assert "write_file" in calls and "read_file" in calls, calls
                await prompt("QuickJS", "Use eval to calculate 6 * 7. Then reply with the result. Do not use another tool.")
                assert any(e["type"] == "tool_call" and e["name"] == "eval" for e in session.record["events"])
                assert any(e["type"] == "tool_result" and e.get("name") == "eval"
                           and "<result>42</result>" in e.get("output", "") for e in session.record["events"])
                for index in range(3):
                    await prompt(f"History {index}", "Remember that our exact marker is MIRA_UPGRADE_42. Reply OK, without tools.")
                snapshot = await session.agent.aget_state({"configurable": {"thread_id": session.id}})
                session.agent.mira_summarization._lc_helper.keep = ("messages", 2)
                plan = await asyncio.wait_for(
                    session.agent.mira_compaction.aplan_forced_compaction_update(snapshot.values), timeout=args.timeout,
                )
                compacted = await commit_forced_compaction(session.agent, session.id, snapshot, plan)
                assert compacted.compacted, compacted.reason
                results.append({"check": "Forced native compaction", "passed": True})
                await session.set_mode("plan")
                reply = await prompt("Compacted Act to Plan recall", "What is our exact marker? Reply only with it. Do not create a Plan or call tools.")
                assert "MIRA_UPGRADE_42" in reply.final_text
                await session.set_mode("act")
                reply = await prompt("Plan to Act recall", "What is our exact marker? Reply only with it, without tools.")
                assert "MIRA_UPGRADE_42" in reply.final_text
                await assert_no_synthetic_messages(session)
                retain()
                identity = session.id
                await app.shutdown()
                app = await MiraApplication.start(workspace=workspace, config=config, frontend=frontend)
                session = await app.open_session(identity)
                frontend.session = session
                reply = await prompt("Cold restart recall", "What is our exact marker? Reply only with it, without tools.")
                assert "MIRA_UPGRADE_42" in reply.final_text
                await assert_no_synthetic_messages(session)
                results.append({"check": "All assertions", "passed": True})
            except Exception as error:
                results.append({"check": "Failure", "error": f"{type(error).__name__}: {error}"})
                raise
            finally:
                retain()
                await app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="lmstudio-gemma-4-12b")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/dependency-upgrade/gemma-runtime")
    parser.add_argument("--timeout", type=float, default=180)
    asyncio.run(live(parser.parse_args()))
