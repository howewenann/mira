"""Effective model-message probes for transient Plan/Act notices.

    python tests/probes/probe_mode_transition.py
    python tests/probes/probe_mode_transition.py --live
    python tests/probes/probe_mode_transition.py --live-only

Deterministic checks use the real MIRA orchestration, middleware, SQLite saver,
and recording model. Optional live checks use configured local Gemma 4 12B in
a disposable workspace, recording messages at the model's callback boundary.
No source configuration or session is modified. Live behavior is diagnostic;
delivery assertions and absence from saved messages remain deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import yaml
from langchain_core.callbacks import BaseCallbackHandler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.middleware.mode_transition import MODE_TRANSITION_SOURCE  # noqa: E402
from config.loader import load_config  # noqa: E402
from config.settings import normalize_settings, save_settings  # noqa: E402
from core.application.app import MiraApplication  # noqa: E402
from tests.test_mode_transition import ApprovalFrontend, ModeTransitionTests  # noqa: E402


class ModelInputs(BaseCallbackHandler):
    def __init__(self):
        self.requests = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.requests.extend(list(batch) for batch in messages)


async def live() -> None:
    source = load_config(ROOT)
    candidates = [p for p in source["model_registry"].profiles.values()
                  if "gemma-4-12b" in p.model.lower()
                  and "localhost:1234" in str(p.values.get("api_base", ""))]
    if not candidates:
        print("LIVE SKIP: no configured local Gemma 4 12B profile")
        return
    profile = candidates[0]
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            response = await client.get(str(profile.values["api_base"]).rstrip("/") + "/models")
            response.raise_for_status()
            available = {item["id"] for item in response.json().get("data", [])}
    except (httpx.HTTPError, ValueError) as error:
        print(f"LIVE SKIP: endpoint unavailable ({type(error).__name__})")
        return
    if profile.model not in available:
        print(f"LIVE SKIP: {profile.model} unavailable")
        return

    from agent.llm import get_profile_model
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="mira-mode-transition-") as directory:
        workspace = Path(directory)
        settings = normalize_settings({"models": {"main": profile.name}})
        save_settings(workspace, settings)
        (workspace / ".mira" / "models.yml").write_text(yaml.safe_dump({
            "models": {profile.name: profile.values},
        }), encoding="utf-8")
        (workspace / "palindrome.py").write_text(
            "def is_palindrome(text):\n    return text == text[::-1]\n", encoding="utf-8",
        )
        config = {**source, "settings": settings, "workspace": str(workspace),
                  "session_dir": str(workspace / ".mira" / "_sessions")}
        observer = ModelInputs()
        model = get_profile_model(config, profile.name).model_copy(update={"callbacks": [observer]})
        with patch("agent.factory.get_llm", return_value=model):
            app = await MiraApplication.start(workspace=workspace, config=config, frontend=ApprovalFrontend())
            try:
                session = await app.open_session("live-transitions")
                for index, mode in enumerate(("plan", "act", "plan", "act", "plan")):
                    await session.set_mode(mode)
                    prompt = (
                        "Read /palindrome.py and briefly explain it. This is read-only inspection; "
                        "do not create a formal Plan."
                        if mode == "plan" else
                        f"Append the comment '# Act transition {index}' to /palindrome.py now."
                    )
                    start = len(observer.requests)
                    before = (workspace / "palindrome.py").read_text(encoding="utf-8")
                    result = await asyncio.wait_for(session.prompt(prompt), timeout=180)
                    requests = observer.requests[start:]
                    assert requests, "No effective model requests captured"
                    for messages in requests:
                        notices = [m for m in messages if m.name == MODE_TRANSITION_SOURCE]
                        assert len(notices) == 1, "Missing or duplicate transient notice"
                        assert f"{mode.upper()} mode" in notices[0].text
                        assert messages[messages.index(notices[0]) + 1].text == prompt
                    after = (workspace / "palindrome.py").read_text(encoding="utf-8")
                    print(json.dumps({"mode": mode, "calls": len(requests), "tools": result.tool_calls,
                                      "file_changed": before != after, "reply": result.final_text[:400]}), flush=True)
                    if mode == "plan":
                        assert before == after, "Plan unexpectedly changed the file"
                    # Report refusal/confusion without turning stochastic behavior into a delivery assertion.
                    elif before == after:
                        print("LIVE OBSERVATION: Act did not perform the requested edit", flush=True)
                start = len(observer.requests)
                await session.prompt("Briefly explain palindrome.py again; no formal Plan.")
                assert all(not any(m.name == MODE_TRANSITION_SOURCE for m in batch)
                           for batch in observer.requests[start:]), "Notice leaked into a later turn"
                await ModeTransitionTests().assert_clean(app, session)
            finally:
                await app.shutdown()
        completed = await asyncio.to_thread(subprocess.run, [
            sys.executable, "-m", "cli.main", "--workspace", str(workspace),
            "-p", "Read /palindrome.py and briefly explain is_palindrome.",
        ], cwd=ROOT, text=True, capture_output=True, timeout=180)
        print("CLI smoke exit:", completed.returncode)
        print(completed.stdout)
        assert completed.returncode == 0, completed.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    args = parser.parse_args()
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    if not args.live_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModeTransitionTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    if args.live or args.live_only:
        asyncio.run(live())


if __name__ == "__main__":
    main()
