"""Reproduce nine identical palindrome reads through real agent graphs.

Run from the repository with an activated environment:
    python tests/probes/probe_repeated_tool_call_guard.py
    python tests/probes/probe_repeated_tool_call_guard.py --live

The optional live probe uses a configured LM Studio Gemma profile, a disposable
workspace, and real MIRA. It seeds four actual graph requests (including the
first blocked result) to examine recovery without relying on Gemma to loop.
It also smoke tests the current checkout's CLI in that disposable workspace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent.middleware.repeated_tool_call_guard import RepeatedToolCallMiddleware

SOURCE = "def is_palindrome(text):\n    return text == text[::-1]\n"
REQUEST = "Inspect /palindrome.py and explain whether is_palindrome works. Use existing read results when sufficient."


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def deterministic(workspace: Path, *, enabled: bool = True, seed_recovery: bool = False):
    executed = []

    @tool
    def read_file(file_path: str, offset: int = 0, limit: int = 100) -> str:
        """Read the diagnostic fixture."""
        assert file_path == "/palindrome.py"
        executed.append(file_path)
        return (workspace / "palindrome.py").read_text(encoding="utf-8")

    calls = [AIMessage(content="", tool_calls=[{
        "id": f"read-{i}", "name": "read_file", "args": {"file_path": "/palindrome.py"},
    }]) for i in range(9)]
    model = ScriptedModel(responses=[*calls, AIMessage(content="Done.")])
    graph = create_agent(
        model, tools=[read_file],
        middleware=[RepeatedToolCallMiddleware()] if enabled else [],
        checkpointer=InMemorySaver(), interrupt_after=["tools"] if seed_recovery else None,
    )
    config = {"configurable": {"thread_id": "palindrome-probe"}}
    result = graph.invoke({"messages": [HumanMessage(content=REQUEST)]}, config)
    if seed_recovery:
        for _ in range(3):
            result = graph.invoke(None, config)
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    expected_statuses = (["success"] * 3 + ["error"] * (1 if seed_recovery else 3)) if enabled else ["success"] * 9
    assert [m.status for m in results] == expected_statuses
    assert len(executed) == (3 if enabled else 9)
    if enabled and not seed_recovery:
        assert model.i == 6
        assert "current agent turn has been stopped" in result["messages"][-1].content
        assert graph.get_state(config).next == ()
    assert [m.tool_call_id for m in results] == [f"read-{i}" for i in range(len(results))]
    print(json.dumps({"probe": "recovery-seed" if seed_recovery else "deterministic",
                      "enabled": enabled, "executed": len(executed), "statuses": expected_statuses}))
    return result["messages"]


async def live(workspace: Path) -> None:
    import httpx
    import yaml

    from agent.factory import build_agent
    from config.loader import load_config
    from config.settings import normalize_settings, save_settings
    from core.execution.runner import run_turn
    from ui.terminal.renderer import Renderer

    source_config = load_config(ROOT)
    candidates = [p for p in source_config["model_registry"].profiles.values()
                  if "gemma" in p.model.lower() and "localhost:1234" in str(p.values.get("api_base", ""))]
    if not candidates:
        print("LIVE SKIP: no configured LM Studio Gemma profile")
        return
    profile = candidates[0]
    base = str(profile.values["api_base"]).rstrip("/")
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            response = await client.get(base + "/models")
            response.raise_for_status()
            available = {item["id"] for item in response.json().get("data", [])}
    except (httpx.HTTPError, ValueError) as error:
        print(f"LIVE SKIP: LM Studio unavailable ({type(error).__name__})")
        return
    if profile.model not in available:
        print(f"LIVE SKIP: configured model {profile.model} is unavailable")
        return

    settings = normalize_settings({
        "models": {"main": profile.name},
        "system": {"repeated_tool_call_guard": {"enabled": True}},
        "hitl": {"tools": {"eval": {"enabled": False}, "task": {"enabled": False}}},
    })
    save_settings(workspace, settings)
    # A disposable registry allows the CLI to use exactly the same profile.
    (workspace / ".mira" / "models.yml").write_text(
        yaml.safe_dump({"models": {profile.name: profile.values}}, sort_keys=False), encoding="utf-8",
    )
    config = {**source_config, "settings": settings, "workspace": str(workspace),
              "session_dir": str(workspace / ".mira" / "_sessions")}
    seed = deterministic(workspace, seed_recovery=True)
    agent = build_agent(config, workspace, InMemorySaver())
    result = await asyncio.wait_for(run_turn(
        agent, REQUEST, Renderer(), "live-recovery", messages=seed,
    ), timeout=120)
    state = await agent.aget_state({"configurable": {"thread_id": "live-recovery"}})
    new_messages = state.values["messages"][len(seed):]
    new_calls = [c for m in new_messages if isinstance(m, AIMessage) for c in m.tool_calls]
    print(json.dumps({"probe": "live-mira-recovery", "profile": profile.name,
                      "new_tool_names": [c["name"] for c in new_calls],
                      "final_text": result.final_text, "pending_nodes": list(state.next)}))
    assert result.final_text and not state.next

    completed = await asyncio.to_thread(subprocess.run, [
        sys.executable, "-m", "cli.main", "--workspace", str(workspace),
        "-p", "Read /palindrome.py once and briefly explain is_palindrome.",
    ], cwd=ROOT, text=True, capture_output=True, timeout=120)
    print("CLI smoke exit:", completed.returncode)
    print(completed.stdout)
    if completed.returncode:
        print(completed.stderr)
    assert completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mira-repeated-tools-") as directory:
        workspace = Path(directory)
        (workspace / "palindrome.py").write_text(SOURCE, encoding="utf-8")
        deterministic(workspace)
        deterministic(workspace, enabled=False)
        if args.live:
            asyncio.run(live(workspace))


if __name__ == "__main__":
    main()
