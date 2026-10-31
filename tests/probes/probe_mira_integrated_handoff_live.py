"""Optional real-model smoke through MIRA's public application orchestration."""

from __future__ import annotations

import asyncio
import copy
import secrets
import tempfile
from pathlib import Path

from config.loader import load_config
from core.application.app import MiraApplication
from core.execution.compaction import commit_forced_compaction
from core.interface.requests import ApprovalRequest, ArtifactReviewRequest, AskUserRequest


class Frontend:
    def __init__(self) -> None:
        self.goal_reviews = 0

    def emit(self, event: object) -> None:
        pass

    async def request(self, request: object) -> object:
        if isinstance(request, ArtifactReviewRequest):
            if request.artifact_type != "goal":
                raise RuntimeError(f"Unexpected formal review: {request.artifact_type}")
            self.goal_reviews += 1
            return {"action": "implement"}
        if isinstance(request, ApprovalRequest):
            return [{"type": "allow_once"} for _ in request.interrupts]
        if isinstance(request, AskUserRequest):
            raise RuntimeError("Live smoke requires an unambiguous Goal")
        raise RuntimeError(f"Unexpected frontend request: {type(request).__name__}")


async def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    config = dict(load_config(repo))
    settings = copy.deepcopy(config["settings"])
    settings["tracing"]["enabled"] = False
    settings["system"]["rubric"]["enabled"] = False
    settings["system"]["dynamic_subagents"]["enabled"] = False
    config["settings"] = settings
    with tempfile.TemporaryDirectory(prefix="mira-integrated-live-") as directory:
        workspace = Path(directory)
        config["workspace"] = str(workspace)
        config["session_dir"] = str(workspace / ".mira" / "_sessions")
        frontend = Frontend()
        app = await MiraApplication.start(workspace=workspace, config=config, frontend=frontend)
        try:
            session = await app.open_session("integrated-live")
            marker = "MIRA-LIVE-" + secrets.token_hex(4).upper()
            print(f"marker={marker}", flush=True)
            await session.prompt(f"Remember the exact marker {marker}. Reply acknowledged. Do not call tools.")
            await session.set_mode("plan")
            answer = await session.prompt("What exact marker did I give you? Reply with only the marker. Do not create a Plan or call tools.")
            assert marker.lower() in answer.final_text.lower(), "Plan failed to inherit Act context"
            print("Act to Plan recall passed", flush=True)
            await session.set_mode("act")
            state = None
            for index in range(4):
                await session.prompt(f"Memory maintenance turn {index}. Reply OK. Do not call tools.")
                state = await session.agent.aget_state({"configurable": {"thread_id": session.id}})
                if state.values.get("_summarization_event"):
                    print(f"Automatic compaction after {index + 1} maintenance turns", flush=True)
                    break
            else:
                # The smoke keeps native summarization but shortens retention
                # so a bounded number of real application turns is enough.
                session.agent.mira_summarization._lc_helper.keep = ("messages", 2)
                plan = await session.agent.mira_compaction.aplan_forced_compaction_update(state.values)
                result = await commit_forced_compaction(session.agent, session.id, state, plan)
                assert result.compacted, f"Compaction did not run: {result.reason}"
                print("Forced compaction completed", flush=True)
            await session.set_mode("plan")
            answer = await session.prompt("Recall my original exact marker. Reply only with that marker. Do not create a Plan or call tools.")
            assert marker.lower() in answer.final_text.lower(), "Plan failed to recall after compacted Act handoff"
            print("Compacted Act to Plan recall passed", flush=True)
            await session.set_mode("act")
            await session.begin_goal(f"Create note.txt containing the exact text {marker} in the workspace root.")
            await session.prompt(f"Create note.txt containing the exact text {marker} in the workspace root.")
            assert frontend.goal_reviews == 1, f"Goal review count: {frontend.goal_reviews}"
            assert session.record["current_goal"]["attempts"] == 1
            assert (workspace / "note.txt").read_text(encoding="utf-8").strip() == marker
            print("Integrated Goal approval and Act execution passed", flush=True)
        finally:
            await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
