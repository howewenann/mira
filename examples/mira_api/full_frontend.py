"""Full interactive terminal frontend for MIRA's direct Python API."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mira import MiraApplication, MiraSession
from mira.api import (
    ApprovalRequest,
    ArtifactDisplayRequest,
    ArtifactReviewRequest,
    AskUserRequest,
    FrontendEvent,
    FrontendRequest,
    InformationEvent,
    MCPApprovalRequest,
    MCPElicitationRequest,
    MessageEvent,
    RuntimeEvent,
    ToolEvent,
)


def interrupt_value(interrupt: Any) -> Any:
    """Return the decision payload carried by a native LangGraph interrupt.

    Depending on where a request originated, MIRA may provide either the
    interrupt object or its already-unwrapped value. A frontend can handle both
    without importing LangGraph itself.
    """
    return getattr(interrupt, "value", interrupt)


class FullFrontend:
    """Example implementation of MIRA's complete blocking-request contract."""

    def __init__(self) -> None:
        self.session: MiraSession | None = None
        self._assistant_open = False
        self.turn_started = asyncio.Event()

    def emit(self, event: FrontendEvent) -> None:
        # Event handling -----------------------------------------------------
        # MIRA calls emit() synchronously with ordered streaming observations.
        # A real GUI would update owned widgets here; this example writes a
        # readable terminal transcript instead.
        if isinstance(event, RuntimeEvent):
            if event.kind == "session" and event.state == "turn_started":
                self.turn_started.set()
            elif event.kind == "inspection" and event.state == "reset":
                # Announce the key; do not fetch full transcripts on every delta.
                print(f"\n[inspection available] {event.detail['inspection_id']}")
            return

        if isinstance(event, MessageEvent):
            if event.phase == "content":
                print(event.text, end="", flush=True)
                self._assistant_open = True
            elif event.phase == "reasoning":
                print(f"\n[reasoning] {event.text}", end="", flush=True)
            elif event.phase == "discard_reasoning":
                print("\n[reasoning hidden]", flush=True)
            elif event.phase == "end" and self._assistant_open:
                print(flush=True)
                self._assistant_open = False
            return

        if isinstance(event, ToolEvent) and event.phase in {"start", "recovered_start"}:
            print(f"\n[tool] {event.name}: {event.arguments!r}", flush=True)
        elif isinstance(event, InformationEvent):
            print(f"\n[{event.kind}] {event.text}", flush=True)
        else:
            # The public event vocabulary can grow. Showing an unfamiliar event
            # is safer for a reference frontend than crashing or hiding it.
            print(f"\n[{type(event).__name__}] {event!r}", flush=True)

    async def request(self, request: FrontendRequest) -> Any:
        # Interactive requests ---------------------------------------------
        # MIRA pauses the active turn until this async callback returns the
        # response shape documented for that request type.
        if isinstance(request, ApprovalRequest):
            return await self._approve_actions(request)
        if isinstance(request, AskUserRequest):
            return await self._answer_question(request)
        if isinstance(request, ArtifactReviewRequest):
            return await self._review_artifact(request)
        if isinstance(request, ArtifactDisplayRequest):
            return self._display_artifact(request)
        if isinstance(request, MCPApprovalRequest):
            return await self._approve_mcp(request)
        if isinstance(request, MCPElicitationRequest):
            return await self._answer_mcp_elicitation(request)
        raise RuntimeError(f"Unsupported request: {type(request).__name__}")

    async def _approve_actions(self, request: ApprovalRequest) -> list[dict[str, Any]]:
        """Collect MIRA decisions; Core translates them to native LangGraph values."""
        decisions: list[dict[str, Any]] = []
        for interrupt in request.interrupts:
            value = interrupt_value(interrupt)
            actions = value.get("action_requests", []) if isinstance(value, dict) else []
            if not actions:
                actions = [value]
            for action in actions:
                print("\nMIRA requests permission for:")
                print(json.dumps(action, indent=2, default=str))
                answer = await asyncio.to_thread(
                    input,
                    "Decision: allow_once, allow_always, edit, reject_once [reject_once]: ",
                )
                kind = answer.strip().lower()
                if kind == "edit":
                    edited = await asyncio.to_thread(input, "Complete edited action JSON (name and args): ")
                    action = json.loads(edited)
                    if not isinstance(action, dict) or not isinstance(action.get("args"), dict) or not action.get("name"):
                        raise ValueError("An edited action requires name and args")
                    decisions.append({"type": "edit", "edited_action": action})
                elif kind in {"allow_once", "allow_always"}:
                    decisions.append({"type": kind})
                else:
                    decisions.append({"type": "reject_once"})
        return decisions

    async def _answer_question(self, request: AskUserRequest) -> str:
        """Return a selected or free-form answer to MIRA's AskUser request."""
        value = interrupt_value(request.interrupt)
        question = (
            str(value.get("question") or "MIRA needs input.")
            if isinstance(value, dict)
            else str(value)
        )
        options = value.get("options", []) if isinstance(value, dict) else []
        print(f"\n{question}")
        for index, option in enumerate(options, start=1):
            print(f"  {index}. {option}")
        return await asyncio.to_thread(input, "Answer: ")

    async def _review_artifact(self, request: ArtifactReviewRequest) -> dict[str, Any]:
        """Review a proposed formal Goal or Plan using MIRA's public actions."""
        print(f"\nProposed {request.artifact_type}:")
        print(json.dumps(dict(request.artifact or {}), indent=2, default=str))
        answer = await asyncio.to_thread(
            input,
            "Action: implement, close, revise, or clear [close]: ",
        )
        action = answer.strip().lower()
        if action not in {"implement", "close", "revise", "clear"}:
            action = "close"
        decision: dict[str, Any] = {"action": action}
        if action == "revise":
            decision["feedback"] = await asyncio.to_thread(input, "Revision feedback: ")
        return decision

    def _display_artifact(self, request: ArtifactDisplayRequest) -> str:
        """Read the current durable artifact projection from the session."""
        if self.session is None:
            return f"Current {request.artifact_type} is unavailable."
        # snapshot() is the authoritative, read-only projection of session state.
        snapshot = self.session.snapshot()
        artifact = (
            snapshot.current_goal
            if request.artifact_type == "goal"
            else snapshot.current_plan
        )
        print(json.dumps(dict(artifact or {}), indent=2, default=str))
        return f"Current {request.artifact_type} displayed."

    async def _approve_mcp(self, request: MCPApprovalRequest) -> str:
        """Ask before connecting a configured MCP server; denial is the default."""
        print(f"\nMCP connection request:\n{request.preview}")
        answer = await asyncio.to_thread(input, "Type allow or always_allow [deny]: ")
        return answer.strip() if answer.strip() in {"allow", "always_allow"} else "deny"

    async def _answer_mcp_elicitation(self, request: MCPElicitationRequest) -> dict[str, Any]:
        """Return keyed MCP form or URL decisions in the native resume shape."""
        payload = interrupt_value(request.interrupt)
        responses: dict[str, Any] = {}
        for item in payload.get("requests", []):
            # Show the server's message and requested schema or URL before asking.
            print(json.dumps(item, indent=2, default=str))
            answer = await asyncio.to_thread(input, "accept, decline, or cancel [decline]: ")
            action = answer.strip().lower()
            if action not in {"accept", "cancel"}:
                action = "decline"
            response: dict[str, Any] = {"action": action}
            if action == "accept" and item.get("mode") == "form":
                text = await asyncio.to_thread(input, "Form values as a JSON object: ")
                content = json.loads(text)
                if not isinstance(content, dict):
                    raise ValueError("MCP form values must be a JSON object")
                response["content"] = content
            responses[str(item["key"])] = response
        return {"responses": responses}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Send one prompt and exit.")
    parser.add_argument("--resume", action="store_true", help="Open saved state without executing it.")
    parser.add_argument("--session", help="Durable session ID to open.")
    parser.add_argument("--cancel-after", type=float, help="Cancel a one-shot prompt after this many seconds.")
    arguments = parser.parse_args()

    frontend = FullFrontend()

    # The application owns resources shared by every session in this workspace.
    application = await MiraApplication.start(workspace=".", frontend=frontend)
    try:
        # Open one durable MIRA conversation and give the frontend access to its
        # public snapshot for ArtifactDisplayRequest handling.
        session = await application.open_session(session_id=arguments.session, resume=arguments.resume)
        frontend.session = session

        if arguments.prompt is not None:
            if arguments.cancel_after is None:
                await session.prompt(arguments.prompt)
            else:
                # Wait for Core to begin before cancellation can target this turn.
                frontend.turn_started.clear()
                turn = asyncio.create_task(session.prompt(arguments.prompt))
                started = asyncio.create_task(frontend.turn_started.wait())
                try:
                    await asyncio.wait({turn, started}, return_when=asyncio.FIRST_COMPLETED)
                    if not turn.done():
                        await asyncio.sleep(max(0, arguments.cancel_after))
                        await session.cancel()
                    try:
                        await turn
                    except asyncio.CancelledError:
                        print("\nCancelled; captured inspection history remains available.")
                finally:
                    started.cancel()
                    await asyncio.gather(started, return_exceptions=True)
        else:
            # Reuse the same durable MiraSession for each message. Responses
            # continue to arrive through FullFrontend.emit() and request().
            print("Connected to MIRA's in-process Python API.")
            print("Type /quit, /inspections, /inspect ID, /resume-pending, or /continue-incomplete.")
            print(f"Session: {session.snapshot().session_id}")
            while True:
                try:
                    prompt = await asyncio.to_thread(input, "\n> ")
                except (EOFError, KeyboardInterrupt):
                    break
                prompt = prompt.strip()
                if prompt == "/quit":
                    break
                if prompt == "/inspections":
                    print(json.dumps(session.inspections(), indent=2, default=str))
                elif prompt.startswith("/inspect "):
                    # Selection fetches one detached payload through the public API.
                    inspection = session.inspection(prompt.removeprefix("/inspect ").strip())
                    if inspection is None:
                        print("Inspection unavailable.")
                    else:
                        print(f"{inspection.title} [{inspection.status}]")
                        for item in inspection.events:
                            print(item.kind, item.name, item.text, item.args or "")
                elif prompt == "/resume-pending":
                    await session.resume_pending()
                elif prompt == "/continue-incomplete":
                    answer = await asyncio.to_thread(input, "Side effects may repeat. Continue? [y/N] ")
                    if answer.strip().lower() == "y":
                        await session.continue_incomplete()
                elif prompt:
                    await session.prompt(prompt)
    finally:
        # Application shutdown closes sessions and releases MCP/runtime resources.
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
