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
    ConfirmationRequest,
    FrontendEvent,
    FrontendRequest,
    InformationEvent,
    MCPApprovalRequest,
    MessageEvent,
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

    def emit(self, event: FrontendEvent) -> None:
        # Event handling -----------------------------------------------------
        # MIRA calls emit() synchronously with ordered streaming observations.
        # A real GUI would update owned widgets here; this example writes a
        # readable terminal transcript instead.
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
        if isinstance(request, ConfirmationRequest):
            answer = await asyncio.to_thread(input, f"{request.message} [y/N] ")
            return answer.strip().lower() == "y"
        raise RuntimeError(f"Unsupported request: {type(request).__name__}")

    async def _approve_actions(self, request: ApprovalRequest) -> list[dict[str, Any]]:
        """Return the native approve/reject decisions expected by LangGraph."""
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
                    "Approve? Type 'approve' to allow [reject]: ",
                )
                # Never silently auto-approve. A real application should show
                # the allowed native approve/edit/reject choices in trusted UX.
                decisions.append(
                    {"type": "approve"}
                    if answer.strip().lower() == "approve"
                    else {"type": "reject"}
                )
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


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Send one prompt and exit.")
    arguments = parser.parse_args()

    frontend = FullFrontend()

    # The application owns resources shared by every session in this workspace.
    application = await MiraApplication.start(workspace=".", frontend=frontend)
    try:
        # Open one durable MIRA conversation and give the frontend access to its
        # public snapshot for ArtifactDisplayRequest handling.
        session = await application.open_session()
        frontend.session = session

        if arguments.prompt is not None:
            await session.prompt(arguments.prompt)
        else:
            # Reuse the same durable MiraSession for each message. Responses
            # continue to arrive through FullFrontend.emit() and request().
            print("Connected to MIRA's in-process Python API.")
            print("Type /quit to stop.")
            while True:
                try:
                    prompt = await asyncio.to_thread(input, "\n> ")
                except (EOFError, KeyboardInterrupt):
                    break
                prompt = prompt.strip()
                if prompt == "/quit":
                    break
                if prompt:
                    await session.prompt(prompt)
    finally:
        # Application shutdown closes sessions and releases MCP/runtime resources.
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
