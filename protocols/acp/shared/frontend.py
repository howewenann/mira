"""Shared ACP frontend consuming MIRA's public application contract."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

from acp.schema import PermissionOption, ToolCallUpdate

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
from protocols.acp.shared.mapping import (
    artifact_ready,
    artifact_text,
    conversational_message_updates,
    interaction_end,
    message_updates,
    replay_message,
    todo_update,
    tool_result_update,
    tool_start,
)


_active_session: ContextVar[str | None] = ContextVar("mira_acp_session", default=None)
_STOP = object()
_REPLY_IN_CHAT = "reply-in-chat"


class ReplyInChat(asyncio.CancelledError):
    """End this turn so the next ordinary ACP prompt can supply open text."""


class InteractionCancelled(asyncio.CancelledError):
    """End this turn after the ACP client cancels a blocking interaction."""


class ACPFrontend:
    """Translate MIRA events and blocking requests onto one ACP connection."""

    def __init__(self, *, snapshot: Any) -> None:
        self.connection: Any = None
        self._snapshot = snapshot
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._senders: dict[str, asyncio.Task[None]] = {}
        self._errors: dict[str, BaseException] = {}
        self._tools: dict[str, dict[str, dict[str, Any]]] = {}
        self._precompleted_tools: dict[str, set[str]] = {}
        self._interaction_sequence = 0

    @contextmanager
    def bind(self, session_id: str) -> Iterator[None]:
        """Route request-only callbacks to the current ACP session task-locally."""
        token = _active_session.set(session_id)
        try:
            yield
        finally:
            _active_session.reset(token)

    def emit(self, event: FrontendEvent) -> None:
        """Queue ordered ACP updates without blocking MIRA's synchronous emitter."""
        session_id = event.session_id or _active_session.get()
        if not session_id:
            return
        if isinstance(event, MessageEvent):
            for update in message_updates(event):
                self.enqueue(session_id, update)
        elif isinstance(event, InformationEvent) and event.text:
            self._enqueue_conversational_message(session_id, event.text)
        elif isinstance(event, ToolEvent):
            self._tool_event(session_id, event)

    async def request(self, request: FrontendRequest) -> Any:
        """Resolve MIRA interactions with stable ACP permission buttons."""
        session_id = self._require_session()
        await self.flush(session_id)
        if isinstance(request, ApprovalRequest):
            return await self._approvals(session_id, request)
        if isinstance(request, AskUserRequest):
            value = getattr(request.interrupt, "value", request.interrupt)
            value = value if isinstance(value, Mapping) else {}
            return await self._ask_user(
                session_id,
                str(value.get("question") or "Input required"),
                [str(item) for item in value.get("options", ())],
                str(value.get("open_option") or ""),
            )
        if isinstance(request, ArtifactReviewRequest):
            return await self._review_artifact(session_id, request)
        if isinstance(request, ArtifactDisplayRequest):
            return await self._display_artifact(session_id, request.artifact_type)
        if isinstance(request, MCPApprovalRequest):
            selected = await self._permission_choice(
                session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id=self._interaction_id("mira-mcp-approval"),
                    title=request.preview or "Allow this MCP server connection?",
                    raw_input={},
                ),
                options=[
                    PermissionOption(option_id="allow", name="Allow once", kind="allow_once"),
                    PermissionOption(
                        option_id="always_allow",
                        name="Always allow",
                        kind="allow_always",
                    ),
                    PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
                ],
            )
            return selected if selected in {"allow", "always_allow"} else "deny"
        if isinstance(request, ConfirmationRequest):
            selected = await self._permission_choice(
                session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id=self._interaction_id(f"mira-confirm-{request.kind}"),
                    title=request.message,
                    raw_input=dict(request.context or {}),
                ),
                options=[
                    PermissionOption(option_id="confirm", name="Continue", kind="allow_once"),
                    PermissionOption(option_id="cancel", name="Cancel", kind="reject_once"),
                ],
            )
            return selected == "confirm"
        raise RuntimeError(f"unsupported frontend request: {type(request).__name__}")

    def enqueue(self, session_id: str, item: Any) -> None:
        queue = self._queues.setdefault(session_id, asyncio.Queue())
        if session_id not in self._senders or self._senders[session_id].done():
            self._senders[session_id] = asyncio.create_task(self._send(session_id, queue))
        queue.put_nowait(item)

    async def flush(self, session_id: str) -> None:
        queue = self._queues.get(session_id)
        if queue is not None:
            await queue.join()
        error = self._errors.pop(session_id, None)
        if error is not None:
            raise RuntimeError("ACP session update failed") from error

    async def replay(self, session_id: str, transcript: tuple[Mapping[str, Any], ...]) -> None:
        """Replay MIRA's authoritative transcript rather than graph checkpoints."""
        tools: dict[str, dict[str, Any]] = {}
        for event in transcript:
            update = replay_message(event)
            if update is not None:
                self.enqueue(session_id, update)
                continue
            if event.get("type") == "tool_call":
                call_id = str(event.get("call_id") or f"replay-tool-{event.get('id')}")
                name = str(event.get("name") or "tool")
                args = event.get("args") if isinstance(event.get("args"), dict) else {}
                tools[call_id] = {"name": name, "args": args}
                self.enqueue(session_id, tool_start(call_id, name, args))
                if name == "write_todos":
                    self.enqueue(session_id, todo_update(args.get("todos", [])))
            elif event.get("type") == "tool_result":
                call_id = str(event.get("call_id") or "")
                if call_id and call_id in tools:
                    replay_event = ToolEvent(
                        session_id=session_id,
                        phase=(
                            "completed_error"
                            if event.get("status") == "error"
                            else "completed_result"
                        ),
                        name=str(event.get("name") or ""),
                        tool_call_id=call_id,
                        result=event.get("output"),
                    )
                    self.enqueue(session_id, tool_result_update(replay_event, tools[call_id]))
        await self.flush(session_id)

    async def shutdown(self) -> None:
        first_error: BaseException | None = None
        for session_id in tuple(self._queues):
            try:
                await self.flush(session_id)
            except BaseException as exc:
                first_error = first_error or exc
            finally:
                self._queues[session_id].put_nowait(_STOP)
        if self._senders:
            await asyncio.gather(*self._senders.values(), return_exceptions=True)
        if first_error is not None:
            raise first_error

    async def _send(self, session_id: str, queue: asyncio.Queue[Any]) -> None:
        while True:
            item = await queue.get()
            try:
                if item is _STOP:
                    return
                await self.connection.session_update(
                    session_id=session_id,
                    update=item,
                    source="MIRA",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors.setdefault(session_id, exc)
            finally:
                queue.task_done()

    def _tool_event(self, session_id: str, event: ToolEvent) -> None:
        tools = self._tools.setdefault(session_id, {})
        if event.phase in {"start", "recovered_start"} and event.tool_call_id:
            args = event.arguments if isinstance(event.arguments, dict) else {}
            tools[event.tool_call_id] = {"name": event.name, "args": args}
            self.enqueue(session_id, tool_start(event.tool_call_id, event.name, args))
            if event.name == "write_todos":
                self.enqueue(session_id, todo_update(args.get("todos", [])))
        elif event.phase in {
            "result",
            "error",
            "completed_result",
            "completed_error",
            "recovered_result",
            "recovered_error",
        } and event.tool_call_id:
            precompleted = self._precompleted_tools.get(session_id, set())
            if event.tool_call_id in precompleted:
                precompleted.discard(event.tool_call_id)
                return
            self.enqueue(session_id, tool_result_update(event, tools.get(event.tool_call_id)))

    async def _approvals(self, session_id: str, request: ApprovalRequest) -> list[dict[str, str]]:
        decisions: list[dict[str, str]] = []
        used_tool_ids: set[str] = set()
        for interrupt in request.interrupts:
            value = getattr(interrupt, "value", interrupt)
            actions = value.get("action_requests", ()) if isinstance(value, Mapping) else ()
            for index, action_value in enumerate(actions):
                action = action_value if isinstance(action_value, Mapping) else {}
                name = str(action.get("name") or "tool")
                call_id = str(
                    action.get("id")
                    or action.get("call_id")
                    or action.get("tool_call_id")
                    or self._active_tool_id(session_id, name, excluding=used_tool_ids)
                    or f"approval-{index}"
                )
                used_tool_ids.add(call_id)
                selected = await self._permission_choice(
                    session_id,
                    tool_call=ToolCallUpdate(
                        tool_call_id=call_id,
                        title=name.replace("_", " ").title(),
                        raw_input=action.get("args", {}),
                    ),
                    options=[
                        PermissionOption(option_id="approve", name="Approve", kind="allow_once"),
                        PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
                    ],
                )
                decisions.append({"type": "approve" if selected == "approve" else "reject"})
        return decisions

    async def _ask_user(
        self,
        session_id: str,
        question: str,
        options: list[str],
        open_option: str,
    ) -> str:
        closed_options = [option for option in options if option and option != open_option]
        call_id = self._active_tool_id(
            session_id,
            "ask_user",
            excluding=set(),
        ) or self._interaction_id("mira-ask-user")
        selected = await self._permission_choice(
            session_id,
            tool_call=ToolCallUpdate(
                tool_call_id=call_id,
                title=question,
                raw_input={"question": question},
            ),
            options=[
                *[
                    PermissionOption(
                        option_id=f"choice-{index}",
                        name=option,
                        kind="allow_once",
                    )
                    for index, option in enumerate(closed_options)
                ],
                PermissionOption(
                    option_id=_REPLY_IN_CHAT,
                    name="Reply in chat",
                    kind="reject_once",
                ),
            ],
        )
        if selected is None:
            self.enqueue(session_id, interaction_end(call_id, cancelled=True))
            await self.flush(session_id)
            raise InteractionCancelled()
        if selected == _REPLY_IN_CHAT:
            self.enqueue(session_id, interaction_end(call_id))
            await self.flush(session_id)
            raise ReplyInChat()
        index_text = selected.removeprefix("choice-")
        if not index_text.isdigit() or int(index_text) >= len(closed_options):
            raise InteractionCancelled()
        return closed_options[int(index_text)]

    async def _review_artifact(
        self,
        session_id: str,
        request: ArtifactReviewRequest,
    ) -> dict[str, str]:
        artifact = dict(request.artifact or {})
        finalizer_id = self._active_tool_id(
            session_id,
            f"finalize_{request.artifact_type}",
            excluding=set(),
        )
        if not finalizer_id:
            finalizer_id = self._interaction_id(f"mira-{request.artifact_type}-artifact")
            self.enqueue(
                session_id,
                tool_start(
                    finalizer_id,
                    f"finalize_{request.artifact_type}",
                    artifact,
                ),
            )
        self.enqueue(session_id, artifact_ready(finalizer_id, artifact))
        self._precompleted_tools.setdefault(session_id, set()).add(finalizer_id)
        await self.flush(session_id)
        self._enqueue_conversational_message(
            session_id,
            artifact_text(request.artifact_type, artifact),
        )
        await self.flush(session_id)
        review_id = self._interaction_id(f"mira-{request.artifact_type}-review")
        selected = await self._permission_choice(
            session_id,
            tool_call=ToolCallUpdate(
                tool_call_id=review_id,
                title=f"Review MIRA {request.artifact_type.title()}",
                raw_input=artifact,
            ),
            options=[
                PermissionOption(option_id="implement", name="Implement", kind="allow_once"),
                PermissionOption(option_id="close", name="Keep", kind="allow_once"),
                PermissionOption(
                    option_id=_REPLY_IN_CHAT,
                    name="Revise in chat",
                    kind="reject_once",
                ),
            ],
        )
        if selected is None:
            self._precompleted_tools[session_id].discard(finalizer_id)
            raise InteractionCancelled()
        if selected == _REPLY_IN_CHAT:
            self._precompleted_tools[session_id].discard(finalizer_id)
            raise ReplyInChat()
        if selected not in {"implement", "close"}:
            raise InteractionCancelled()
        return {"action": selected}

    async def _display_artifact(self, session_id: str, kind: str) -> str:
        snapshot = self._snapshot(session_id)
        artifact = getattr(snapshot, f"current_{kind}")
        text = artifact_text(kind, artifact) if artifact else f"No retained MIRA {kind}."
        self._enqueue_conversational_message(session_id, text)
        await self.flush(session_id)
        return f"Displayed retained {kind}." if artifact else f"No retained {kind}."

    async def _permission_choice(
        self,
        session_id: str,
        *,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
    ) -> str | None:
        response = await self.connection.request_permission(
            session_id=session_id,
            tool_call=tool_call,
            options=options,
        )
        outcome = getattr(response, "outcome", None)
        if getattr(outcome, "outcome", "") != "selected":
            return None
        selected = str(getattr(outcome, "option_id", ""))
        return selected if selected in {option.option_id for option in options} else None

    def _enqueue_conversational_message(self, session_id: str, text: str) -> None:
        for update in conversational_message_updates(text):
            self.enqueue(session_id, update)

    def _active_tool_id(self, session_id: str, name: str, *, excluding: set[str]) -> str:
        tools = self._tools.get(session_id, {})
        return next(
            (
                tool_id
                for tool_id, tool in reversed(tuple(tools.items()))
                if tool.get("name") == name and tool_id not in excluding
            ),
            "",
        )

    def _interaction_id(self, prefix: str) -> str:
        self._interaction_sequence += 1
        return f"{prefix}-{self._interaction_sequence}"

    @staticmethod
    def _require_session() -> str:
        session_id = _active_session.get()
        if not session_id:
            raise RuntimeError("MIRA frontend request has no active ACP session")
        return session_id


__all__ = ["ACPFrontend", "InteractionCancelled", "ReplyInChat"]
