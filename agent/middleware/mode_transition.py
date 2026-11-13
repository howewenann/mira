"""Turn-scoped mode guidance at the model boundary only."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import HumanMessage


MODE_TRANSITION_SOURCE = "mira_mode_transition"
MODE_NOTICES = {
    "plan": (
        "You are now in PLAN mode. This environment is read-only.\n"
        "Previous Act-mode instructions are historical context.\n"
        "Follow current Plan restrictions and use only currently available tools."
    ),
    "act": (
        "You are now in ACT mode.\n"
        "Previous Plan-mode restrictions are historical context and no longer govern execution.\n"
        "Follow current Act instructions and use currently available tools.\n"
        "Current tool permissions and approval requirements still apply."
    ),
}


class ModeTransitionMiddleware(AgentMiddleware):
    """Insert a notice before the bound user message on every call of its turn.

    The orchestrator owns pending/active metadata and supplies it through trusted
    runtime context, including on checkpoint resume. Never update graph state or
    mutate the original request: only the model-bound copy contains the notice.
    """

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        return handler(self._with_notice(request))

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        return await handler(self._with_notice(request))

    def _with_notice(self, request: ModelRequest) -> ModelRequest:
        context = getattr(request.runtime, "context", None)
        transition = getattr(context, "mode_transition", None)
        if not transition or transition.get("mode") not in MODE_NOTICES:
            return request
        anchor = transition.get("message_id")
        if not anchor:
            return request
        # Rewrapping an already overridden request must not accumulate notices.
        notice_id = f"{MODE_TRANSITION_SOURCE}:{anchor}"
        messages = [m for m in request.messages if not (
            isinstance(m, HumanMessage) and m.name == MODE_TRANSITION_SOURCE and m.id == notice_id
        )]
        index = next((i for i, m in enumerate(messages) if m.id == anchor), None)
        if index is None:
            # Native compaction may summarize even the current turn's user
            # message. Keep guidance before its retained continuation, without
            # resurrecting archived messages or disturbing tool-call pairs.
            history = request.state.get("messages", [])
            start = next((i for i, m in enumerate(history) if m.id == anchor), None)
            if start is None:
                return request
            continuation = {m.id for m in history[start + 1:] if m.id}
            index = next((i for i, m in enumerate(messages) if m.id in continuation), len(messages))
        notice = HumanMessage(
            content=MODE_NOTICES[transition["mode"]], name=MODE_TRANSITION_SOURCE, id=notice_id,
        )
        return request.override(messages=[*messages[:index], notice, *messages[index:]])
