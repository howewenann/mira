"""Generic deterministic correction loops for natural agent stops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Protocol

from langchain.agents.middleware.types import AgentMiddleware, AgentState, PrivateStateAttr, hook_config
from langchain_core.messages import AIMessage, SystemMessage

CORRECTION_EVENT = "correction"
CORRECTION_SOURCE = "mira_correction"


@dataclass(frozen=True)
class CorrectionDecision:
    """Result of checking one natural-stop assistant message."""

    accepted: bool
    failed_check: str = ""
    retry_prompt: str = ""


class CorrectionRule(Protocol):
    """Workflow-owned policy consumed by the generic correction lifecycle."""

    protocol_id: str
    check_name: str
    workflow_label: str
    failure_text: str

    def applies(self, state: AgentState) -> bool:
        """Return whether this rule owns the current workflow state."""

    def reminder(self, state: AgentState) -> str:
        """Return transient model guidance for the active workflow state."""

    def inspect(self, message: AIMessage, state: AgentState) -> CorrectionDecision:
        """Classify one no-tool natural stop."""


class CorrectionState(AgentState):
    """Private, turn-scoped correction state keyed by protocol identifier."""

    _correction_retries: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    _correction_pending: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class CorrectionMiddleware(AgentMiddleware[CorrectionState, Any, Any]):
    """Apply deterministic, bounded correction rules to natural agent stops."""

    state_schema = CorrectionState

    def __init__(self, *, rules: tuple[CorrectionRule, ...], max_retries: int = 2) -> None:
        if not rules:
            raise ValueError("CorrectionMiddleware requires at least one rule")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError("CorrectionMiddleware max_retries must be a positive integer")
        protocol_ids = [rule.protocol_id for rule in rules]
        if len(protocol_ids) != len(set(protocol_ids)):
            raise ValueError("CorrectionMiddleware rule protocol IDs must be unique")
        self.rules = rules
        self.max_retries = max_retries

    def before_agent(self, state: CorrectionState, runtime: Any) -> dict[str, Any]:  # noqa: ARG002
        """Start every externally invoked turn with fresh retry bookkeeping."""
        return {"_correction_retries": {}, "_correction_pending": {}}

    async def abefore_agent(
        self,
        state: CorrectionState,
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Async variant of before_agent."""
        return self.before_agent(state, runtime)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Inject active correction reminders into synchronous model calls."""
        return handler(self._request_with_reminders(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Inject active correction reminders into asynchronous model calls."""
        return await handler(self._request_with_reminders(request))

    def after_model(self, state: CorrectionState, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Tool calls bypass natural-stop correction and clear retry counts."""
        return self._after_model_update(state)

    async def aafter_model(
        self,
        state: CorrectionState,
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of after_model."""
        return self._after_model_update(state)

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: CorrectionState, runtime: Any) -> dict[str, Any] | None:
        """Accept, correct, or exhaust one no-tool natural stop."""
        return self._after_agent_update(state, runtime)

    async def aafter_agent(self, state: CorrectionState, runtime: Any) -> dict[str, Any] | None:
        """Async variant of after_agent."""
        return self._after_agent_update(state, runtime)

    def _request_with_reminders(self, request: Any) -> Any:
        rules = self._active_rules(request.state)
        if not rules:
            return request
        reminders = [rule.reminder(request.state).strip() for rule in rules]
        pending = request.state.get("_correction_pending") or {}
        reminders.extend(pending[rule.protocol_id] for rule in rules if pending.get(rule.protocol_id))
        reminder = "\n\n".join(value for value in reminders if value)
        if not reminder:
            return request
        current = getattr(request, "system_message", None)
        if current is None:
            system_message = SystemMessage(content=reminder)
        else:
            system_message = current.model_copy(
                update={
                    "content": [
                        *current.content_blocks,
                        {"type": "text", "text": f"\n\n{reminder}"},
                    ]
                }
            )
        return request.override(system_message=system_message)

    def _after_model_update(self, state: CorrectionState) -> dict[str, Any] | None:
        message = _last_ai_message(state.get("messages", []))
        if message is None or not message.tool_calls:
            return None
        if not state.get("_correction_retries") and not state.get("_correction_pending"):
            return None
        # The requested tool action fulfilled the correction. Its result starts
        # the normal model continuation; do not repeat the old action instruction.
        return {"_correction_retries": {}, "_correction_pending": {}}

    def _after_agent_update(self, state: CorrectionState, runtime: Any) -> dict[str, Any] | None:
        rules = self._active_rules(state)
        if not rules:
            return {"_correction_retries": {}, "_correction_pending": {}}
        if len(rules) != 1:
            names = ", ".join(rule.protocol_id for rule in rules)
            raise RuntimeError(f"multiple correction rules matched the same state: {names}")

        message = _last_ai_message(state.get("messages", []))
        if message is None:
            return {"_correction_retries": {}, "_correction_pending": {}}
        if message.tool_calls:
            return {"_correction_retries": {}, "_correction_pending": {}}

        rule = rules[0]
        decision = rule.inspect(message, state)
        retries = dict(state.get("_correction_retries") or {})
        if decision.accepted:
            retries.pop(rule.protocol_id, None)
            return {"_correction_retries": retries, "_correction_pending": {}}

        used = int(retries.get(rule.protocol_id) or 0)
        if used >= self.max_retries:
            self._emit(
                runtime,
                rule=rule,
                decision=decision,
                attempt=self.max_retries,
                exhausted=True,
                terminal_text=rule.failure_text,
            )
            retries.pop(rule.protocol_id, None)
            return {
                "messages": [AIMessage(content=rule.failure_text)],
                "_correction_retries": retries,
                "_correction_pending": {},
            }

        attempt = used + 1
        self._emit(runtime, rule=rule, decision=decision, attempt=attempt)
        retries[rule.protocol_id] = attempt
        return {
            # Keep this in graph state until the correction resolves, including
            # checkpoint resume of this retry. before_agent resets independent
            # turns, and request overrides never add guidance to messages.
            "_correction_pending": {rule.protocol_id: decision.retry_prompt},
            "_correction_retries": retries,
            "jump_to": "model",
        }

    def _active_rules(self, state: AgentState) -> list[CorrectionRule]:
        return [rule for rule in self.rules if rule.applies(state)]

    def _emit(
        self,
        runtime: Any,
        *,
        rule: CorrectionRule,
        decision: CorrectionDecision,
        attempt: int,
        exhausted: bool = False,
        terminal_text: str = "",
    ) -> None:
        writer = getattr(runtime, "stream_writer", None)
        if not callable(writer):
            return
        payload = {
            "type": CORRECTION_EVENT,
            "protocol": rule.protocol_id,
            "check_name": rule.check_name,
            "workflow": rule.workflow_label,
            "failed_check": decision.failed_check,
            "retry_prompt": decision.retry_prompt if not exhausted else "",
            "attempt": attempt,
            "max_retries": self.max_retries,
            "exhausted": exhausted,
        }
        if terminal_text:
            payload["terminal_text"] = terminal_text
        writer(payload)


def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)


__all__ = [
    "CORRECTION_EVENT",
    "CORRECTION_SOURCE",
    "CorrectionDecision",
    "CorrectionMiddleware",
    "CorrectionRule",
    "CorrectionState",
]
