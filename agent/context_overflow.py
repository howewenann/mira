"""Context overflow helpers for DeepAgents model calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages.utils import count_tokens_approximately

from runtime.usage import positive_int, usage_from_message

DEFAULT_CONTEXT_PRESSURE_FRACTION = 0.98
MAX_SIGNATURES = 128


class ContextPressureMiddleware(AgentMiddleware[Any, Any, Any]):
    """Raise DeepAgents-compatible overflow errors when context is already full."""

    def __init__(
        self,
        *,
        context_limit_tokens: int | None,
        threshold_fraction: float = DEFAULT_CONTEXT_PRESSURE_FRACTION,
        enabled: bool = True,
        token_counter: Callable[..., int] = count_tokens_approximately,
    ) -> None:
        self.context_limit_tokens = positive_int(context_limit_tokens)
        self.threshold_fraction = valid_fraction(threshold_fraction, DEFAULT_CONTEXT_PRESSURE_FRACTION)
        self.enabled = bool(enabled)
        self.token_counter = token_counter
        self._raised_signatures: set[str] = set()

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize provider context errors for sync model calls."""
        self._raise_if_context_is_full(request)
        try:
            return handler(request)
        except ContextOverflowError:
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize provider context errors for async model calls."""
        self._raise_if_context_is_full(request)
        try:
            return await handler(request)
        except ContextOverflowError:
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise

    def _raise_if_context_is_full(self, request: Any) -> None:
        if not self.enabled or not self.context_limit_tokens:
            return

        threshold = max(1, int(self.context_limit_tokens * self.threshold_fraction))
        pressure = self._request_pressure(request, threshold)
        if pressure is None:
            return

        source, tokens, signature = pressure
        if not self._remember_signature(signature):
            return
        raise ContextOverflowError(
            "MIRA simulated a context overflow so DeepAgents can compact: "
            f"{source} context is {tokens} tokens, threshold is {threshold}, "
            f"configured limit is {self.context_limit_tokens}."
        )

    def _request_pressure(self, request: Any, threshold: int) -> tuple[str, int, str] | None:
        messages = list(getattr(request, "messages", []) or [])
        reported = reported_context_pressure(messages, threshold, thread_id(request))
        if reported is not None:
            return reported

        estimated = self._count_request_tokens(request)
        if estimated >= threshold:
            return (
                "estimated",
                estimated,
                f"{thread_id(request)}:estimated:{len(messages)}:{estimated}",
            )
        return None

    def _count_request_tokens(self, request: Any) -> int:
        messages = list(getattr(request, "messages", []) or [])
        system_message = getattr(request, "system_message", None)
        counted_messages = [system_message, *messages] if system_message is not None else messages
        tools = getattr(request, "tools", None)
        try:
            return positive_int(self.token_counter(counted_messages, tools=tools))
        except TypeError:
            return positive_int(self.token_counter(counted_messages))

    def _remember_signature(self, signature: str) -> bool:
        if signature in self._raised_signatures:
            return False
        if len(self._raised_signatures) >= MAX_SIGNATURES:
            self._raised_signatures.clear()
        self._raised_signatures.add(signature)
        return True


def reported_context_pressure(messages: list[Any], threshold: int, thread: str) -> tuple[str, int, str] | None:
    """Return provider-reported pressure from the newest usage-bearing message."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        usage = usage_from_message(message)
        tokens = positive_int(usage.get("context_tokens"))
        if tokens < threshold:
            continue
        source = str(usage.get("source") or "reported")
        message_id = str(getattr(message, "id", "") or "")
        fingerprint = message_id or f"{type(message).__name__}:{index}:{tokens}"
        return (source, tokens, f"{thread}:reported:{fingerprint}:{tokens}")
    return None


def raise_context_overflow_if_detected(exc: Exception) -> None:
    """Raise `ContextOverflowError` when a provider error is really context overflow."""
    if is_context_overflow_error(exc):
        raise ContextOverflowError(str(exc) or "context overflow") from exc


def is_context_overflow_error(exc: BaseException) -> bool:
    """Return whether an arbitrary provider exception means context overflow."""
    if isinstance(exc, ContextOverflowError):
        return True

    text = exception_text(exc)
    if not text or "rate limit" in text:
        return False

    direct_markers = (
        "context_length_exceeded",
        "input tokens exceed the configured limit",
        "exceeds the context window",
        "prompt is too long",
    )
    if any(marker in text for marker in direct_markers):
        return True

    overflow_words = ("exceed", "exceeded", "exceeds", "too long", "maximum", "configured limit")
    if ("context" in text or "prompt" in text) and any(word in text for word in overflow_words):
        return True
    return "token" in text and any(word in text for word in overflow_words) and "limit" in text


def exception_text(exc: BaseException) -> str:
    """Collect common exception fields into one lowercase search string."""
    parts = [str(exc)]
    for name in ("message", "body", "response"):
        value = getattr(exc, name, None)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def thread_id(request: Any) -> str:
    """Return the LangGraph thread id if available."""
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return ""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    return str(configurable.get("thread_id") or "")


def valid_fraction(value: Any, default: float) -> float:
    """Return a positive threshold fraction, falling back for invalid values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
