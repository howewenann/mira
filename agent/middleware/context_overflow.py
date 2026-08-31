"""Provider context overflow normalization for DeepAgents model calls."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError

CONTEXT_NOTICE_ATTR = "_mira_context_notice"
CONTEXT_NOTICE_RENDERED_ATTR = "_mira_context_notice_rendered"
PROVIDER_CONTEXT_NOTICE = "Provider context limit reached. Compacting older context and retrying."
GENERIC_CONTEXT_NOTICE = "Context limit pressure detected. Compacting older context and retrying."

_context_notice: ContextVar[str | None] = ContextVar("mira_context_notice", default=None)
_fallback_context_notice: str | None = None


class ProviderContextOverflowMiddleware(AgentMiddleware[Any, Any, Any]):
    """Convert provider-specific context failures into LangChain overflow errors."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize context overflow errors for synchronous model calls."""
        try:
            return handler(request)
        except ContextOverflowError as exc:
            attach_context_notice(exc, PROVIDER_CONTEXT_NOTICE)
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize context overflow errors for asynchronous model calls."""
        try:
            return await handler(request)
        except ContextOverflowError as exc:
            attach_context_notice(exc, PROVIDER_CONTEXT_NOTICE)
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise


def raise_context_overflow_if_detected(exc: Exception) -> None:
    """Raise `ContextOverflowError` when a provider error is really context overflow."""
    if is_context_overflow_error(exc):
        set_context_overflow_notice(PROVIDER_CONTEXT_NOTICE)
        raise context_overflow_error("provider context limit reached", PROVIDER_CONTEXT_NOTICE) from exc


def context_overflow_error(message: str, notice: str) -> ContextOverflowError:
    """Build a DeepAgents-compatible overflow carrying a separate UI notice."""
    error = ContextOverflowError(message)
    return attach_context_notice(error, notice)


def attach_context_notice(error: ContextOverflowError, notice: str) -> ContextOverflowError:
    """Attach a UI notice to an overflow if it does not already have one."""
    existing = str(getattr(error, CONTEXT_NOTICE_ATTR, "") or "").strip()
    if existing:
        return error
    setattr(error, CONTEXT_NOTICE_ATTR, notice)
    set_context_overflow_notice(notice)
    return error


def set_context_overflow_notice(notice: str) -> None:
    """Store a pending context-overflow notice for the renderer."""
    global _fallback_context_notice
    text = str(notice or "").strip()
    if not text:
        return
    _context_notice.set(text)
    _fallback_context_notice = text


def pop_context_overflow_notice(exc: BaseException | None = None) -> str:
    """Return and clear the best available context-overflow notice."""
    global _fallback_context_notice
    notice = str(getattr(exc, CONTEXT_NOTICE_ATTR, "") or "").strip()
    if not notice:
        notice = str(_context_notice.get() or "").strip()
    if not notice:
        notice = str(_fallback_context_notice or "").strip()
    _context_notice.set(None)
    _fallback_context_notice = None
    if notice:
        return notice
    return context_overflow_fallback_notice(exc) if exc is not None else ""


def context_overflow_fallback_notice(exc: BaseException | None = None) -> str:
    """Return a concise fallback notice for unclassified context pressure."""
    text = str(exc or "").lower()
    if "provider" in text or "context limit" in text:
        return PROVIDER_CONTEXT_NOTICE
    return GENERIC_CONTEXT_NOTICE


def mark_context_notice_rendered(exc: BaseException) -> None:
    """Mark an escaped overflow as already rendered by an inner layer."""
    setattr(exc, CONTEXT_NOTICE_RENDERED_ATTR, True)


def context_notice_rendered(exc: BaseException) -> bool:
    """Return whether an escaped overflow already produced a visible notice."""
    return bool(getattr(exc, CONTEXT_NOTICE_RENDERED_ATTR, False))


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
