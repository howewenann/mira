"""Context overflow helpers for DeepAgents model calls."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages.utils import count_tokens_approximately

from runtime.usage import positive_int, usage_from_message

DEFAULT_CONTEXT_PRESSURE_FRACTION = 0.98
MAX_SIGNATURES = 128
CONTEXT_NOTICE_ATTR = "_mira_context_notice"
CONTEXT_NOTICE_RENDERED_ATTR = "_mira_context_notice_rendered"
PROVIDER_CONTEXT_NOTICE = "Provider context limit reached. Compacting older context and retrying."
GENERIC_CONTEXT_NOTICE = "Context limit pressure detected. Compacting older context and retrying."
COMPACTION_SUMMARY_HEADINGS = ("session intent", "summary", "artifacts", "next steps")

_context_notice: ContextVar[str | None] = ContextVar("mira_context_notice", default=None)
_fallback_context_notice: str | None = None


class ContextPressureMiddleware(AgentMiddleware[Any, Any, Any]):
    """Trigger DeepAgents-compatible compaction when context pressure is high."""

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
        self._triggered_signatures: set[str] = set()
        self._skip_next_threads: set[str] = set()

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize provider context errors for sync model calls."""
        self._raise_if_context_is_full(request)
        try:
            return handler(request)
        except ContextOverflowError as exc:
            attach_context_notice(exc, PROVIDER_CONTEXT_NOTICE)
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Normalize provider context errors for async model calls."""
        self._raise_if_context_is_full(request)
        try:
            return await handler(request)
        except ContextOverflowError as exc:
            attach_context_notice(exc, PROVIDER_CONTEXT_NOTICE)
            raise
        except Exception as exc:
            raise_context_overflow_if_detected(exc)
            raise

    def _raise_if_context_is_full(self, request: Any) -> None:
        if not self.enabled or not self.context_limit_tokens:
            return

        thread = thread_id(request)
        if self._consume_retry_skip(thread):
            return

        threshold = max(1, int(self.context_limit_tokens * self.threshold_fraction))
        pressure = self._request_pressure(request, threshold, thread)
        if pressure is None:
            return

        source, tokens, signature = pressure
        if not self._remember_signature(signature):
            return
        notice = configured_threshold_notice(source, tokens, threshold, self.context_limit_tokens)
        self._skip_next_threads.add(thread)
        # DeepAgents' summarization middleware compacts by catching this
        # internal exception, then immediately retries with summarized context.
        raise context_overflow_error("configured context threshold reached", notice)

    def _request_pressure(self, request: Any, threshold: int, thread: str) -> tuple[str, int, str] | None:
        messages = list(getattr(request, "messages", []) or [])
        reported = reported_context_pressure(messages, threshold, thread)
        if reported is not None:
            return reported

        estimated = self._count_request_tokens(request)
        if estimated >= threshold:
            return (
                "estimated",
                estimated,
                f"{thread}:estimated:{len(messages)}:{estimated}",
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
        if signature in self._triggered_signatures:
            return False
        if len(self._triggered_signatures) >= MAX_SIGNATURES:
            self._triggered_signatures.clear()
        self._triggered_signatures.add(signature)
        return True

    def _consume_retry_skip(self, thread: str) -> bool:
        if thread not in self._skip_next_threads:
            return False
        self._skip_next_threads.remove(thread)
        return True


def reported_context_pressure(messages: list[Any], threshold: int, thread: str) -> tuple[str, int, str] | None:
    """Return provider-reported pressure from the newest usage-bearing message."""
    summary_index = newest_compaction_summary_index(messages)
    for index in range(len(messages) - 1, summary_index, -1):
        message = messages[index]
        usage = usage_from_message(message)
        tokens = positive_int(usage.get("context_tokens"))
        if tokens < threshold:
            continue
        source = str(usage.get("source") or "reported")
        fingerprint = message_fingerprint(message, tokens)
        return (source, tokens, f"{thread}:reported:{fingerprint}:{tokens}")
    return None


def newest_compaction_summary_index(messages: list[Any]) -> int:
    """Return the newest message index that marks summarized history."""
    for index in range(len(messages) - 1, -1, -1):
        if is_compaction_summary_message(messages[index]):
            return index
    return -1


def is_compaction_summary_message(message: Any) -> bool:
    """Return whether a message represents DeepAgents summarized context."""
    kwargs = message_field(message, "additional_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization":
        return True

    text = compact_lower_text(message_text(message))
    if not text:
        return False
    return all(heading in text for heading in COMPACTION_SUMMARY_HEADINGS)


def message_fingerprint(message: Any, tokens: int) -> str:
    """Return a stable signature for usage-bearing messages."""
    message_id = str(message_field(message, "id") or "").strip()
    if message_id:
        return message_id

    text = message_text(message)
    if text:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{type(message).__name__}:{digest}"
    return f"{type(message).__name__}:usage:{positive_int(tokens)}"


def message_text(message: Any) -> str:
    """Extract a plain-text message body from common dict/object shapes."""
    text = message_field(message, "text")
    if text is not None:
        return str(text)

    content = message_field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return ""


def message_field(message: Any, name: str) -> Any:
    """Return a dict key or object attribute."""
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def compact_lower_text(text: str) -> str:
    """Normalize text for summary marker checks."""
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def raise_context_overflow_if_detected(exc: Exception) -> None:
    """Raise `ContextOverflowError` when a provider error is really context overflow."""
    if is_context_overflow_error(exc):
        set_context_overflow_notice(PROVIDER_CONTEXT_NOTICE)
        raise context_overflow_error("provider context limit reached", PROVIDER_CONTEXT_NOTICE) from exc


def configured_threshold_notice(source: str, tokens: int, threshold: int, limit: int) -> str:
    """Return the concise user-facing notice for proactive compaction."""
    token_kind = "estimated" if source == "estimated" else "reported"
    return (
        f"Configured context threshold reached: {format_tokens(tokens)} tokens {token_kind}, "
        f"threshold {format_tokens(threshold)}, limit {format_tokens(limit)}. "
        "Compacting before the provider rejects the request."
    )


def format_tokens(value: int) -> str:
    """Return compact token counts for chat notices."""
    tokens = positive_int(value)
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)


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
    """Store a pending context-pressure notice for the renderer."""
    global _fallback_context_notice
    text = str(notice or "").strip()
    if not text:
        return
    _context_notice.set(text)
    _fallback_context_notice = text


def pop_context_overflow_notice(exc: BaseException | None = None) -> str:
    """Return and clear the best available context-pressure notice."""
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
