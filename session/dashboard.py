"""Session dashboard stats and context-limit helpers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately

DEFAULT_DASHBOARD = {
    "model": "",
    "context": {
        "used_tokens": 0,
        "limit_tokens": 0,
        "percent": 0.0,
        "source": "unknown",
    },
    "tokens": {
        "in": 0,
        "out": 0,
    },
    "duration_seconds": 0,
}


def normalize_dashboard(value: Any) -> dict[str, Any]:
    """Return a dashboard object with the stable persisted shape."""
    dashboard = deepcopy(DEFAULT_DASHBOARD)
    if not isinstance(value, dict):
        return dashboard

    dashboard["model"] = str(value.get("model") or "")

    context = value.get("context")
    if isinstance(context, dict):
        used = positive_int(context.get("used_tokens"))
        limit = positive_int(context.get("limit_tokens"))
        dashboard["context"] = {
            "used_tokens": used,
            "limit_tokens": limit,
            "percent": context_percent(used, limit),
            "source": str(context.get("source") or "unknown"),
        }

    tokens = value.get("tokens")
    if isinstance(tokens, dict):
        dashboard["tokens"] = {
            "in": positive_int(tokens.get("in")),
            "out": positive_int(tokens.get("out")),
        }

    dashboard["duration_seconds"] = positive_int(value.get("duration_seconds"))
    return dashboard


def apply_turn_usage(
    record: dict[str, Any],
    result: Any,
    *,
    model_name: str = "",
    context_limit_tokens: int | None = None,
    context_limit_source: str = "unknown",
) -> dict[str, Any]:
    """Add one turn's usage metadata to a session dashboard."""
    dashboard = ensure_dashboard(
        record,
        model_name=model_name,
        context_limit_tokens=context_limit_tokens,
        context_limit_source=context_limit_source,
    )
    usage = result_usage(result)

    input_tokens = positive_int(usage.get("input_tokens"))
    output_tokens = positive_int(usage.get("output_tokens"))
    context_tokens = positive_int(usage.get("context_tokens")) or input_tokens

    dashboard["tokens"]["in"] += input_tokens
    dashboard["tokens"]["out"] += output_tokens

    context = dashboard["context"]
    if context_tokens:
        context["used_tokens"] = context_tokens
        if context.get("source") == "unknown":
            context["source"] = str(usage.get("source") or "usage_metadata")

    if context_limit_tokens:
        context["limit_tokens"] = positive_int(context_limit_tokens)
        if context_limit_source and context_limit_source != "unknown":
            context["source"] = context_limit_source

    context["percent"] = context_percent(context["used_tokens"], context["limit_tokens"])
    update_duration(record)
    return record["dashboard"]


def ensure_dashboard(
    record: dict[str, Any],
    *,
    model_name: str = "",
    context_limit_tokens: int | None = None,
    context_limit_source: str = "unknown",
) -> dict[str, Any]:
    """Ensure an in-memory session has dashboard data for display."""
    dashboard = normalize_dashboard(record.get("dashboard"))
    if model_name:
        dashboard["model"] = model_name

    if context_limit_tokens:
        dashboard["context"]["limit_tokens"] = positive_int(context_limit_tokens)
        dashboard["context"]["percent"] = context_percent(
            dashboard["context"]["used_tokens"],
            dashboard["context"]["limit_tokens"],
        )
        if context_limit_source and context_limit_source != "unknown":
            dashboard["context"]["source"] = context_limit_source

    record["dashboard"] = dashboard
    update_duration(record)
    return record["dashboard"]


def update_duration(record: dict[str, Any], now: datetime | None = None) -> int:
    """Update and return the dashboard duration in whole seconds."""
    dashboard = normalize_dashboard(record.get("dashboard"))
    created_at = parse_datetime(record.get("created_at"))
    if created_at is None:
        duration = dashboard["duration_seconds"]
    else:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        duration = max(0, int((current - created_at).total_seconds()))

    dashboard["duration_seconds"] = duration
    record["dashboard"] = dashboard
    return duration


def context_limit_for_model(model: Any) -> tuple[int | None, str]:
    """Return the model context limit MIRA should display."""
    profile_limit = model_profile_limit(model)
    if profile_limit:
        return profile_limit, "model_profile.max_input_tokens"

    trigger = deepagents_compaction_trigger(model)
    if trigger is None:
        return None, "unknown"

    kind, value = trigger
    if kind == "tokens":
        limit = positive_int(value)
        return (limit, "deepagents.compaction_trigger") if limit else (None, "unknown")

    return None, "unknown"


def model_profile_limit(model: Any) -> int | None:
    """Return `model.profile.max_input_tokens` when a LangChain model exposes it."""
    try:
        profile = model.profile
    except AttributeError:
        return None

    if not isinstance(profile, dict):
        return None

    limit = positive_int(profile.get("max_input_tokens"))
    return limit or None


def deepagents_compaction_trigger(model: Any) -> tuple[str, Any] | None:
    """Return DeepAgents' auto-compaction trigger for a model."""
    try:
        from deepagents.middleware.summarization import compute_summarization_defaults

        trigger = compute_summarization_defaults(model)["trigger"]
    except Exception:
        return None

    if isinstance(trigger, tuple) and len(trigger) == 2:
        return trigger
    return None


def token_counter_for_model(model: Any | None = None) -> Callable[[str], int]:
    """Return a model-independent LangChain approximate token counter."""

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return positive_int(
            count_tokens_approximately(
                [{"role": "user", "content": text}],
                use_usage_metadata_scaling=False,
            )
        )

    return count_tokens


def result_usage(result: Any) -> dict[str, Any]:
    """Extract normalized usage from a TurnResult-like object."""
    usage = getattr(result, "usage", None)
    if isinstance(usage, dict):
        return usage

    return {
        "input_tokens": positive_int(getattr(result, "input_tokens", 0)),
        "output_tokens": positive_int(getattr(result, "output_tokens", 0)),
        "context_tokens": positive_int(getattr(result, "context_tokens", 0)),
        "source": str(getattr(result, "usage_source", "unknown") or "unknown"),
    }


def context_percent(used_tokens: int, limit_tokens: int) -> float:
    """Return context usage percent rounded to one decimal place."""
    if limit_tokens <= 0:
        return 0.0
    return round(min(999.9, (used_tokens / limit_tokens) * 100), 1)


def positive_int(value: Any) -> int:
    """Return a non-negative integer from loose provider metadata."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def parse_datetime(value: Any) -> datetime | None:
    """Parse a session timestamp into an aware datetime."""
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
