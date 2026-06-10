"""LLM provider configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_PROVIDER = "lmstudio"
DEFAULT_MODEL = "local-model"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"

CANONICAL_PROVIDER = "MIRA_LLM_PROVIDER"
CANONICAL_MODEL = "MIRA_LLM_MODEL"
CANONICAL_API_KEY = "MIRA_LLM_API_KEY"
CANONICAL_BASE_URL = "MIRA_LLM_BASE_URL"
CANONICAL_TEMPERATURE = "MIRA_LLM_TEMPERATURE"
CANONICAL_MAX_TOKENS = "MIRA_LLM_MAX_TOKENS"
CANONICAL_TOP_P = "MIRA_LLM_TOP_P"
CANONICAL_CONTEXT_TOKENS = "MIRA_LLM_CONTEXT_TOKENS"


class ConfigError(ValueError):
    """Raised when environment configuration is ambiguous or invalid."""


def load_llm_config(env: Mapping[str, str]) -> dict[str, Any]:
    """Return normalized LLM config from environment values."""
    provider = _text(env.get(CANONICAL_PROVIDER))
    canonical_values = _canonical_values_present(env)

    if canonical_values and not provider:
        raise ConfigError(
            "MIRA_LLM_PROVIDER is required when using MIRA_LLM_MODEL, "
            "MIRA_LLM_API_KEY, or MIRA_LLM_BASE_URL."
        )

    if provider:
        return _config_for_provider(
            provider=provider,
            model=_text(env.get(CANONICAL_MODEL)),
            api_key=_text(env.get(CANONICAL_API_KEY)),
            base_url=_text(env.get(CANONICAL_BASE_URL)),
            env=env,
        )

    return _config_for_provider(provider=DEFAULT_PROVIDER, model=None, api_key=None, base_url=None, env=env)


def _config_for_provider(
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    env: Mapping[str, str],
) -> dict[str, Any]:
    provider = provider.strip().lower()
    model = model or _default_model(provider)

    if not model:
        raise ConfigError(f"MIRA_LLM_MODEL is required for provider '{provider}'.")

    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_key": api_key if api_key is not None else _default_api_key(provider),
        "llm_base_url": base_url if base_url is not None else _default_base_url(provider),
        "llm_temperature": _float_value(env, CANONICAL_TEMPERATURE),
        "llm_max_tokens": _int_value(env, CANONICAL_MAX_TOKENS),
        "llm_top_p": _float_value(env, CANONICAL_TOP_P),
        "llm_context_tokens": _int_value(env, CANONICAL_CONTEXT_TOKENS),
    }


def _default_model(provider: str) -> str | None:
    return DEFAULT_MODEL if provider == DEFAULT_PROVIDER else None


def _default_api_key(provider: str) -> str | None:
    return DEFAULT_API_KEY if provider == DEFAULT_PROVIDER else None


def _default_base_url(provider: str) -> str | None:
    return DEFAULT_BASE_URL if provider == DEFAULT_PROVIDER else None


def _canonical_values_present(env: Mapping[str, str]) -> bool:
    return any(_text(env.get(name)) for name in (CANONICAL_MODEL, CANONICAL_API_KEY, CANONICAL_BASE_URL))


def _float_value(env: Mapping[str, str], name: str) -> float | None:
    value = _text(env.get(name))
    if value is None:
        return None

    try:
        return float(value)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number.") from error


def _int_value(env: Mapping[str, str], name: str) -> int | None:
    value = _text(env.get(name))
    if value is None:
        return None

    try:
        return int(value)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer.") from error


def _text(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None
