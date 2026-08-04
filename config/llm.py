"""LLM provider configuration helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


DEFAULT_PROVIDER = "lmstudio"
DEFAULT_MODEL = "local-model"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_CONTEXT_TOKENS = 32768

CANONICAL_PROVIDER = "MIRA_LLM_PROVIDER"
CANONICAL_MODEL = "MIRA_LLM_MODEL"
CANONICAL_API_KEY = "MIRA_LLM_API_KEY"
CANONICAL_BASE_URL = "MIRA_LLM_BASE_URL"
CANONICAL_TEMPERATURE = "MIRA_LLM_TEMPERATURE"
CANONICAL_MAX_TOKENS = "MIRA_LLM_MAX_TOKENS"
CANONICAL_TOP_P = "MIRA_LLM_TOP_P"
CANONICAL_CONTEXT_TOKENS = "MIRA_LLM_CONTEXT_TOKENS"
CANONICAL_MODEL_KWARGS = "MIRA_LLM_MODEL_KWARGS"

RUBRIC_PROVIDER = "MIRA_RUBRIC_LLM_PROVIDER"
RUBRIC_MODEL = "MIRA_RUBRIC_LLM_MODEL"
RUBRIC_API_KEY = "MIRA_RUBRIC_LLM_API_KEY"
RUBRIC_BASE_URL = "MIRA_RUBRIC_LLM_BASE_URL"
RUBRIC_TEMPERATURE = "MIRA_RUBRIC_LLM_TEMPERATURE"
RUBRIC_MAX_TOKENS = "MIRA_RUBRIC_LLM_MAX_TOKENS"
RUBRIC_TOP_P = "MIRA_RUBRIC_LLM_TOP_P"
RUBRIC_CONTEXT_TOKENS = "MIRA_RUBRIC_LLM_CONTEXT_TOKENS"
RUBRIC_MODEL_KWARGS = "MIRA_RUBRIC_LLM_MODEL_KWARGS"

_RUBRIC_NAMES = (
    RUBRIC_PROVIDER,
    RUBRIC_MODEL,
    RUBRIC_API_KEY,
    RUBRIC_BASE_URL,
    RUBRIC_TEMPERATURE,
    RUBRIC_MAX_TOKENS,
    RUBRIC_TOP_P,
    RUBRIC_CONTEXT_TOKENS,
    RUBRIC_MODEL_KWARGS,
)
_RESERVED_MODEL_KWARGS = {
    "model",
    "model_id",
    "provider",
    "api_key",
    "api_base",
    "base_url",
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "stream",
    "streaming",
    "stream_options",
    "disable_streaming",
    "client_args",
    "client",
    "async_client",
}


class ConfigError(ValueError):
    """Raised when environment configuration is ambiguous or invalid."""


def load_llm_config(env: Mapping[str, str]) -> dict[str, Any]:
    """Return normalized main and rubric LLM profiles."""
    provider = _text(env.get(CANONICAL_PROVIDER))
    canonical_values = _canonical_values_present(env)

    if canonical_values and not provider:
        raise ConfigError(
            "MIRA_LLM_PROVIDER is required when using MIRA_LLM_MODEL, "
            "MIRA_LLM_API_KEY, or MIRA_LLM_BASE_URL."
        )

    main = _config_for_provider(
        provider=provider or DEFAULT_PROVIDER,
        model=_text(env.get(CANONICAL_MODEL)),
        api_key=_text(env.get(CANONICAL_API_KEY)),
        base_url=_text(env.get(CANONICAL_BASE_URL)),
        temperature=_float_value(env, CANONICAL_TEMPERATURE),
        max_tokens=_int_value(env, CANONICAL_MAX_TOKENS),
        top_p=_float_value(env, CANONICAL_TOP_P),
        context_tokens=_int_value(env, CANONICAL_CONTEXT_TOKENS),
        model_kwargs=_model_kwargs(env, CANONICAL_MODEL_KWARGS),
        require_model=bool(provider),
        prefix="llm",
        model_variable=CANONICAL_MODEL,
    )
    return {**main, **_rubric_config(env, main)}


def _rubric_config(env: Mapping[str, str], main: dict[str, Any]) -> dict[str, Any]:
    overridden = any(_text(env.get(name)) is not None for name in _RUBRIC_NAMES)
    rubric_provider = (_text(env.get(RUBRIC_PROVIDER)) or str(main["llm_provider"])).lower()
    same_provider = rubric_provider == main["llm_provider"]

    rubric_kwargs = _model_kwargs(env, RUBRIC_MODEL_KWARGS)
    if same_provider:
        merged_kwargs = {**main["llm_model_kwargs"], **rubric_kwargs}
        values = {
            "rubric_llm_provider": rubric_provider,
            "rubric_llm_model": _text(env.get(RUBRIC_MODEL)) or main["llm_model"],
            "rubric_llm_api_key": _text(env.get(RUBRIC_API_KEY)) or main["llm_api_key"],
            "rubric_llm_base_url": _text(env.get(RUBRIC_BASE_URL)) or main["llm_base_url"],
            "rubric_llm_temperature": _inherited_float(env, RUBRIC_TEMPERATURE, main["llm_temperature"]),
            "rubric_llm_max_tokens": _inherited_int(env, RUBRIC_MAX_TOKENS, main["llm_max_tokens"]),
            "rubric_llm_top_p": _inherited_float(env, RUBRIC_TOP_P, main["llm_top_p"]),
            "rubric_llm_context_tokens": _inherited_int(
                env, RUBRIC_CONTEXT_TOKENS, main["llm_context_tokens"]
            ),
            "rubric_llm_model_kwargs": merged_kwargs,
        }
    else:
        model = _text(env.get(RUBRIC_MODEL))
        if not model:
            raise ConfigError(f"{RUBRIC_MODEL} is required for provider '{rubric_provider}'.")
        values = {
            "rubric_llm_provider": rubric_provider,
            "rubric_llm_model": model,
            "rubric_llm_api_key": _text(env.get(RUBRIC_API_KEY)) or _default_api_key(rubric_provider),
            "rubric_llm_base_url": _text(env.get(RUBRIC_BASE_URL)) or _default_base_url(rubric_provider),
            "rubric_llm_temperature": _float_value(env, RUBRIC_TEMPERATURE),
            "rubric_llm_max_tokens": _int_value(env, RUBRIC_MAX_TOKENS),
            "rubric_llm_top_p": _float_value(env, RUBRIC_TOP_P),
            "rubric_llm_context_tokens": _int_value(env, RUBRIC_CONTEXT_TOKENS) or DEFAULT_CONTEXT_TOKENS,
            "rubric_llm_model_kwargs": rubric_kwargs,
        }
    values["rubric_llm_overridden"] = overridden
    return values


def _config_for_provider(
    *,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    context_tokens: int | None,
    model_kwargs: dict[str, Any],
    require_model: bool,
    prefix: str,
    model_variable: str,
) -> dict[str, Any]:
    provider = provider.strip().lower()
    model = model or (None if require_model and provider != DEFAULT_PROVIDER else _default_model(provider))
    if not model:
        raise ConfigError(f"{model_variable} is required for provider '{provider}'.")

    return {
        f"{prefix}_provider": provider,
        f"{prefix}_model": model,
        f"{prefix}_api_key": api_key if api_key is not None else _default_api_key(provider),
        f"{prefix}_base_url": base_url if base_url is not None else _default_base_url(provider),
        f"{prefix}_temperature": temperature,
        f"{prefix}_max_tokens": max_tokens,
        f"{prefix}_top_p": top_p,
        f"{prefix}_context_tokens": context_tokens or DEFAULT_CONTEXT_TOKENS,
        f"{prefix}_model_kwargs": model_kwargs,
    }


def _model_kwargs(env: Mapping[str, str], name: str) -> dict[str, Any]:
    raw = _text(env.get(name))
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{name} must be a valid JSON object.") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object.")
    reserved = sorted(str(key) for key in value if str(key).lower() in _RESERVED_MODEL_KWARGS)
    if reserved:
        raise ConfigError(f"{name} cannot set runtime-owned key(s): {', '.join(reserved)}.")
    return value


def _default_model(provider: str) -> str | None:
    return DEFAULT_MODEL if provider == DEFAULT_PROVIDER else None


def _default_api_key(provider: str) -> str | None:
    return DEFAULT_API_KEY if provider == DEFAULT_PROVIDER else None


def _default_base_url(provider: str) -> str | None:
    return DEFAULT_BASE_URL if provider == DEFAULT_PROVIDER else None


def _canonical_values_present(env: Mapping[str, str]) -> bool:
    return any(_text(env.get(name)) for name in (CANONICAL_MODEL, CANONICAL_API_KEY, CANONICAL_BASE_URL))


def _inherited_float(env: Mapping[str, str], name: str, fallback: float | None) -> float | None:
    return fallback if _text(env.get(name)) is None else _float_value(env, name)


def _inherited_int(env: Mapping[str, str], name: str, fallback: int | None) -> int | None:
    return fallback if _text(env.get(name)) is None else _int_value(env, name)


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
