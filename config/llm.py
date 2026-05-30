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

LEGACY_LMSTUDIO_VARS = {
    "model": "MIRA_LMSTUDIO_MODEL",
    "api_key": "MIRA_LMSTUDIO_API_KEY",
    "base_url": "MIRA_LMSTUDIO_BASE_URL",
}

PROVIDER_SPECIFIC_BLOCKS = {
    "lmstudio": tuple(LEGACY_LMSTUDIO_VARS.values()),
    "ollama": ("MIRA_OLLAMA_MODEL", "MIRA_OLLAMA_API_KEY", "MIRA_OLLAMA_BASE_URL"),
    "openai": ("MIRA_OPENAI_MODEL", "MIRA_OPENAI_API_KEY", "MIRA_OPENAI_BASE_URL"),
    "anthropic": ("MIRA_ANTHROPIC_MODEL", "MIRA_ANTHROPIC_API_KEY", "MIRA_ANTHROPIC_BASE_URL"),
    "claude": ("MIRA_CLAUDE_MODEL", "MIRA_CLAUDE_API_KEY", "MIRA_CLAUDE_BASE_URL"),
    "gemini": ("MIRA_GEMINI_MODEL", "MIRA_GEMINI_API_KEY", "MIRA_GEMINI_BASE_URL"),
    "groq": ("MIRA_GROQ_MODEL", "MIRA_GROQ_API_KEY", "MIRA_GROQ_BASE_URL"),
    "openrouter": ("MIRA_OPENROUTER_MODEL", "MIRA_OPENROUTER_API_KEY", "MIRA_OPENROUTER_BASE_URL"),
}


class ConfigError(ValueError):
    """Raised when environment configuration is ambiguous or invalid."""


def load_llm_config(env: Mapping[str, str]) -> dict[str, Any]:
    """Return normalized LLM config from environment values."""
    provider = _text(env.get(CANONICAL_PROVIDER))
    canonical_values = _canonical_values_present(env)
    provider_blocks = _provider_specific_blocks(env)

    if provider and provider_blocks:
        raise ConfigError(
            "Use either MIRA_LLM_* or provider-specific MIRA_* variables, not both. "
            "Keep one provider block in .env and remove the others."
        )

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

    if provider_blocks:
        if provider_blocks == ["lmstudio"]:
            return _legacy_lmstudio_config(env)

        names = ", ".join(provider_blocks)
        raise ConfigError(
            f"Provider-specific config was found for {names}. "
            "Set MIRA_LLM_PROVIDER and use the canonical MIRA_LLM_* variables instead."
        )

    return _config_for_provider(provider=DEFAULT_PROVIDER, model=None, api_key=None, base_url=None, env=env)


def _config_for_provider(
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    env: Mapping[str, str],
) -> dict[str, Any]:
    provider = _normalize_provider(provider)
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
    }


def _legacy_lmstudio_config(env: Mapping[str, str]) -> dict[str, Any]:
    return _config_for_provider(
        provider=DEFAULT_PROVIDER,
        model=_text(env.get(LEGACY_LMSTUDIO_VARS["model"])),
        api_key=_text(env.get(LEGACY_LMSTUDIO_VARS["api_key"])),
        base_url=_text(env.get(LEGACY_LMSTUDIO_VARS["base_url"])),
        env=env,
    )


def _normalize_provider(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "claude":
        return "anthropic"
    return provider


def _default_model(provider: str) -> str | None:
    return DEFAULT_MODEL if provider == DEFAULT_PROVIDER else None


def _default_api_key(provider: str) -> str | None:
    return DEFAULT_API_KEY if provider == DEFAULT_PROVIDER else None


def _default_base_url(provider: str) -> str | None:
    return DEFAULT_BASE_URL if provider == DEFAULT_PROVIDER else None


def _canonical_values_present(env: Mapping[str, str]) -> bool:
    return any(_text(env.get(name)) for name in (CANONICAL_MODEL, CANONICAL_API_KEY, CANONICAL_BASE_URL))


def _provider_specific_blocks(env: Mapping[str, str]) -> list[str]:
    return [
        provider
        for provider, names in PROVIDER_SPECIFIC_BLOCKS.items()
        if any(_text(env.get(name)) for name in names)
    ]


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
