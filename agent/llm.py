"""LangChain model construction helpers."""

from __future__ import annotations

from typing import Any

from langchain_anyllm import ChatAnyLLM

STREAM_USAGE_PROVIDERS = {"lmstudio"}


def get_llm(config: dict[str, Any]) -> ChatAnyLLM:
    """Create the LangChain chat model from MIRA's config dictionary."""
    kwargs: dict[str, Any] = {
        "model": config["llm_model"],
        "provider": config["llm_provider"],
    }

    optional_values = {
        "api_base": config.get("llm_base_url"),
        "api_key": config.get("llm_api_key"),
        "temperature": config.get("llm_temperature"),
        "max_tokens": config.get("llm_max_tokens"),
        "top_p": config.get("llm_top_p"),
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    if str(config.get("llm_provider") or "").lower() in STREAM_USAGE_PROVIDERS:
        kwargs["stream_options"] = {"include_usage": True}
    if config.get("llm_insecure_direct"):
        import httpx

        from agent.llm_httpx import ChatAnyLLMWithHttpx

        return ChatAnyLLMWithHttpx(
            **kwargs,
            http_client=httpx.Client(trust_env=False, verify=False),
            async_http_client=httpx.AsyncClient(trust_env=False, verify=False),
        )
    return ChatAnyLLM(**kwargs)


def get_model_name(config: dict[str, Any]) -> str:
    """Return the configured display name for the model."""
    return f"{config['llm_provider']}:{config['llm_model']}"
