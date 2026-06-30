"""LangChain model construction helpers."""

from __future__ import annotations

from typing import Any

from langchain_anyllm import ChatAnyLLM

from config.metadata import ModelMetadata, apply_model_metadata

STREAM_USAGE_PROVIDERS = {"lmstudio"}
OPENAI_COMPAT_TRANSPORT_PROVIDERS = {"lmstudio"}


def get_llm(config: dict[str, Any], metadata: ModelMetadata | None = None) -> ChatAnyLLM:
    """Create the LangChain chat model from MIRA's config dictionary."""
    kwargs = chat_anyllm_transport_kwargs(config)

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
    if config.get("llm_direct"):
        import httpx

        kwargs["model_kwargs"] = {
            "client_args": {
                "http_client": httpx.AsyncClient(trust_env=False, verify=False),
            }
        }
    model = ChatAnyLLM(**kwargs)
    if metadata is not None:
        return apply_model_metadata(model, metadata)
    fallback = config.get("llm_inferred_context_tokens") or config.get("llm_context_tokens")
    return apply_model_metadata(model, ModelMetadata(context_tokens=fallback))


def chat_anyllm_transport_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Return ChatAnyLLM transport kwargs while preserving MIRA provider identity."""
    provider = str(config["llm_provider"]).lower()
    transport_provider = "openai" if provider in OPENAI_COMPAT_TRANSPORT_PROVIDERS else config["llm_provider"]
    return {
        "model": config["llm_model"],
        "provider": transport_provider,
    }


def get_model_name(config: dict[str, Any]) -> str:
    """Return the configured display name for the model."""
    return f"{config['llm_provider']}:{config['llm_model']}"
