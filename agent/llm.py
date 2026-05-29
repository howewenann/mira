from __future__ import annotations

from typing import Any

from langchain_anyllm import ChatAnyLLM


def get_llm(config: dict[str, Any]) -> ChatAnyLLM:
    """Create the LangChain chat model from MIRA's config dictionary."""
    return ChatAnyLLM(
        model=config["lmstudio_model"],
        provider="lmstudio",
        api_base=config["lmstudio_base_url"],
        api_key=config["lmstudio_api_key"],
    )


def get_model_name(config: dict[str, Any]) -> str:
    """Return the configured display name for the model."""
    return str(config["lmstudio_model"])
