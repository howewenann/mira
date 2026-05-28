from langchain_anyllm import ChatAnyLLM


def get_llm(config: dict):
    return ChatAnyLLM(
        model=config["lmstudio_model"],
        provider="lmstudio",
        api_base=config["lmstudio_base_url"],
        api_key=config["lmstudio_api_key"],
    )


def get_model_name(config: dict) -> str:
    return config["lmstudio_model"]
