from langchain_openai import ChatOpenAI
from langchain_anyllm import ChatAnyLLM


def get_llm(config: dict):

    # LM Studio example:
    # return ChatOpenAI(
    #     model=config["lmstudio_model"],
    #     base_url=config["lmstudio_base_url"],
    #     api_key=config["lmstudio_api_key"],
    # )

    return ChatAnyLLM(
        model=config["lmstudio_model"],
        provider='lmstudio',
        api_base=config["lmstudio_base_url"],
        api_key=config["lmstudio_api_key"],
    )

    # Anthropic example:
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(model="claude-sonnet-4-5")

    # Ollama example:
    # from langchain_ollama import ChatOllama
    # return ChatOllama(model="qwen2.5-coder")

    # LlamaCpp example:
    # from langchain_community.chat_models import ChatLlamaCpp
    # return ChatLlamaCpp(model_path="models/local.gguf")


def get_model_name(config: dict) -> str:
    if config["model"]:
        return config["model"]

    return config["lmstudio_model"]
