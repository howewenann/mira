from langchain_anyllm import ChatAnyLLM
import lmstudio as lms

chat = ChatAnyLLM(
    provider="lmstudio",
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    model="google/gemma-4-e4b"
)

lmstudio_model = lms.llm()
context_size = lmstudio_model.get_context_length()

print("LM Studio context size:", context_size)