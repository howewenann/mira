"""Agent package for model, policy, and DeepAgents wiring."""

import os

# Set before any agent submodule can import AnyLLM/OpenAI model classes. Their
# streamed tool-call objects otherwise retain deferred serializers that
# LangSmith cannot serialize when completing a model run.
os.environ.setdefault("DEFER_PYDANTIC_BUILD", "false")
