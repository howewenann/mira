from langgraph.checkpoint.memory import MemorySaver


def make_checkpointer() -> MemorySaver:
    return MemorySaver()
